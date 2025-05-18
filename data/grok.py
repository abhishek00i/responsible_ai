import torch
from typing import Optional, List, Dict, Any, AsyncIterator, Union
from langchain_core.language_models import LLM
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.outputs import Generation, LLMResult
from langchain_core.messages import BaseMessage, AIMessage
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList
import json
import re

class StopOnTokens(StoppingCriteria):
    def __init__(self, stop_token_ids: List[int]):
        self.stop_token_ids = stop_token_ids

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        last_token = input_ids[0, -1].item()
        return last_token in self.stop_token_ids

class CustomHuggingFaceLLM(LLM):
    model: Optional[Any] = None
    tokenizer: Optional[Any] = None
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    tools: List[Dict[str, Any]] = []

    def __init__(self, model_path: str, **kwargs):
        super().__init__(**kwargs)
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto"
            )
            self.model.eval()

            if self.tokenizer.pad_token_id is None:
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
                self.tokenizer.pad_token = self.tokenizer.eos_token
        except Exception as e:
            raise ValueError(f"Failed to load model or tokenizer from {model_path}: {str(e)}")

    @property
    def _llm_type(self) -> str:
        return "custom_huggingface_llm"

    def _generate(
        self,
        prompts: List[str],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs
    ) -> LLMResult:
        generations = []
        for prompt in prompts:
            try:
                # Wrap prompt with ReAct format
                react_prompt = self._create_react_prompt(prompt)
                text = self._call(react_prompt, stop=stop, run_manager=run_manager, **kwargs)
                parsed_output = self._parse_react_output(text)
                generations.append([Generation(text=json.dumps(parsed_output))])
            except Exception as e:
                generations.append([Generation(text=f"Error: {str(e)}")])
        return LLMResult(generations=generations)

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs
    ) -> str:
        if not prompt or not isinstance(prompt, str):
            return ""

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            return_attention_mask=True
        ).to(self.device)
        input_length = inputs["input_ids"].shape[1]

        generate_kwargs = {
            "max_new_tokens": kwargs.get("max_new_tokens", 512),
            "num_return_sequences": 1,
            "do_sample": True,
            "top_k": 50,
            "top_p": 0.95,
            "temperature": kwargs.get("temperature", 0.9),
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "no_repeat_ngram_size": 2,
            "min_length": 1,
        }
        generate_kwargs.update(kwargs)

        stopping_criteria = None
        if stop:
            stop_token_ids = []
            for stop_token in stop:
                token_ids = self.tokenizer.encode(stop_token, add_special_tokens=False)
                if token_ids:
                    stop_token_ids.extend(token_ids)
            if stop_token_ids:
                stopping_criteria = StoppingCriteriaList([StopOnTokens(stop_token_ids)])
                generate_kwargs["stopping_criteria"] = stopping_criteria

        try:
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs["input_ids"],
                    attention_mask=inputs.get("attention_mask"),
                    **generate_kwargs
                )
            generated_tokens = outputs[0, input_length:]
            generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
            if stop and not stopping_criteria:
                for stop_token in stop:
                    generated_text = generated_text.split(stop_token)[0]
            generated_text = generated_text.replace("assistant", "").strip()
            return generated_text
        except Exception as e:
            return f"Error during generation: {str(e)}"

    async def _astream(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs
    ) -> AsyncIterator[str]:
        if not prompt or not isinstance(prompt, str):
            yield ""
            return

        react_prompt = self._create_react_prompt(prompt)
        inputs = self.tokenizer(
            react_prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            return_attention_mask=True
        ).to(self.device)
        input_length = inputs["input_ids"].shape[1]

        generate_kwargs = {
            "max_new_tokens": kwargs.get("max_new_tokens", 512),
            "do_sample": True,
            "top_k": 50,
            "top_p": 0.95,
            "temperature": kwargs.get("temperature", 0.9),
            "pad_token_id": self.tokenizer.pad_token_id,
            "no_repeat_ngram_size": 2,
            "min_length": 1,
        }
        generate_kwargs.update(kwargs)

        try:
            with torch.no_grad():
                for output in self.model.generate(
                    inputs["input_ids"],
                    attention_mask=inputs.get("attention_mask"),
                    **generate_kwargs
                ):
                    generated_tokens = output[input_length:]
                    chunk = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
                    chunk = chunk.replace("assistant", "").strip()
                    if stop and any(stop_token in chunk for stop_token in stop):
                        break
                    parsed_chunk = self._parse_react_output(chunk)
                    if run_manager:
                        await run_manager.on_llm_new_token(json.dumps(parsed_chunk))
                    yield json.dumps(parsed_chunk)
        except Exception as e:
            yield f"Error during streaming: {str(e)}"

    def bind_tools(self, tools: List[Dict[str, Any]], **kwargs) -> "CustomHuggingFaceLLM":
        self.tools = tools
        return self

    def _create_react_prompt(self, prompt: str) -> str:
        tool_desc = "\n".join(
            f"- {tool.get('name')}: {tool.get('description', '')}"
            for tool in self.tools
        ) if self.tools else "Python REPL: Execute Python code to process CSV data."
        
        react_template = f"""You are an agent that processes CSV data using a ReAct (Reasoning and Acting) approach. For the given query, provide a response in the following JSON format:

```json
{{
  "action": "python",
  "action_input": "<Python code or query result>"
}}
```

Available tools:
{tool_desc}

Query: {prompt}

Follow these steps:
1. Reason about the query.
2. If the query requires data processing, write Python code to execute in the Python REPL.
3. If the query is a simple question, provide the answer directly in action_input.
4. Ensure the output is valid JSON with 'action' and 'action_input' fields.

Response:
"""
        return react_template

    def _parse_react_output(self, text: str) -> Dict[str, Any]:
        # Try to parse as JSON
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            # Fallback: Extract action and action_input using regex
            action_match = re.search(r'"action"\s*:\s*"([^"]+)"', text)
            action_input_match = re.search(r'"action_input"\s*:\s*"([^"]+)"', text)
            
            if action_match and action_input_match:
                return {
                    "action": action_match.group(1),
                    "action_input": action_input_match.group(1)
                }
            
            # If parsing fails, return a default Python action
            return {
                "action": "python",
                "action_input": f"print('{text}')"
            }

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        return {
            "model_path": self.model.config._name_or_path if self.model else "unknown",
            "device": self.device,
            "llm_type": self._llm_type,
            "tools": [tool.get("name") for tool in self.tools]
        }

# Example instantiation
if __name__ == "__main__":
    model_path = "/exacto/abhishek/responsible_ai/models"
    try:
        phi4_llm = CustomHuggingFaceLLM(model_path=model_path)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Failed to load model: {str(e)}")
