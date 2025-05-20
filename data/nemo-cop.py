from typing import Optional, Tuple, List, Dict, Any
import regex as re
from nemoguardrails import LLMRails, RailsConfig
from nemoguardrails.actions import action
from nemoguardrails.llm.task import Task
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline as hf_pipeline
from sentence_transformers import SentenceTransformer, util
import numpy as np
import asyncio
from langchain.base_language import BaseLanguageModel
from langchain_core.prompts import PromptTemplate
import os

# --- Your pipeline code (Phi-4 model) ---
MODEL_PATH = os.getenv("MODEL_PATH")

phi4_pipeline = hf_pipeline(
    "text-generation",
    model=MODEL_PATH,
    model_kwargs={"torch_dtype": "auto"},
    device_map="auto",
)

def call_llm(question): 
    print("---> GenAI call Initiated")
    text_data= retrieve_context(question)
    try:
        anonymizer = PIIAnonymizer(context = text_data, question = question)
        text_data, question = anonymizer.main()
    except :
        pass
        
    messages=[
        {"role": "assistant", "content": f"""You are a helpful asssistant that provide specific answer 
        only in minimal words from the provided context of the document.
        Note :- Extract only the requested information. Do not include any extra word or explanation."""},

        {"role": "user", "content": f"""Question:{question} /nContext:{text_data}"""}
    ]
    outputs = phi4_pipeline(messages, max_new_tokens=128)
    print('question--->',question)
    responseOutput=outputs[0]["generated_text"][-1]['content']
    print("Response--->",responseOutput)
    return(responseOutput)

# --- End of pipeline code ---

# --- Adapter for NeMo Guardrails ---
class Phi4LLM(BaseLanguageModel):
    """
    Adapter to wrap the HuggingFace pipeline for NeMo Guardrails.
    Only implements agenerate_prompt as required by guardrails (for intent parsing etc).
    """
    def __init__(self, pipeline):
        self.pipeline = pipeline

    async def agenerate_prompt(
        self,
        prompts: List[PromptTemplate],
        stop: Optional[List[str]] = None,
        **kwargs: Any
    ) -> Any:
        # Take only the first prompt (Guardrails usually sends one at a time)
        prompt_text = prompts[0].format() if prompts else ""
        # Use pipeline to generate output (synchronously, since pipeline isn't async)
        # If needed, run in executor to prevent blocking event loop
        import asyncio
        loop = asyncio.get_running_loop()
        def run_pipeline(prompt):
            # The pipeline expects a list of messages but for intent parsing, only the user prompt is needed
            # You may customize the system/user prompt format if your model expects it
            messages = [
                {"role": "user", "content": prompt}
            ]
            result = self.pipeline(messages, max_new_tokens=64)
            # Fallback: return the first string found
            try:
                if isinstance(result, list) and isinstance(result[0], dict):
                    return result[0].get("generated_text", "")
                else:
                    return str(result)
            except Exception:
                return str(result)
        generated_text = await loop.run_in_executor(None, run_pipeline, prompt_text)
        # Guardrails expects a .generations field with a list-of-lists of dicts with "text"
        return type('Result', (), {
            'generations': [[{'text': generated_text}]],
            'llm_output': {}
        })()

    async def generate(self, messages: List[List[Dict[str, str]]], stop: Optional[List[str]] = None, **kwargs: Any) -> Any:
        raise NotImplementedError("Only agenerate_prompt is supported for NeMo Guardrails flows.")

    def get_num_tokens(self, text: str) -> int:
        return len(text.split())

# --- The rest of your guardrails processor code (unchanged except for DummyLLM -> Phi4LLM) ---

# (custom actions omitted for brevity: use as in your original code)

class GuardrailsProcessor:
    def __init__(self):
        """
        Initialize the Guardrails Processor using NeMo Guardrails.
        """
        try:
            # Initialize toxicity detection model
            global toxicity_classifier
            model = AutoModelForSequenceClassification.from_pretrained("/Users/abhishek/Downloads/responsible_ai/models/share/unbaised-toxic-roberta")
            tokenizer = AutoTokenizer.from_pretrained("/Users/abhishek/Downloads/responsible_ai/models/share/unbaised-toxic-roberta")
            toxicity_classifier = hf_pipeline("text-classification", model=model, tokenizer=tokenizer, device=-1)  # device=-1 forces CPU
            print("Initialized toxicity detection model")

            # Initialize off-topic detection model
            global sentence_encoder
            sentence_encoder = SentenceTransformer("/Users/abhishek/Downloads/responsible_ai/models/share/all-MIniLM-L6-v2", device='cpu')
            print("Initialized off-topic detection model")

            # Define NeMo Guardrails configuration using Colang
            colang_content = """
            define user validate input
              "validate_input: *"

            define user validate off topic
              "validate_off_topic: *"

            define user validate output
              "validate_output: *"

            define flow validate_input
              user validate input
              $text = $last_user_message
              $toxicity_ok = await detect_toxicity($text)
              if not $toxicity_ok
                bot say "Error: Input contains toxic content."
                stop

              $prompt_injection_ok = await detect_prompt_injection($text)
              if not $prompt_injection_ok
                bot say "Error: Input contains prompt injection attempt."
                stop

              $jailbreak_ok = await detect_jailbreak($text)
              if not $jailbreak_ok
                bot say "Error: Input contains jailbreak attempt."
                stop

              $code_injection_ok = await detect_code_injection($text)
              if not $code_injection_ok
                bot say "Error: Input contains code injection attempt."
                stop

              bot say "Input validation passed."

            define flow validate_off_topic
              user validate off topic
              $text = $last_user_message
              $document_context = $context
              $off_topic_ok = await detect_off_topic($text, $document_context)
              if not $off_topic_ok
                bot say "Error: Query is off-topic relative to the document context."
                stop

              bot say "Off-topic validation passed."

            define flow validate_output
              user validate output
              $text = $last_user_message
              $toxicity_ok = await detect_toxicity($text)
              if not $toxicity_ok
                bot say "Error: Output contains toxic content."
                stop

              $code_injection_ok = await detect_code_injection($text)
              if not $code_injection_ok
                bot say "Error: Output contains code injection attempt."
                stop

              bot say "Output validation passed."
            """
            config = RailsConfig.from_content(colang_content=colang_content)
            
            # Use YOUR Phi4 pipeline for NeMo guardrails
            self.guardrails = LLMRails(config, llm=Phi4LLM(phi4_pipeline))

            # Register custom actions
            self.guardrails.register_action(detect_toxicity)
            self.guardrails.register_action(detect_prompt_injection)
            self.guardrails.register_action(detect_jailbreak)
            self.guardrails.register_action(detect_off_topic)
            self.guardrails.register_action(detect_code_injection)
            print("Initialized NeMo Guardrails for input and output validation")

        except Exception as e:
            print(f"Failed to initialize GuardrailsProcessor: {str(e)}")
            raise

    # ... (rest of GuardrailsProcessor unchanged)
    # Your validate_input, validate_off_topic, validate_output, process, main, etc.

# (main() and entrypoint unchanged; use your original code)
