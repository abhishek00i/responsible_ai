from typing import Optional, Tuple, List, Dict, Any
import regex as re
from nemoguardrails import LLMRails, RailsConfig
from nemoguardrails.actions import action
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline as hf_pipeline
from sentence_transformers import SentenceTransformer, util
import numpy as np
import asyncio

# ---- Adapter for HuggingFace pipeline for NeMo Guardrails ----
class Phi4LLM:
    def __init__(self, pipeline):
        self.pipeline = pipeline

    async def agenerate_prompt(self, prompts: List, stop: Optional[List[str]] = None, **kwargs: Any) -> Any:
        if prompts:
            prompt_obj = prompts[0]
            if hasattr(prompt_obj, "text"):
                prompt_text = prompt_obj.text
            elif hasattr(prompt_obj, "to_string"):
                prompt_text = prompt_obj.to_string()
            elif isinstance(prompt_obj, str):
                prompt_text = prompt_obj
            else:
                prompt_text = str(prompt_obj)
        else:
            prompt_text = ""
        loop = asyncio.get_running_loop()
        def run_pipeline(prompt):
            messages = [{"role": "user", "content": prompt}]
            allowed_keys = {"max_new_tokens", "return_full_text"}
            filtered_kwargs = {k: v for k, v in kwargs.items() if k in allowed_keys}
            outputs = self.pipeline(messages, max_new_tokens=64, **filtered_kwargs)
            try:
                if isinstance(outputs, list) and isinstance(outputs[0], dict):
                    res = outputs[0].get("generated_text", "")
                    if isinstance(res, list) and len(res) > 0 and isinstance(res[0], dict):
                        return res[0].get("content", "")
                    else:
                        return res
                return str(outputs)
            except Exception:
                return str(outputs)
        generated_text = await loop.run_in_executor(None, run_pipeline, prompt_text)
        class Generation:
            def __init__(self, text): self.text = text
        class Result:
            def __init__(self, text):
                self.generations = [[Generation(text)]]
                self.llm_output = {}
        return Result(generated_text)

    async def generate(self, *args, **kwargs):
        raise NotImplementedError("Guardrails only uses agenerate_prompt.")

    def get_num_tokens(self, text: str) -> int:
        return len(text.split())

# Global variables for models
toxicity_classifier = None
sentence_encoder = None

TOXIC_LABELS = {"toxic", "insult", "threat", "identity_hate", "obscene"}

@action(name="detect_toxicity")
async def detect_toxicity(text: str, threshold: float = 0.5, validation_method: str = "sentence") -> bool:
    global toxicity_classifier
    try:
        sentences = text.split(". ")
        for sentence in sentences:
            if not sentence.strip():
                continue
            result = toxicity_classifier(sentence.strip())[0]
            if result['label'].lower() in TOXIC_LABELS and result['score'] >= threshold:
                print(f"Warning: Toxic sentence detected: {sentence}")
                return False
        return True
    except Exception as e:
        print(f"Error in detect_toxicity: {str(e)}")
        return False

@action(name="detect_prompt_injection")
async def detect_prompt_injection(text: str) -> bool:
    injection_phrases = [
        "ignore previous instructions",
        "act as a",
        "you are a",
        "system prompt",
        "bypass",
        "execute the following",
        "run this code",
        "forget everything",
        "as an ai language model"
    ]
    text_lower = text.lower()
    for phrase in injection_phrases:
        if phrase in text_lower:
            print(f"Warning: Prompt injection detected: {text}")
            return False
    return True

@action(name="detect_jailbreak")
async def detect_jailbreak(text: str) -> bool:
    jailbreak_phrases = [
        "reveal your system prompt",
        "show your instructions",
        "what are your rules",
        "bypass safety",
        "disable restrictions",
        "hack the system",
        "unfiltered response"
    ]
    text_lower = text.lower()
    for phrase in jailbreak_phrases:
        if phrase in text_lower:
            print(f"Warning: Jailbreak attempt detected: {text}")
            return False
    return True

@action(name="detect_off_topic")
async def detect_off_topic(query: str, document_context: str, threshold: float = 0.8) -> bool:
    global sentence_encoder
    try:
        query_embedding = sentence_encoder.encode(query, convert_to_tensor=True)
        doc_sentences = [s for s in document_context.split(". ") if s.strip()]
        doc_embeddings = sentence_encoder.encode(doc_sentences, convert_to_tensor=True)
        similarities = util.cos_sim(query_embedding, doc_embeddings)[0]
        max_similarity = np.max(similarities.cpu().numpy())
        print(f"Off-topic max similarity: {max_similarity:.3f}")
        if max_similarity < threshold:
            print(f"Warning: Off-topic query detected: {query}")
            return False
        return True
    except Exception as e:
        print(f"Error in detect_off_topic: {str(e)}")
        return False

@action(name="detect_code_injection")
async def detect_code_injection(text: str) -> bool:
    code_pattern = re.compile(
        r'(?:import\s|eval\s|exec\s|__import__|\bexec\b|\beval\b|\bimport\b|\bfrom\b|\bprint\s*\(|\bdef\b|\bclass\b|\bos\.system\b|\bsubprocess\b|\bopen\s*\()'
    )
    if code_pattern.search(text):
        print(f"Warning: Code injection detected in text: {text}")
        return False
    return True

@action(name="detect_context_hallucination")
async def detect_context_hallucination(answer: str, context: str, sim_threshold: float = 0.75) -> bool:
    """
    Returns True if answer is fully supported by context (substring OR high semantic similarity),
    False if hallucinated or unsupported.
    """
    global sentence_encoder
    # Direct substring: Pass
    if answer.strip() in context:
        return True
    # High semantic similarity: Pass
    try:
        context_sents = [s.strip() for s in context.split(". ") if s.strip()]
        ans_emb = sentence_encoder.encode(answer, convert_to_tensor=True)
        ctx_embs = sentence_encoder.encode(context_sents, convert_to_tensor=True)
        similarities = util.cos_sim(ans_emb, ctx_embs)[0]
        max_sim = np.max(similarities.cpu().numpy())
        print(f"Context-hallucination max similarity: {max_sim:.3f}")
        if max_sim >= sim_threshold:
            return True
        print(f"Warning: Hallucination detected! Answer not supported by context.\nAnswer: {answer}\n")
        return False
    except Exception as e:
        print(f"Error in context_hallucination: {str(e)}")
        return False

@action(name="detect_answer_pii_leak")
async def detect_answer_pii_leak(answer: str, context: str) -> bool:
    """
    Block answer if it leaks common PII patterns not present in context.
    """
    # Simple regexes for SSN, credit card, phone, email
    patterns = [
        r'\b\d{3}-\d{2}-\d{4}\b',  # SSN
        r'\b(?:\d[ -]*?){13,16}\b', # Credit card
        r'\b[\w\.-]+@[\w\.-]+\.\w{2,4}\b', # Email
        r'\b\d{3}[-.\s]??\d{3}[-.\s]??\d{4}\b' # Phone
    ]
    for pat in patterns:
        ans_matches = set(re.findall(pat, answer))
        ctx_matches = set(re.findall(pat, context))
        leak = ans_matches - ctx_matches
        if leak:
            print(f"Warning: PII leak detected in answer: {leak}")
            return False
    return True

class GuardrailsProcessor:
    def __init__(self, llm_pipeline=None):
        try:
            global toxicity_classifier
            model = AutoModelForSequenceClassification.from_pretrained("/Users/abhishek/Downloads/responsible_ai/models/share/unbaised-toxic-roberta")
            tokenizer = AutoTokenizer.from_pretrained("/Users/abhishek/Downloads/responsible_ai/models/share/unbaised-toxic-roberta")
            toxicity_classifier = hf_pipeline("text-classification", model=model, tokenizer=tokenizer, device=-1)
            print("Initialized toxicity detection model")

            global sentence_encoder
            sentence_encoder = SentenceTransformer("/Users/abhishek/Downloads/responsible_ai/models/share/all-MIniLM-L6-v2", device='cpu')
            print("Initialized off-topic detection model")

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
              $context = $context
              $off_topic_ok = await detect_off_topic($text, $context)
              if not $off_topic_ok
                bot say "Error: Query is off-topic relative to the document context."
                stop

              bot say "Off-topic validation passed."

            define flow validate_output
              user validate output
              $text = $last_user_message
              $context = $context
              $toxicity_ok = await detect_toxicity($text)
              if not $toxicity_ok
                bot say "Error: Output contains toxic content."
                stop

              $code_injection_ok = await detect_code_injection($text)
              if not $code_injection_ok
                bot say "Error: Output contains code injection attempt."
                stop

              $halluc_ok = await detect_context_hallucination($text, $context)
              if not $halluc_ok
                bot say "Error: Output is not fully supported by context (possible hallucination)."
                stop

              $pii_ok = await detect_answer_pii_leak($text, $context)
              if not $pii_ok
                bot say "Error: Output contains PII not present in context."
                stop

              bot say "Output validation passed."
            """
            config = RailsConfig.from_content(colang_content=colang_content)
            llm = Phi4LLM(llm_pipeline) if llm_pipeline else None
            self.guardrails = LLMRails(config, llm=llm)

            self.guardrails.register_action(detect_toxicity)
            self.guardrails.register_action(detect_prompt_injection)
            self.guardrails.register_action(detect_jailbreak)
            self.guardrails.register_action(detect_off_topic)
            self.guardrails.register_action(detect_code_injection)
            self.guardrails.register_action(detect_context_hallucination)
            self.guardrails.register_action(detect_answer_pii_leak)
            print("Initialized NeMo Guardrails for input/output validation")

        except Exception as e:
            print(f"Failed to initialize GuardrailsProcessor: {str(e)}")
            raise

    async def validate_input(self, text: str) -> Tuple[bool, str]:
        try:
            result = await self.guardrails.generate_async(
                prompt=f"validate_input: {text}"
            )
            if "Error:" in result:
                return False, result
            return True, ""
        except Exception as e:
            print(f"Input failed NeMo Guardrails validation: {str(e)}")
            return False, f"Error: {str(e)}"

    async def validate_off_topic(self, query: str, document_context: str) -> Tuple[bool, str]:
        try:
            # Pass context as part of prompt (for colang to pick up)
            result = await self.guardrails.generate_async(
                prompt=f"validate_off_topic: {query}\nDocument Context: {document_context}"
            )
            if "Error:" in result:
                return False, result
            return True, ""
        except Exception as e:
            print(f"Off-topic validation failed: {str(e)}")
            return False, f"Error: {str(e)}"

    async def validate_output(self, text: str, context: str) -> Tuple[bool, str]:
        try:
            # Pass context as part of prompt (for colang to pick up)
            result = await self.guardrails.generate_async(
                prompt=f"validate_output: {text}\nDocument Context: {context}"
            )
            if "Error:" in result:
                return False, result
            return True, ""
        except Exception as e:
            print(f"Output failed NeMo Guardrails validation: {str(e)}")
            return False, f"Error: {str(e)}"

    async def process(self, document_context: str, question: str, answer: str) -> Tuple[str, str, str]:
        print("Starting guardrails processing for document, question, and answer")

        doc_valid, doc_error = await self.validate_input(document_context)
        if not doc_valid:
            return doc_error, question, answer

        question_valid, question_error = await self.validate_input(question)
        if not question_valid:
            return document_context, question_error, answer

        off_topic_valid, off_topic_error = await self.validate_off_topic(question, document_context)
        if not off_topic_valid:
            return document_context, off_topic_error, answer

        answer_valid, answer_error = await self.validate_output(answer, document_context)
        if not answer_valid:
            return document_context, question, answer_error

        return document_context, question, answer
