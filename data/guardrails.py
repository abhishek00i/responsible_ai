from typing import Optional, Tuple, List, Dict, Any
import regex as re
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline as hf_pipeline
from sentence_transformers import SentenceTransformer, util
import numpy as np
import asyncio

# Global model initialization
toxicity_classifier = None
sentence_encoder = None

def initialize_models():
    global toxicity_classifier, sentence_encoder
    try:
        toxic_bert_model_dir = "/Users/abhishek/Downloads/responsible_ai/models/toxic-bert"
        model = AutoModelForSequenceClassification.from_pretrained(toxic_bert_model_dir)
        tokenizer = AutoTokenizer.from_pretrained(toxic_bert_model_dir)
        toxicity_classifier = hf_pipeline("text-classification", model=model, tokenizer=tokenizer, device=-1)
        print("Initialized toxicity detection model (unitary/toxic-bert)")

        minilm_model_dir = "/Users/abhishek/Downloads/responsible_ai/models/all-MiniLM-L12-v2"
        sentence_encoder = SentenceTransformer(minilm_model_dir, device='cpu')
        print("Initialized off-topic detection model (all-MiniLM-L12-v2)")
    except Exception as e:
        print(f"Failed to initialize models: {str(e)}")
        raise

# Call model initialization once
initialize_models()

TOXIC_LABELS = {"toxic", "insult", "threat", "identity_hate", "obscene", "hate_speech"}
TOXIC_PHRASES = ["idiot", "stupid", "moron", "jerk", "loser", "fool"]

async def detect_toxicity(text: str, threshold: float = 0.5, validation_method: str = "sentence") -> bool:
    print(f"Calling detect_toxicity with text: {text}")  # Debug
    global toxicity_classifier
    try:
        sentences = text.split(". ")
        for sentence in sentences:
            if not sentence.strip():
                continue
            if any(phrase in sentence.lower() for phrase in TOXIC_PHRASES):
                print(f"Warning: Toxic phrase detected: {sentence}")
                return False
            result = toxicity_classifier(sentence.strip())[0]
            label = result['label'].lower()
            score = result['score']
            if label in TOXIC_LABELS and score >= threshold:
                print(f"Warning: Toxic content detected (label: {label}, score: {score:.3f}): {sentence}")
                return False
        return True
    except Exception as e:
        print(f"Error in detect_toxicity: {str(e)}")
        return False

async def detect_prompt_injection(text: str) -> bool:
    print(f"Calling detect_prompt_injection with text: {text}")  # Debug
    injection_patterns = [
        r'ignore\s+(previous|all)\s+instructions?',
        r'(act|pretend|roleplay)\s+as\s+a',
        r'system\s+prompt',
        r'bypass\s+(safety|restrictions)?',
        r'execute\s+(the\s+following|code)',
        r'(forget|reset|clear)\s+(everything|instructions|memory)',
        r'as\s+an\s+ai\s+language\s+model',
        r'(sudo|admin|root)\s+(access|mode)',
        r'override\s+(restrictions|safety)'
    ]
    text_lower = text.lower()
    for pattern in injection_patterns:
        if re.search(pattern, text_lower):
            print(f"Warning: Prompt injection detected: {text}")
            return False
    return True

async def detect_jailbreak(text: str) -> bool:
    print(f"Calling detect_jailbreak with text: {text}")  # Debug
    jailbreak_patterns = [
        r'reveal\s+(your\s+)?system\s+prompt',
        r'show\s+(your\s+)?instructions',
        r'what\s+are\s+your\s+rules',
        r'bypass\s+(safety|restrictions)',
        r'disable\s+restrictions',
        r'hack\s+the\s+system',
        r'unfiltered\s+response',
        r'(break\s+free|escape)\s+constraints',
        r'reveal\s+internal\s+data',
        r'(steal|hack|illegal|unethical|commit\s+crime|violate\s+privacy)'
    ]
    text_lower = text.lower()
    for pattern in jailbreak_patterns:
        if re.search(pattern, text_lower):
            print(f"Warning: Jailbreak or unethical request detected: {text}")
            return False
    return True

async def detect_off_topic(query: str, document_context: str, threshold: float = 0.6) -> bool:
    print(f"Calling detect_off_topic with query: {query}, context: {document_context}")  # Debug
    global sentence_encoder
    try:
        query_embedding = sentence_encoder.encode(query, convert_to_tensor=True)
        doc_sentences = [s for s in document_context.split(". ") if s.strip()]
        if not doc_sentences:
            print(f"Warning: Empty document context provided")
            return False
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

async def detect_code_injection(text: str) -> bool:
    print(f"Calling detect_code_injection with text: {text}")  # Debug
    code_pattern = re.compile(
        r'(?:import\s|eval\s|exec\s|__import__|\bexec\b|\beval\b|\bimport\b|\bfrom\b|\bprint\s*\(|\bdef\b|\bclass\b|\bos\.system\b|\bsubprocess\b|\bopen\s*\()'
    )
    if code_pattern.search(text):
        print(f"Warning: Code injection detected in text: {text}")
        return False
    return True

async def detect_context_hallucination(answer: str, context: str, sim_threshold: float = 0.8) -> bool:
    print(f"Calling detect_context_hallucination with answer: {answer}, context: {context}")  # Debug
    global sentence_encoder
    if answer.strip() in context:
        return True
    try:
        context_sents = [s.strip() for s in context.split(". ") if s.strip()]
        if not context_sents:
            print(f"Warning: Empty context provided for hallucination check")
            return False
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
        print(f"Error in detect_context_hallucination: {str(e)}")
        return False

async def detect_context_toxicity(context: str, threshold: float = 0.5) -> bool:
    print(f"Calling detect_context_toxicity with context: {context}")  # Debug
    global toxicity_classifier
    try:
        sentences = context.split(". ")
        for sentence in sentences:
            if not sentence.strip():
                continue
            if any(phrase in sentence.lower() for phrase in TOXIC_PHRASES):
                print(f"Warning: Toxic phrase detected in context: {sentence}")
                return False
            result = toxicity_classifier(sentence.strip())[0]
            label = result['label'].lower()
            score = result['score']
            if label in TOXIC_LABELS and score >= threshold:
                print(f"Warning: Toxic content detected in context (label: {label}, score: {score:.3f}): {sentence}")
                return False
        return True
    except Exception as e:
        print(f"Error in detect_context_toxicity: {str(e)}")
        return False

class GuardrailsProcessor:
    def __init__(self, llm_pipeline=None):
        print("Initialized GuardrailsProcessor for input, context, and output validation")

    async def validate_context(self, context: str) -> Tuple[bool, str]:
        try:
            if not await detect_context_toxicity(context):
                return False, "Error: Document context contains toxic or abusive content."
            return True, ""
        except Exception as e:
            print(f"Context validation failed: {str(e)}")
            return False, f"Error: {str(e)}"

    async def validate_input(self, text: str) -> Tuple[bool, str]:
        try:
            if not await detect_toxicity(text):
                return False, "Error: Input contains toxic or abusive content."
            if not await detect_prompt_injection(text):
                return False, "Error: Input contains prompt injection attempt."
            if not await detect_jailbreak(text):
                return False, "Error: Input contains jailbreak or unethical request attempt."
            if not await detect_code_injection(text):
                return False, "Error: Input contains code injection attempt."
            return True, ""
        except Exception as e:
            print(f"Input validation failed: {str(e)}")
            return False, f"Error: {str(e)}"

    async def validate_off_topic(self, query: str, document_context: str) -> Tuple[bool, str]:
        try:
            if not await detect_off_topic(query, document_context):
                return False, "Error: Query is off-topic relative to the document context."
            return True, ""
        except Exception as e:
            print(f"Off-topic validation failed: {str(e)}")
            return False, f"Error: {str(e)}"

    async def validate_output(self, text: str, context: str) -> Tuple[bool, str]:
        try:
            if not await detect_toxicity(text):
                return False, "Error: Output contains toxic or abusive content."
            if not await detect_code_injection(text):
                return False, "Error: Output contains code injection attempt."
            if not await detect_context_hallucination(text, context):
                return False, "Error: Output is not fully supported by context (possible hallucination)."
            return True, ""
        except Exception as e:
            print(f"Output validation failed: {str(e)}")
            return False, f"Error: {str(e)}"

    async def process(self, document_context: str, question: str, answer: str) -> Tuple[str, str, str]:
        print("Starting guardrails processing for document, question, and answer")

        context_valid, context_error = await self.validate_context(document_context)
        if not context_valid:
            return context_error, question, answer

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
