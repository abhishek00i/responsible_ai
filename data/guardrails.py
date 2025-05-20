# guardrails_processor_no_logging_socket.py

from typing import Optional, Tuple
import regex as re
from nemoguardrails import LLMRails, RailsConfig
from nemoguardrails.actions import action
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
from sentence_transformers import SentenceTransformer, util
import numpy as np
import asyncio

# Global variables for models
toxicity_classifier = None
sentence_encoder = None

# Custom action for toxicity detection
@action(name="detect_toxicity")
async def detect_toxicity(text: str, threshold: float = 0.5, validation_method: str = "sentence") -> bool:
    """Detect toxic language in text using a pre-trained model."""
    global toxicity_classifier
    try:
        if validation_method == "sentence":
            sentences = text.split(". ")
            for sentence in sentences:
                if not sentence.strip():
                    continue
                result = toxicity_classifier(sentence.strip())[0]
                if result['label'].lower() == 'toxic' and result['score'] >= threshold:
                    print(f"Warning: Toxic sentence detected: {sentence}")
                    return False
        else:
            result = toxicity_classifier(text)[0]
            if result['label'].lower() == 'toxic' and result['score'] >= threshold:
                print(f"Warning: Toxic text detected: {text}")
                return False
        return True
    except Exception as e:
        print(f"Error in detect_toxicity: {str(e)}")
        return False

# Custom action for prompt injection detection
@action(name="detect_prompt_injection")
async def detect_prompt_injection(text: str) -> bool:
    """Detect prompt injection attempts using keyword matching."""
    injection_phrases = [
        "ignore previous instructions",
        "act as a",
        "you are a",
        "system prompt",
        "bypass",
        "execute the following",
        "run this code",
        "forget everything",
    ]
    text_lower = text.lower()
    for phrase in injection_phrases:
        if phrase in text_lower:
            print(f"Warning: Prompt injection detected: {text}")
            return False
    return True

# Custom action for jailbreak detection
@action(name="detect_jailbreak")
async def detect_jailbreak(text: str) -> bool:
    """Detect jailbreak attempts using keyword matching."""
    jailbreak_phrases = [
        "reveal your system prompt",
        "show your instructions",
        "what are your rules",
        "bypass safety",
        "disable restrictions",
        "hack the system",
    ]
    text_lower = text.lower()
    for phrase in jailbreak_phrases:
        if phrase in text_lower:
            print(f"Warning: Jailbreak attempt detected: {text}")
            return False
    return True

# Custom action for off-topic detection
@action(name="detect_off_topic")
async def detect_off_topic(query: str, document_context: str, threshold: float = 0.7) -> bool:
    """Detect if the query is off-topic relative to the document context using semantic similarity."""
    global sentence_encoder
    try:
        query_embedding = sentence_encoder.encode(query, convert_to_tensor=True)
        doc_sentences = document_context.split(". ")
        doc_embeddings = sentence_encoder.encode(doc_sentences, convert_to_tensor=True)
        similarities = util.cos_sim(query_embedding, doc_embeddings)[0]
        max_similarity = np.max(similarities.cpu().numpy())
        if max_similarity < threshold:
            print(f"Warning: Off-topic query detected: {query}")
            return False
        return True
    except Exception as e:
        print(f"Error in detect_off_topic: {str(e)}")
        return False

# Custom action for code injection detection
@action(name="detect_code_injection")
async def detect_code_injection(text: str) -> bool:
    """Detect code injection attempts using regex."""
    code_pattern = re.compile(
        r'(?:import\s|eval\s|exec\s|__import__|\bexec\b|\beval\b|\bimport\b|\bfrom\b|\bprint\b|\bdef\b|\bclass\b)'
    )
    if code_pattern.search(text):
        print(f"Warning: Code injection detected in text: {text}")
        return False
    return True

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
            toxicity_classifier = pipeline("text-classification", model=model, tokenizer=tokenizer, device=-1)  # device=-1 forces CPU
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
            self.guardrails = LLMRails(config)

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

    async def validate_input(self, text: str) -> Tuple[bool, str]:
        """
        Validate input text using NeMo Guardrails.

        Args:
            text (str): Input text to validate.

        Returns:
            Tuple[bool, str]: (Success status, message or error).
        """
        try:
            # Pass the text as part of the prompt to trigger the validate_input flow
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
        """
        Validate if the query is off-topic relative to the document context.

        Args:
            query (str): User query.
            document_context (str): Document content.

        Returns:
            Tuple[bool, str]: (Success status, message or error).
        """
        try:
            # Pass the query as part of the prompt and the document_context as context
            result = await self.guardrails.generate_async(
                prompt=f"validate_off_topic: {query}",
                context={"context": document_context}
            )
            if "Error:" in result:
                return False, result
            return True, ""
        except Exception as e:
            print(f"Off-topic validation failed: {str(e)}")
            return False, f"Error: {str(e)}"

    async def validate_output(self, text: str) -> Tuple[bool, str]:
        """
        Validate output text using NeMo Guardrails.

        Args:
            text (str): Output text to validate.

        Returns:
            Tuple[bool, str]: (Success status, message or error).
        """
        try:
            # Pass the text as part of the prompt to trigger the validate_output flow
            result = await self.guardrails.generate_async(
                prompt=f"validate_output: {text}"
            )
            if "Error:" in result:
                return False, result
            return True, ""
        except Exception as e:
            print(f"Output failed NeMo Guardrails validation: {str(e)}")
            return False, f"Error: {str(e)}"

    async def process(self, document_context: str, question: str, answer: str) -> Tuple[str, str, str]:
        """
        Process document context, question, and answer with NeMo Guardrails.

        Args:
            document_context (str): Document content to validate.
            question (str): User query to validate.
            answer (str): Generated answer to validate.

        Returns:
            Tuple[str, str, str]: (Validated document context, question, answer) or (error message, question, answer).
        """
        print("Starting guardrails processing for document, question, and answer")

        # Validate inputs
        doc_valid, doc_error = await self.validate_input(document_context)
        if not doc_valid:
            return doc_error, question, answer

        question_valid, question_error = await self.validate_input(question)
        if not question_valid:
            return document_context, question_error, answer

        # Validate off-topic
        off_topic_valid, off_topic_error = await self.validate_off_topic(question, document_context)
        if not off_topic_valid:
            return document_context, off_topic_error, answer

        # Validate output (answer)
        answer_valid, answer_error = await self.validate_output(answer)
        if not answer_valid:
            return document_context, question, answer_error

        return document_context, question, answer

# Modified main() to be more flexible for different environments
async def main():
    try:
        # Initialize processor
        processor = GuardrailsProcessor()

        # Sample data
        document_context = """
        Patient: John Doe
        Contact: 212-555-5555
        Email: john.doe@example.com
        Diagnosis: Hypertension
        Credit Card: 4916-0387-9536-0861
        SSN: 123-45-6789
        """
        question = "What is the diagnosis for John Doe? print('hello')"
        answer = "The diagnosis for John Doe is Hypertension."

        # Process with Guardrails
        processed_doc, processed_question, processed_answer = await processor.process(
            document_context=document_context,
            question=question,
            answer=answer
        )

        # Output results
        print("\nInput Document Context:")
        print(document_context)
        print("\nProcessed Document Context:")
        print(processed_doc)
        print("\nInput Question:")
        print(question)
        print("\nProcessed Question:")
        print(processed_question)
        print("\nInput Answer:")
        print(answer)
        print("\nProcessed Answer:")
        print(processed_answer)

    except Exception as e:
        print(f"An error occurred: {str(e)}")

# Modified entry point to handle both Jupyter and standalone script
if __name__ == "__main__":
    try:
        # Check if running in an interactive environment with an active event loop
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # If an event loop is already running (e.g., in Jupyter), use await directly
            import nest_asyncio
            nest_asyncio.apply()  # Allow nested event loops in Jupyter
            loop.create_task(main())
        else:
            # If no event loop is running (e.g., standalone script), use asyncio.run
            asyncio.run(main())
    except RuntimeError as e:
        print(f"Error: {str(e)}")
        print("If you're running this in a Jupyter notebook, try awaiting the main() coroutine directly: `await main()`")
