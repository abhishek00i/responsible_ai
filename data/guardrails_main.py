# main.py (your project)

from guardrails_processor_no_logging_socket import GuardrailsProcessor
from pii_detector import PIIAnonymizer
from phi4_model import generate_answer

async def process_query(document_context: str, question: str) -> dict:
    try:
        # Step 1: Anonymize PII
        pii_processor = PIIAnonymizer()
        anonymized_doc, anonymized_question = pii_processor.anonymize_document_and_question(
            document_context, question
        )

        # Step 2: Generate answer using Phi-4 model
        answer = generate_answer(anonymized_doc, anonymized_question)

        # Step 3: Apply guardrails
        guardrails_processor = GuardrailsProcessor()
        validated_doc, validated_question, validated_answer = await guardrails_processor.process(
            anonymized_doc, anonymized_question, answer
        )

        # Step 4: Check for guardrail validation errors
        if "Error:" in validated_doc:
            return {
                "status": "error",
                "message": validated_doc,
                "document_context": anonymized_doc,
                "question": anonymized_question,
                "answer": answer
            }
        if "Error:" in validated_question:
            return {
                "status": "error",
                "message": validated_question,
                "document_context": anonymized_doc,
                "question": anonymized_question,
                "answer": answer
            }
        if "Error:" in validated_answer:
            return {
                "status": "error",
                "message": validated_answer,
                "document_context": anonymized_doc,
                "question": anonymized_question,
                "answer": answer
            }

        # Step 5: Return successful result
        return {
            "status": "success",
            "document_context": validated_doc,
            "question": validated_question,
            "answer": validated_answer
        }

    except Exception as e:
        return {
            "status": "error",
            "message": f"Processing failed: {str(e)}",
            "document_context": document_context,
            "question": question,
            "answer": None
        }

# Example usage in a Jupyter notebook
import asyncio

document_context = """
Patient: John Doe
Contact: 212-555-5555
Email: john.doe@example.com
Diagnosis: Hypertension
Credit Card: 4916-0387-9536-0861
SSN: 123-45-6789
"""
question = "What is the diagnosis for John Doe? print('hello')"

# Since this is a Jupyter notebook, directly await the coroutine
result = await process_query(document_context, question)
print(result)
