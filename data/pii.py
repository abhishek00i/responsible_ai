
# Use Presidio or spaCy to detect PII/PHI
import logging
from typing import Dict, List, Optional
from presidio_analyzer import AnalyzerEngine, RecognizerResult
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig
from faker import Faker
import spacy
from spacy.language import Language

class PIIAnonymizer():
    def __init__(self, context: str, question: str, language: str="en", spacy_model: str="en_core_web_lg"):
        self.language = language
        self.context = context
        self.question = question
        self.fake = Faker(locale=language)
        try :
            self.analyzer = AnalyzerEngine()
            self.anonymizer = AnonymizerEngine()
            self.nlp = spacy.load(spacy_model)
        
        except Exception as e :
            raise

    def _create_custom_operators(self) -> Dict[str,OperatorConfig] :
        
        return {
            "PERSON": OperatorConfig("custom", {"lambda": lambda _:self.fake.name()}),
            "PHONE_NUMBER": OperatorConfig("custom", {"lambda": lambda _:self.fake.phone_number()}),
            "EMAIL_ADDRESS": OperatorConfig("custom", {"lambda": lambda _:self.fake.email()}),
            "CREDIT_CARD": OperatorConfig("custom", {"lambda": lambda _:self.fake.credit_card_number()}),
            "LOCATION": OperatorConfig("custom", {"lambda": lambda _:self.fake.city()}),
            "DEFAULT": OperatorConfig("replace", {"new_value": "<REDACTED>"}),

        }

    def anonymize_text(self, text: str, entities: Optional[List[str]]=None) -> str :
        if not isinstance(text, str) or not text.strip():
            return text

        try:
            analyzer_results = self.analyzer.analyze(
                text=text,
                language=self.language,
                entities=entities if entities else None
            )

            
            operators = self._create_custom_operators()
            anonymized_result = self.anonymizer.anonymize(
                text=text,
                analyzer_results=analyzer_results,
                operators=operators
            )

            return anonymized_result.text

        except Exception as e :
            return text
    
    def anonymize_document_and_question(self, document_context: str, question: str, entities: Optional[List[str]]=None) -> tuple[str, str] :

        anonymized_document = self.anonymize_text(document_context, entities)
        anonymized_question = self.anonymize_text(question, entities)

        return anonymized_document, anonymized_question

    def main(self):
        try :
            entities_to_detect=["PERSON","PHONE_NUMBER","EMAIL_ADDRESS","CREDIT_CARD"]

            anonymized_context, anonymized_question = self.anonymize_document_and_question(
                document_context=self.context,
                question=self.question,
                entities=entities_to_detect
            )

            return anonymized_context, anonymized_question

        except Exception as e:

            return self.context, self.question

