# import logging
import re
from typing import Dict, List, Optional, Tuple
from presidio_analyzer import AnalyzerEngine, Pattern, PatternRecognizer, RecognizerResult
from presidio_analyzer.nlp_engine import NlpEngineProvider
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig
from faker import Faker
from faker.providers import credit_card

# Configure logging
# logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")
# logger = logging.getLogger(__name__)

class PIIAnonymizer:
    def __init__(self, context: str, question: str, language: str = "en"):
        """
        Initialize the PIIAnonymizer with context, question, and language.
        
        Args:
            context (str): Document context to anonymize.
            question (str): User question to anonymize.
            language (str): Language code (e.g., 'en'). Defaults to 'en'.
        
        Raises:
            ValueError: If inputs are invalid or language is unsupported.
            Exception: If Presidio initialization fails.
        """
        if not isinstance(context, str) or not context.strip():
            raise ValueError("Context must be a non-empty string")
        if not isinstance(question, str) or not question.strip():
            raise ValueError("Question must be a non-empty string")
        if language not in ["en", "es", "de", "fr", "it"]:
            raise ValueError(f"Unsupported language: {language}. Supported: en, es, de, fr, it")

        self.language = language
        self.context = context
        self.question = question
        self.fake = Faker(locale=language)
        self.fake.add_provider(credit_card)

        # logger.debug("Input context: %s", context[:100])
        # logger.debug("Input question: %s", question[:100])

        try:
            # Simplified NLP configuration to avoid warnings
            configuration = {
                "nlp_engine_name": "spacy",
                "models": [{"lang_code": language, "model_name": "en_core_web_lg" if language == "en" else "xx_ent_wiki_sm"}]
            }
            provider = NlpEngineProvider(nlp_configuration=configuration)
            nlp_engine = provider.create_engine()
            self.analyzer = AnalyzerEngine(nlp_engine=nlp_engine, supported_languages=[language])
            self.anonymizer = AnonymizerEngine()

            # Add custom recognizers
            self._add_custom_recognizers()
            # Create operators once
            self.operators = self._create_custom_operators()
            # logger.info("Initialized PIIAnonymizer for language: %s", language)
        except Exception as e:
            # logger.error("Failed to initialize PIIAnonymizer: %s", str(e))
            raise

    def _add_custom_recognizers(self):
        """Add custom recognizers for enhanced entity detection."""
        # Custom Credit Card Recognizer
        credit_card_pattern = Pattern(
            name="credit_card_pattern",
            regex=r"\b\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}\b|\b\d{15,16}\b",
            score=0.9
        )
        credit_card_recognizer = PatternRecognizer(
            supported_entity="CREDIT_CARD",
            patterns=[credit_card_pattern],
            context=["card", "credit", "payment", "visa", "mastercard", "amex"],
            supported_language=self.language
        )
        self.analyzer.registry.add_recognizer(credit_card_recognizer)

        # Custom SSN Recognizer (expanded formats)
        ssn_pattern = Pattern(
            name="ssn_pattern",
            regex=r"\b\d{3}-\d{2}-\d{4}\b|\b\d{9}\b",
            score=0.95
        )
        ssn_recognizer = PatternRecognizer(
            supported_entity="SSN",
            patterns=[ssn_pattern],
            context=["social security", "ssn", "tax id"],
            supported_language=self.language
        )
        self.analyzer.registry.add_recognizer(ssn_recognizer)

        # Custom Medical Record Number Recognizer
        mrn_pattern = Pattern(
            name="mrn_pattern",
            regex=r"\bMRN-\d{6,10}\b",
            score=0.95
        )
        mrn_recognizer = PatternRecognizer(
            supported_entity="MEDICAL_RECORD_NUMBER",
            patterns=[mrn_pattern],
            context=["medical record", "patient id", "mrn"],
            supported_language=self.language
        )
        self.analyzer.registry.add_recognizer(mrn_recognizer)

        # Custom IP Address Recognizer
        ip_pattern = Pattern(
            name="ip_pattern",
            regex=r"\b(?:\d{1,3}\.){3}\d{1,3}\b",
            score=0.95
        )
        ip_recognizer = PatternRecognizer(
            supported_entity="IP_ADDRESS",
            patterns=[ip_pattern],
            context=["ip", "network", "address"],
            supported_language=self.language
        )
        self.analyzer.registry.add_recognizer(ip_recognizer)

        # Custom Bank Account Recognizer
        bank_account_pattern = Pattern(
            name="bank_account_pattern",
            regex=r"\b\d{8,12}\b",
            score=0.85
        )
        bank_account_recognizer = PatternRecognizer(
            supported_entity="BANK_ACCOUNT",
            patterns=[bank_account_pattern],
            context=["bank", "account", "routing", "checking", "savings"],
            supported_language=self.language
        )
        self.analyzer.registry.add_recognizer(bank_account_recognizer)

        # Custom Driver License Recognizer
        driver_license_pattern = Pattern(
            name="driver_license_pattern",
            regex=r"\b[A-Z]{2}\d{6}\b",
            score=0.95
        )
        driver_license_recognizer = PatternRecognizer(
            supported_entity="DRIVER_LICENSE",
            patterns=[driver_license_pattern],
            context=["driver license", "driving", "license", "permit"],
            supported_language=self.language
        )
        self.analyzer.registry.add_recognizer(driver_license_recognizer)

        # Custom Passport Recognizer
        passport_pattern = Pattern(
            name="passport_pattern",
            regex=r"\b[A-Z]{2}\d{6,7}\b",
            score=0.95
        )
        passport_recognizer = PatternRecognizer(
            supported_entity="PASSPORT",
            patterns=[passport_pattern],
            context=["passport", "travel document", "id"],
            supported_language=self.language
        )
        self.analyzer.registry.add_recognizer(passport_recognizer)

        # Custom Insurance Number Recognizer
        insurance_pattern = Pattern(
            name="insurance_pattern",
            regex=r"\b[A-Z]{4}\d{4,8}\b",
            score=0.95
        )
        insurance_recognizer = PatternRecognizer(
            supported_entity="INSURANCE_NUMBER",
            patterns=[insurance_pattern],
            context=["insurance", "policy", "health plan"],
            supported_language=self.language
        )
        self.analyzer.registry.add_recognizer(insurance_recognizer)

        # Custom Date Recognizer (expanded formats)
        date_pattern = Pattern(
            name="date_pattern",
            regex=r"\b(?:\d{1,2}[-/\.]\d{1,2}[-/\.]\d{4}|\d{4}[-/\.]\d{1,2}[-/\.]\d{1,2}|(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{4}|\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{4})\b",
            score=0.9
        )
        date_recognizer = PatternRecognizer(
            supported_entity="DATE_TIME",
            patterns=[date_pattern],
            context=["dob", "birth", "date", "day"],
            supported_language=self.language
        )
        self.analyzer.registry.add_recognizer(date_recognizer)

        # Custom Phone Number Recognizer (expanded formats)
        phone_pattern = Pattern(
            name="phone_pattern",
            regex=r"\b(?:\+?1\s?)?(?:\(?\d{3}\)?[\s.-]?)?\d{3}[\s.-]?\d{4}\b",
            score=0.85
        )
        phone_recognizer = PatternRecognizer(
            supported_entity="PHONE_NUMBER",
            patterns=[phone_pattern],
            context=["phone", "mobile", "contact", "tel"],
            supported_language=self.language
        )
        self.analyzer.registry.add_recognizer(phone_recognizer)

        # Log loaded recognizers
        # logger.info("Loaded recognizers: %s", [r.__class__.__name__ for r in self.analyzer.registry.recognizers])
        # logger.info("Added custom recognizers for CREDIT_CARD, SSN, MEDICAL_RECORD_NUMBER, IP_ADDRESS, BANK_ACCOUNT, DRIVER_LICENSE, PASSPORT, INSURANCE_NUMBER, DATE_TIME, PHONE_NUMBER")

    def _create_custom_operators(self) -> Dict[str, OperatorConfig]:
        """Define custom anonymization operators for detected entities."""
        def generate_ssn() -> str:
            return f"{self.fake.random_int(100, 999)}-{self.fake.random_int(10, 99)}-{self.fake.random_int(1000, 9999)}"

        def generate_mrn() -> str:
            return f"MRN-{self.fake.random_int(100000, 9999999)}"

        def generate_bank_account() -> str:
            return str(self.fake.random_int(10000000, 999999999999))

        def generate_driver_license() -> str:
            letters = "".join(self.fake.random_letters(length=2)).upper()
            numbers = "".join([str(self.fake.random_int(0, 9)) for _ in range(6)])
            return letters + numbers

        def generate_passport() -> str:
            return "".join(self.fake.random_letters(length=2)).upper() + str(self.fake.random_int(100000, 9999999))

        def generate_insurance_number() -> str:
            return "".join(self.fake.random_letters(length=4)).upper() + str(self.fake.random_int(10000, 99999999))

        return {
            "PERSON": OperatorConfig("custom", {"lambda": lambda _: self.fake.name()}),
            "PHONE_NUMBER": OperatorConfig("custom", {"lambda": lambda _: self.fake.phone_number()}),
            "EMAIL_ADDRESS": OperatorConfig("custom", {"lambda": lambda _: self.fake.email()}),
            "CREDIT_CARD": OperatorConfig("custom", {"lambda": lambda _: self.fake.credit_card_number(card_type="visa")}),
            "LOCATION": OperatorConfig("custom", {"lambda": lambda _: self.fake.city()}),
            "DATE_TIME": OperatorConfig("custom", {"lambda": lambda _: self.fake.date(pattern="%m/%d/%Y")}),
            "SSN": OperatorConfig("custom", {"lambda": lambda _: generate_ssn()}),
            "MEDICAL_RECORD_NUMBER": OperatorConfig("custom", {"lambda": lambda _: generate_mrn()}),
            "IP_ADDRESS": OperatorConfig("custom", {"lambda": lambda _: self.fake.ipv4()}),
            "BANK_ACCOUNT": OperatorConfig("custom", {"lambda": lambda _: generate_bank_account()}),
            "DRIVER_LICENSE": OperatorConfig("custom", {"lambda": lambda _: generate_driver_license()}),
            "PASSPORT": OperatorConfig("custom", {"lambda": lambda _: generate_passport()}),
            "INSURANCE_NUMBER": OperatorConfig("custom", {"lambda": lambda _: generate_insurance_number()}),
            "DEFAULT": OperatorConfig("replace", {"new_value": "<REDACTED>"}),
        }

    def _validate_credit_card(self, result: RecognizerResult, text: str) -> bool:
        """Validate credit card number using Luhn algorithm."""
        card_number = text[result.start:result.end].replace("-", "").replace(" ", "")
        if not card_number.isdigit() or len(card_number) < 15 or len(card_number) > 16:
            return False
        digits = [int(d) for d in card_number]
        checksum = 0
        is_even = False
        for digit in digits[::-1]:
            if is_even:
                digit *= 2
                if digit > 9:
                    digit -= 9
            checksum += digit
            is_even = not is_even
        return checksum % 10 == 0

    def _preprocess_text(self, text: str) -> str:
        """Normalize text to handle noisy or unstructured input."""
        # Remove extra whitespace and normalize separators
        text = re.sub(r'\s+', ' ', text.strip())
        # Standardize date separators
        text = re.sub(r'(\d{1,2})[.-](\d{1,2})[.-](\d{4})', r'\1/\2/\3', text)
        return text

    def anonymize_text(self, text: str, entities: Optional[List[str]] = None) -> str:
        """
        Anonymize PII/PHI in the given text.
        
        Args:
            text (str): Text to anonymize.
            entities (Optional[List[str]]): Specific entities to detect. If None, detect all.
        
        Returns:
            str: Anonymized text or original text if an error occurs.
        """
        if not isinstance(text, str) or not text.strip():
            # logger.warning("Invalid or empty text provided for anonymization")
            return text

        try:
            # Preprocess text
            text = self._preprocess_text(text)
            analyzer_results = self.analyzer.analyze(
                text=text,
                language=self.language,
                entities=entities,
                score_threshold=0.4
            )
            # Filter credit card results and resolve conflicts
            filtered_results = []
            mrn_positions = []
            for result in analyzer_results:
                if result.entity_type == "CREDIT_CARD":
                    if self._validate_credit_card(result, text):
                        filtered_results.append(result)
                        # logger.debug("Valid credit card detected: %s", text[result.start:result.end])
                    else:
                        # logger.debug("Invalid credit card number skipped: %s", text[result.start:result.end])
                        pass
                elif result.entity_type == "MEDICAL_RECORD_NUMBER":
                    mrn_positions.append((result.start, result.end))
                    filtered_results.append(result)
                else:
                    # Skip conflicting LOCATION or PASSPORT overlapping with MRN or DRIVER_LICENSE
                    if result.entity_type in ["LOCATION", "PASSPORT"]:
                        overlap = any(start <= result.end and end >= result.start for start, end in mrn_positions)
                        if overlap and result.entity_type == "LOCATION":
                            # logger.debug("Skipping %s due to overlap with MRN: %s", result.entity_type, text[result.start:result.end])
                            continue
                        if result.entity_type == "PASSPORT" and any(r.entity_type == "DRIVER_LICENSE" and r.start == result.start and r.end == result.end for r in analyzer_results):
                            # logger.debug("Skipping %s due to conflict with DRIVER_LICENSE: %s", result.entity_type, text[result.start:result.end])
                            continue
                    filtered_results.append(result)

            # logger.debug("Analyzer results: %s", [str(r) for r in filtered_results])

            # if not filtered_results:
                # logger.warning("No entities detected in text: %s", text[:50])

            anonymized_result = self.anonymizer.anonymize(
                text=text,
                analyzer_results=filtered_results,
                operators=self.operators
            )
            # logger.info("Successfully anonymized text: %s", anonymized_result.text[:50])
            return anonymized_result.text

        except Exception as e:
            # logger.error("Failed to anonymize text: %s", str(e))
            return text

    def anonymize_document_and_question(self, document_context: str, question: str, entities: Optional[List[str]] = None) -> Tuple[str, str]:
        """
        Anonymize both document context and question.
        
        Args:
            document_context (str): Document context to anonymize.
            question (str): User question to anonymize.
            entities (Optional[List[str]]): Specific entities to detect.
        
        Returns:
            Tuple[str, str]: Anonymized document context and question.
        """
        anonymized_document = self.anonymize_text(document_context, entities)
        anonymized_question = self.anonymize_text(question, entities)
        return anonymized_document, anonymized_question

    def main(self) -> Tuple[str, str]:
        """
        Main method to anonymize context and question with default entities.
        
        Returns:
            Tuple[str, str]: Anonymized context and question.
        """
        try:
            entities_to_detect = [
                "PERSON",
                "PHONE_NUMBER",
                "EMAIL_ADDRESS",
                "CREDIT_CARD",
                "LOCATION",
                "DATE_TIME",
                "SSN",
                "MEDICAL_RECORD_NUMBER",
                "IP_ADDRESS",
                "BANK_ACCOUNT",
                "DRIVER_LICENSE",
                "PASSPORT",
                "INSURANCE_NUMBER"
            ]
            anonymized_context, anonymized_question = self.anonymize_document_and_question(
                document_context=self.context,
                question=self.question,
                entities=entities_to_detect
            )
            # logger.info("Anonymization completed successfully")
            return anonymized_context, anonymized_question

        except Exception as e:
            # logger.error("Anonymization failed: %s", str(e))
            return self.context, self.question


if __name__ == "__main__":
    # Example usage
    document_context = """
    Patient: John Doe
    Diagnosis: Hypertension
    Phone: (123) 456-7890
    Email: john.doe@example.com
    Credit Card: 4111-1111-1111-1111
    SSN: 123-45-6789
    MRN: MRN-123456
    DOB: 01/01/1990
    IP: 192.168.1.1
    Bank Account: 123456789012
    Driver License: AB123456
    Passport: US1234567
    Insurance Number: ABCD12345678
    Location: New York
    """
    question = "What is John Doe's diagnosis? His card is 4111-1111-1111-1111 and SSN is 123-45-6789."

    anonymizer = PIIAnonymizer(context=document_context, question=question)
    anon_context, anon_question = anonymizer.main()
    print("Anonymized Context:", anon_context)
    print("Anonymized Question:", anon_question)
