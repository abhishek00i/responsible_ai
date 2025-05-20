from typing import Optional, Tuple 
import regex as re
from nemoguardrails import LLMRails, RailsConfig
from nemoguardrails.actions import action
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
from sentence_transformers import SentenceTransformer, util
import numpy as np

# Global variables for models
toxicity_classifier = None
sentence_encoder = None

#custom action for toxicity detector
@action(name="detect_toxicity")
async def detect_toxicity(text: str, threshold: float = 0.5, validation_method: str = "sentence") -> bool :
    global toxicity_classifier
    try:
        if validation_method == "sentence" :
            sentence = text.split(". ")
            for sentence in sentences:
                if not sentence.strip():
                    continue
                result = toxicity_classifier(sentence.strip())[0]
                if result['label'].lower() == 'toxic' and result['score'] >= threshold :
                    print(f"Warning toxic language detected : {sentence}")
                    return False
        
        else :
            result = toxicity_classifier(text)[0]
            if result['label'].lower() == 'texic' and result['score'] >= threshold :
                print("warning toxic text detected: {text}")
                return False
        
        return True
    
    except Exaception as e:
        print(f"error in detect_toxicity: {str(e)}")
        return False

@action(name="detect_prompt_injection")
async def detect_prompt_injection(text: str) -> bool :
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
        if phrase in text_lower():
            print(f" prompt injection detected: {text}")
            return False
    
    return True

@action(name="detect_jailbreak")
async def detect_jailbreak(text:str) -> bool :
    jailbreak_phrases=[
        "reveal your system prompt",
        "show your instructions",
        "what are your rules",
        "bypass safety",
        "disable restrictions",
        "hact the system",
    ]

    text_lower = text.lower()
    for phrase in jailbreak_phrases:
        if phrase in text_lower:
            print(f" jailbreak detected: {text}")
            return False

    return True

@action(name="detect_off_topic")
async def detect_off_topic(query: str, document_context: str, threshold: float=0.7) -> bool:
    global sentence_encoder
    try:
        query_embedding = sentence_encoder.encode(query, convert_to_tensor=True)
        doc_sentences = document_context.split(". ").strip()
        doc_embeddings = sentence_encoder.encode(doc_sentences, convert_to_tensor=True)
        similarities = util.cos_slim(query_embedding, doc_embeddings)[0]
        max_similarity = np.max(similarities.cpu().numpy())
        if max_similarity < threshold :
            print(f"off-topic query selected: {query}")
            return False
        
        return True
    
    except Exception as e:
        print(f"error in detect off topic - {e}")
        return False

@action(name="detect_code_injection")
async def detect_code_injection(text: str) -> bool :
    code_pattern = re.compile(
        r'(?:import\s|eval\s|exec\s|__import__|\bexec\b|\bfrom\b|\bprint\b|\bdef\b|\bclass\b)'
    )

    if code_pattern.search(text):
        print(f"code injection detected in text: {text}")
        return False
    return True

class GuardrailsProcessor :
    def __init__(self):
        try:
            global toxicity_classifier
            model=AutoModelForSequenceClassification.from_pretrained("/toxic_language_model")
            tokenizer=AutoTokenizer.from_pretrained("/toxic_lan_model")

            toxicity_classifier = pipeline(
                "text-classification",
                model=model,
                tokenizer=tokenizer,
                device="auto" # for cpu use -1
            )

            global sentence_encoder
            sentence_encoder = SentenceTransformer("/all-miniLM-l6-vs",device='cpu')

            colang_content = """
            define user input
                $text = $user_message
                $document_context = $context

            define flow validate_input
                $toxicity_ok = await detect_toxicity($text)
                if not $toxicity_ok
                    bot say "Error : Input contains toxic content."
                    stop
                
                $prompt_injection_ok = await detect_prompt_injection($text)
                if not $prompt_injection_ok
                    bot say "Error : Input contains prompt injection attempt."
                    stop
                
                $jailbreak_ok = await detect_jailbreak($text)
                if not $jailbreak_ok
                    bot say "Error : Input contains jailbreak attempt."
                    stop
                
                $code_injection_ok = await detect_code_injection($text)
                if not $code_injection_ok
                    bot say "Error : Input contains code injection attempt."
                    stop

            define flow validate_off_topic
                $off_topic_ok = await detect_off_topic($text, $document_context)
                if not $off_topic_ok
                    bot say "Error : Query if off-topic relative to the document context."
                    stop

            define flow validate_output
                $toxicity_ok =  await detect_toxicity($text)
                if not $toxicity_ok
                    bot say "Error : Output contains toxic content."
                    stop
                
                $code_injection_ok = await detect_code_injection($text)
                if not $code_injection_ok 
                    bot say "Error : Output contains code injection attempt."
                    stop
            """

            config = RailsConfig.from_content(colang_content=colang_content)
            self.guardrails = LLMRails(config)

            #Register Custom Actions
            self.guardrails.register_action(detect_toxicity)  
            self.guardrails.register_action(detect_prompt_injection) 
            self.guardrails.register_action(detect_jailbreak) 
            self.guardrails.register_action(detect_off_topic) 
            self.guardrails.register_action(detect_code_injection) 

            print("Initialized Nemo Guardrails for input and output validation.")

        except Exaception as e :
            print("Failed to initialize GuardrailsProcesser: {e}")
            raise
        
    async def validate_input(self, text: str) -> Tuple[bool, str] :
        try:
            result = await self.guardrails.generate_async(
                prompt = text,
                flows = ["validate_input"],
                user_message = text
            )
                   
            if "Error : " in result:
                retrun False , result
            
            return True, ""

        except Exaception as e :
            print(f"Input Failed Nemo Guardrails validation : {str(e)}")
            retrun False, f"Error : {str(e)}"

    async def validate_off_topic(self, query: str, document_context: str) -> Tuple[bool, str] :
        try:
            result = await self.guardrails.generate_async(
                prompt = query,
                flows = ["validate_off_topic"],
                user_message = query,
                context = document_context
            )

            if "Error : " in result:
                retrun False , result
            
            return True, ""

        except Exaception as e :
            print(f"Off-topic validation failed : {str(e)}")
            retrun False, f"Error : {str(e)}"

    async def validate_output(self, text: str ) -> Tuple[bool, str] :
        try:
            result = await self.guardrails.generate_async(
                prompt = text,
                flows = ["validate_output"],
                user_message = text 
            )

            if "Error : " in result:
                retrun False , result
            
            return True, ""

        except Exaception as e :
            print(f"Output Failed Nemo Guardrails validation : {str(e)}")
            retrun False, f"Error : {str(e)}"

    async def process(self, document_context: str, question: str, answer :str) -> Tuple[str, str, str] :
        doc_valid, doc_error = await self.validate_input(document_context)
        if not doc_valid:
            return doc_error, question, answer
        
        question_valid, question_error = await self.validate_input(question)
        if not question_valid:
            return document_context, question_error, answer

        off_topic_valid, off_topic_error = await self.validate_off_topic(question, document_context)
        if not off_topic_valid:
            return document_context, off_topic_error, answer

        answer_valid, answer_error = await self.validate_output(answer)
        if not answer_valid:
            return document_context, question, answer

        return document_context, question, answer

