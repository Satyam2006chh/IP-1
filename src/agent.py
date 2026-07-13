import os
import spacy
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from langchain_openai import ChatOpenAI

# Load our custom trained spaCy model lazily
nlp = None
def get_nlp():
    global nlp
    if nlp is None:
        try:
            nlp = spacy.load("models/custom_ner")
        except:
            pass
    return nlp

class ProductAttributes(BaseModel):
    Silhouette: Optional[str] = Field(default=None, description="The shape or outline (e.g., A line, mermaid, sheath, fitted)")
    Fabric: Optional[str] = Field(default=None, description="The material used (e.g., chiffon, lace, satin, tulle, velvet)")
    Neckline: Optional[str] = Field(default=None, description="The neckline style (e.g., V neckline, sweetheart, scoop, halter)")
    Sleeve: Optional[str] = Field(default=None, description="The sleeve style (e.g., long sleeves, sleeveless, cap sleeves)")
    Length: Optional[str] = Field(default=None, description="The length (e.g., floor length, short, midi length)")
    Embellishment: Optional[List[str]] = Field(default_factory=list, description="Decorations or details (e.g., floral embroidery, open back)")
    Color: Optional[List[str]] = Field(default_factory=list, description="Colors mentioned in the text (e.g., sage, dusty blue)")
    Category: Optional[str] = Field(default=None, description="The type of clothing (e.g., bridesmaid dress, evening gown)")

class ExtractionAgent:
    def __init__(self):
        api_key = os.environ.get("GROQ_API_KEY", "")
        self.llm = ChatOpenAI(api_key=api_key, base_url="https://api.groq.com/openai/v1", model="llama-3.1-70b-versatile", temperature=0)
        self.use_llm = bool(api_key)  
    def extract_with_spacy(self, text: str) -> dict:
        model = get_nlp()
        if not model:
            print("Custom ner model not found !!")
            return {}
            
        doc = model(text)
        
        extracted = {}
        for ent in doc.ents:
            label = ent.label_
            value = ent.text
            
            if label in ["Embellishment", "Color"]:
                if label not in extracted:
                    extracted[label] = []
                if value not in extracted[label]:
                    extracted[label].append(value)
            else:
                if label not in extracted:
                    extracted[label] = value
                    
        return extracted
        
    def check_missing_fields(self, extracted: dict) -> list:
        all_fields = ["Silhouette", "Fabric", "Neckline", "Sleeve", "Length", "Embellishment", "Color", "Category"]
        missing = [f for f in all_fields if f not in extracted or not extracted[f]]
        return missing

    def extract_with_llm_fallback(self, text: str, missing_fields: list, current_extracted: dict) -> dict:
        """Step 2: If the ML model missed stuff, use LLM tool calling to fill the gaps"""
        
        prompt_str = f"""
        You are a strict Data Extraction AI. 
        Extract the requested attributes from the product description below.
        Description: "{text}"
        
        We are specifically looking for these missing fields: {', '.join(missing_fields)}.
        
        CRITICAL RULE: If a field is NOT explicitly mentioned in the text, you MUST return null/None for that field. 
        DO NOT guess. DO NOT hallucinate.
        """
        
        try:
            print(f"Triggering llm fallback : {missing_fields}")
            structured_llm = self.llm.with_structured_output(ProductAttributes)
            llm_result = structured_llm.invoke(prompt_str)
            
            llm_dict = llm_result.model_dump(exclude_none=True)
            
            for key, value in llm_dict.items():
                if key in missing_fields and value:
                    current_extracted[key] = value
            return current_extracted
        except Exception as e:
            print(f"LLM Fallback failed: {e}")
            return current_extracted

    def run(self, text: str) -> ProductAttributes:
        extracted_data = self.extract_with_spacy(text)
        missing = self.check_missing_fields(extracted_data)
        if missing and self.use_llm:
            extracted_data = self.extract_with_llm_fallback(text, missing, extracted_data)
        return ProductAttributes(**extracted_data)
