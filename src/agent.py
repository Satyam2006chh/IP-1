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

# Define the Output Schema exactly as requested
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
        # We use Cerebras API (which is OpenAI compatible) for lightning fast inference
        api_key = os.environ.get("CEREBRAS_API_KEY", "")
        self.llm = ChatOpenAI(
            api_key=api_key,
            base_url="https://api.cerebras.ai/v1",
            model="llama3.1-70b", # Using 70b for high accuracy in tool calling
            temperature=0
        )
        self.use_llm = bool(api_key)
        
    def extract_with_spacy(self, text: str) -> dict:
        """Step 1: Try to extract everything using our fast custom trained model"""
        model = get_nlp()
        if not model:
            print("Warning: Custom NER model not found.")
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
        
    def check_missing_critical_fields(self, extracted: dict) -> list:
        """Check what the ML model missed that we really care about."""
        # For a product, we usually always want a Category and Fabric
        critical_fields = ["Category", "Fabric", "Color"]
        missing = [f for f in critical_fields if f not in extracted or not extracted[f]]
        return missing

    def extract_with_llm_fallback(self, text: str, missing_fields: list, current_extracted: dict) -> dict:
        """Step 2: If the ML model missed stuff, use LLM tool calling to fill the gaps"""
        
        prompt_str = f"""
        You are a fashion AI assistant. Extract attributes from the product description.
        Description: "{text}"
        
        We are specifically missing these fields: {', '.join(missing_fields)}.
        Please extract them accurately.
        """
        
        try:
            print(f"Triggering LLM Fallback for missing fields: {missing_fields}")
            structured_llm = self.llm.with_structured_output(ProductAttributes)
            llm_result = structured_llm.invoke(prompt_str)
            
            llm_dict = llm_result.model_dump(exclude_none=True)
            
            for key, value in llm_dict.items():
                # Only use the LLM to fill in the missing gaps, or if the list is empty
                if key in missing_fields and value:
                    current_extracted[key] = value
                    
            return current_extracted
        except Exception as e:
            print(f"LLM Fallback failed: {e}")
            return current_extracted

    def run(self, text: str) -> ProductAttributes:
        """The main workflow"""
        # Node 1: Fast ML Extraction
        extracted_data = self.extract_with_spacy(text)
        
        # Node 2: Validation
        missing = self.check_missing_critical_fields(extracted_data)
        
        # Node 3: Smart Fallback
        if missing and self.use_llm:
            extracted_data = self.extract_with_llm_fallback(text, missing, extracted_data)
        
        # Node 4: Final Validation and Return
        return ProductAttributes(**extracted_data)
