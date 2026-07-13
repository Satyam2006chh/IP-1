from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.agent import ExtractionAgent, ProductAttributes
import os

app = FastAPI(
    title="Product Attribute Extraction API",
    description="API for extracting attributes from product descriptions using a hybrid ML + GenAI approach.",
    version="1.0.0"
)

# Initialize our hybrid agent
agent = ExtractionAgent()

class ExtractRequest(BaseModel):
    description: str

@app.post("/extract", response_model=ProductAttributes)
async def extract_attributes(request: ExtractRequest):
    if not request.description or not request.description.strip():
        raise HTTPException(status_code=400, detail="Product description cannot be empty")
        
    try:
        # The agent handles the custom spaCy model and the LLM fallback automatically
        result = agent.run(request.description)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy", "llm_fallback_enabled": agent.use_llm}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api:app", host="0.0.0.0", port=8000, reload=True)
