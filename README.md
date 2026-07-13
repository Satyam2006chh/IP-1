# Product Attribute Extraction Pipeline

This repository contains a robust, hybrid AI pipeline for extracting structured attributes (Silhouette, Fabric, Neckline, Sleeve, Length, Embellishment, Color, Category) from unstructured product descriptions.

## Architecture: The "Hybrid Fallback" Approach

To balance latency, cost, and high accuracy, this pipeline utilizes a two-step Agentic workflow using **FastAPI** and **LangGraph**:

1. **The Fast Path (Custom ML):** We trained a lightweight, custom Named Entity Recognition (NER) model using `spaCy`. When a request comes in, the text is first processed by this model. It is extremely fast and cost-effective, handling the majority of standard extractions instantly.
2. **The Smart Path (LLM Tool Calling Fallback):** Real-world e-commerce data is messy. If our primary ML model misses critical fields (like `Category`, `Fabric`, or `Color`), our LangGraph agent dynamically triggers a fallback to a Large Language Model (configured to use the lightning-fast **Cerebras API**). The LLM uses strict tool-calling to extract only the missing attributes. 

This ensures we get the speed of a traditional trained model with the reasoning capabilities of modern GenAI for edge cases.

## Repository Structure
- `/data/`: Contains the generated training data (`dataset.json`) and the holdout test set.
- `/models/custom_ner/`: The compiled weights for the custom-trained spaCy model.
- `/src/train.py`: The script used to train the custom ML model on the dataset.
- `/src/agent.py`: The LangGraph agent that orchestrates the ML model and the LLM fallback.
- `/src/api.py`: The FastAPI server.
- `/src/evaluate.py`: The script to calculate F1 scores and accuracy.

## How to Run the API

1. **Install dependencies:**
   ```bash
   pip install fastapi uvicorn spacy scikit-learn langgraph langchain-openai pydantic
   ```
   *(Note: The custom spaCy model is already trained and saved in `/models/`)*

2. **Set your API Key (for the LLM fallback):**
   This project uses Cerebras for ultra-low latency LLM inference.
   **Windows (PowerShell):**
   ```powershell
   $env:CEREBRAS_API_KEY="your-api-key-here"
   ```
   **Mac/Linux:**
   ```bash
   export CEREBRAS_API_KEY="your-api-key-here"
   ```

3. **Start the FastAPI server:**
   ```bash
   python -m uvicorn src.api:app --reload
   ```

4. **Test the API:**
   Send a POST request to `http://127.0.0.1:8000/extract`:
   ```json
   {
       "description": "Short velvet evening gown with a sweetheart neckline in burgundy."
   }
   ```

## Evaluation Metrics

We evaluated the system on a holdout test set of 10 complex product descriptions. 

*Run the evaluation yourself using:* `python src/evaluate.py`

**Typical Results (Without LLM Fallback - ML Only):**
- Overall F1 Score: ~0.85
- The custom ML model performs exceptionally well on standard patterns but occasionally struggles with rare vocabulary given the small dataset size (50 training examples).

**With LLM Fallback Enabled:**
- Overall F1 Score: **> 0.95**
- The LLM successfully catches the edge cases missed by the fast ML model.

### Common Failure Cases (When the LLM Fallback is disabled)
1. **Implicit Categories:** Descriptions like "A perfect fit for your special day" might lack an explicit category (like "dress"). The ML model misses this, but the LLM fallback can infer it.
2. **Color vs. Fabric Confusion:** Rare color names (e.g., "champagne", "ivory") can sometimes be misclassified if they were not sufficiently represented in the small training dataset. 
3. **Overlapping Entities:** "Lace applique" might be tagged as a Fabric ("Lace") instead of an Embellishment. We handle this in the agent by prioritizing longer entity matches during training data preparation, but ambiguities remain.
