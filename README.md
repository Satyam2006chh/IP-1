# Product Attribute Extraction Pipeline

This repository contains a robust, hybrid AI pipeline for extracting structured attributes (Silhouette, Fabric, Neckline, Sleeve, Length, Embellishment, Color, Category) from unstructured product descriptions.

## Architecture: The "Hybrid Fallback" Approach

To balance latency, cost, and high accuracy, this pipeline utilizes a two-step Agentic workflow using **FastAPI** and **LangChain**:

1. **The Fast Path (Custom ML):** We trained a lightweight, custom Named Entity Recognition (NER) model using `spaCy`. When a request comes in, the text is first processed by this model. It is extremely fast and cost-effective, handling the majority of standard extractions instantly.
2. **The Smart Path (LLM Tool Calling Fallback):** Real-world e-commerce data is messy. If our primary ML model misses critical fields, our agent dynamically triggers a fallback to a Large Language Model (configured to use the lightning-fast **Groq API** with Llama 3.3). The LLM uses a strict Anti-Hallucination prompt and Tool Calling to accurately extract only the missing attributes.

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
   pip install fastapi uvicorn spacy scikit-learn langchain-groq pydantic
   ```
   *(Note: The custom spaCy model is already trained and saved in `/models/`)*

### 2. Set your Groq API Key
We use the **Groq API** with the Llama 3.3 model as an intelligent fallback mechanism if the custom NER model misses fields. It is blazingly fast.

```bash
# Windows
$env:GROQ_API_KEY="your_groq_api_key"
```

### 3. Run the API
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
Because the custom ML model was trained on a small synthetic dataset (50 examples), it suffers from known ML limitations:
1. **Label Confusion (Overfitting to Position):** The model occasionally tags a Color as a Length (e.g., tagging "Beautiful red" as a `Length`). It memorized word order templates rather than the semantic meaning of the words.
2. **Missed Extractions (False Negatives):** It completely ignores attributes (like "halter neck") if that specific phrasing was not present in the tiny 50-item training set.
3. **Boundary Detection Issues:** It struggles to predict the exact start and end of multi-word phrases (extracting "embroidery" instead of "floral embroidery").

**How these are solved:**
In this project, the **Groq Llama 3.3 GenAI Agent** acts as a safety net to catch and fix these errors using strict Guardrail Prompts to prevent hallucination. 
In a pure-ML production environment, these issues would be solved by scaling the training dataset to 10,000+ real-world examples, introducing varied sentence structures, and potentially upgrading to a Transformer-based NER architecture (like `spaCy-transformers`).
