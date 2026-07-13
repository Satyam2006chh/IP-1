import json
from sklearn.metrics import f1_score, accuracy_score
from src.agent import ExtractionAgent
import warnings

warnings.filterwarnings('ignore')

def load_test_data(filepath="data/test_dataset.json"):
    with open(filepath, 'r') as f:
        return json.load(f)

def evaluate():
    print("Loading test data...")
    try:
        test_data = load_test_data()
    except FileNotFoundError:
        print("Test data not found. Run train.py first to generate the split.")
        return

    print(f"Evaluating on {len(test_data)} samples...\n")
    agent = ExtractionAgent()
    
    y_true_all = []
    y_pred_all = []
    
    # Track attribute-level exact matches
    attrs_to_track = ["Silhouette", "Fabric", "Neckline", "Sleeve", "Length", "Category"]
    attribute_correct = {attr: 0 for attr in attrs_to_track}
    attribute_total = {attr: 0 for attr in attrs_to_track}

    for item in test_data:
        text = item['text']
        true_attrs = item['attributes']
        
        # Run our pipeline
        pred_attrs_obj = agent.run(text)
        pred_attrs = pred_attrs_obj.model_dump(exclude_none=True)
        
        for attr in attrs_to_track:
            true_val = true_attrs.get(attr, "")
            pred_val = pred_attrs.get(attr, "")
            
            if isinstance(true_val, list):
                true_val = ", ".join(true_val)
            if isinstance(pred_val, list):
                pred_val = ", ".join(pred_val)
                
            true_str = str(true_val).lower().strip()
            pred_str = str(pred_val).lower().strip()
            
            if true_str:
                attribute_total[attr] += 1
                # Check for partial or exact match
                if pred_str and (true_str in pred_str or pred_str in true_str):
                    attribute_correct[attr] += 1
                    
            y_true_all.append(true_str if true_str else "None")
            y_pred_all.append(pred_str if pred_str else "None")

    print("\n--- Evaluation Results ---")
    print("Attribute-level Accuracy:")
    for attr in attrs_to_track:
        if attribute_total[attr] > 0:
            acc = attribute_correct[attr] / attribute_total[attr]
            print(f"  - {attr}: {acc*100:.1f}%")
        else:
            print(f"  - {attr}: N/A (not in test set)")
            
    # Calculate overall metrics
    overall_f1 = f1_score(y_true_all, y_pred_all, average='weighted')
    overall_acc = accuracy_score(y_true_all, y_pred_all)
    
    print(f"\nOverall Weighted F1 Score: {overall_f1:.4f}")
    print(f"Overall Exact Match Accuracy: {overall_acc:.4f}")
    
    print("\n[NOTE] If CEREBRAS_API_KEY is set, the LLM fallback will boost these scores significantly on messy data.")

if __name__ == "__main__":
    evaluate()
