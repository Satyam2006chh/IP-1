import json
import spacy
from spacy.training.example import Example
import random
import os

def load_data(filepath):
    with open(filepath, 'r') as f:
        return json.load(f)

def prepare_spacy_data(dataset):
    """
    Converts our JSON dataset into the format spaCy needs for NER training.
    Format: (text, {"entities": [(start, end, label), ...]})
    """
    spacy_data = []
    for item in dataset:
        text = item['text']
        attributes = item['attributes']
        entities = []
        
        for label, value in attributes.items():
            if isinstance(value, list):
                values_to_find = value
            else:
                values_to_find = [value]
                
            for v in values_to_find:
                start = text.lower().find(v.lower())
                if start != -1:
                    end = start + len(v)
                    entities.append((start, end, label))
                    
        # Sort by length descending to prioritize longer matches if they overlap
        entities = sorted(entities, key=lambda x: (x[1]-x[0]), reverse=True)
        filtered_entities = []
        occupied_indices = set()
        
        for start, end, label in entities:
            # Check for overlaps
            if not any(i in occupied_indices for i in range(start, end)):
                filtered_entities.append((start, end, label))
                occupied_indices.update(range(start, end))
                
        spacy_data.append((text, {"entities": filtered_entities}))
    return spacy_data

def train_ner_model(data, output_dir="models/custom_ner", iterations=20):
    print("Preparing data for spaCy...")
    train_data = prepare_spacy_data(data)
    
    # Start with a blank English model
    nlp = spacy.blank("en")
    
    if "ner" not in nlp.pipe_names:
        ner = nlp.add_pipe("ner", last=True)
    else:
        ner = nlp.get_pipe("ner")
        
    # Add labels
    for _, annotations in train_data:
        for ent in annotations.get("entities"):
            ner.add_label(ent[2])
            
    pipe_exceptions = ["ner"]
    unaffected_pipes = [pipe for pipe in nlp.pipe_names if pipe not in pipe_exceptions]
    
    print("Starting training...")
    with nlp.disable_pipes(*unaffected_pipes):
        optimizer = nlp.begin_training()
        for itn in range(iterations):
            random.shuffle(train_data)
            losses = {}
            for text, annotations in train_data:
                doc = nlp.make_doc(text)
                example = Example.from_dict(doc, annotations)
                nlp.update([example], drop=0.3, sgd=optimizer, losses=losses)
            print(f"Iteration {itn + 1} - Loss: {losses['ner']:.4f}")
            
    os.makedirs(output_dir, exist_ok=True)
    nlp.to_disk(output_dir)
    print(f"Model saved to {output_dir}")

if __name__ == "__main__":
    dataset = load_data("data/dataset.json")
    
    # Split: 50 for training, rest (10) for testing
    train_split = dataset[:50]
    test_split = dataset[50:]
    
    with open('data/test_dataset.json', 'w') as f:
        json.dump(test_split, f, indent=4)
        
    train_ner_model(train_split)
