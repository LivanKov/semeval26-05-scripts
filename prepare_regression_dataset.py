import json
import pandas as pd
import sys
import random

# ---- PATHS ----
DEV_PATH = "data/dev.cleaned.minified.json"
SOLUTION_PATH = "input/ref/solution.jsonl"
OUTPUT_PATH = "output/train_llm_regression.jsonl"

# ---- CONFIG ----
SHUFFLE_LABELS = True  # Mélanger l'ordre des annotations (ordre non important)
RANDOM_SEED = 42       # Pour reproductibilité

def load_dev_data(path):
    """Load dev dataset from JSON dict file."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            dev = json.load(f)
        df = pd.DataFrame.from_dict(dev, orient="index")
        print(f"✓ Loaded DEV set: {len(df)} rows")
        return df
    except Exception as e:
        print(f"Error loading {path}: {e}")
        sys.exit(1)

def load_solution_data(path):
    """Load solution data from JSONL file."""
    try:
        sol_df = pd.read_json(path, lines=True)
        sol_df = sol_df.rename(columns={"id": "sample_id", "judgment": "label"})
        print(f"✓ Loaded solution set: {len(sol_df)} rows")
        return sol_df
    except Exception as e:
        print(f"Error loading solution {path}: {e}")
        sys.exit(1)

def build_input(row):
    """Format row data into LLM input text."""
    return (
        f"Homonym: {row.get('homonym', '')}\n"
        f"Judged meaning: {row.get('judged_meaning', '')}\n"
        f"Context: {row.get('precontext', '')}\n"
        f"Sentence: {row.get('sentence', '')}\n"
        f"Ending: {row.get('ending', '')}\n"
        f"Example meaning sentence: {row.get('example_sentence', '')}"
    ).strip()

def expand_annotations(merged_df, shuffle=True, seed=42):
    """
    Crée un échantillon d'entraînement pour CHAQUE annotation individuelle.
    
    Args:
        merged_df: DataFrame avec colonnes input_text et label (liste)
        shuffle: Si True, mélange l'ordre des annotations dans chaque liste
        seed: Seed pour reproductibilité
    
    Returns:
        Liste de dictionnaires avec text et label individuel
    """
    if seed:
        random.seed(seed)
    
    output_records = []
    
    for _, row in merged_df.iterrows():
        text = row['input_text']
        labels = row['label']
        
        # Vérifier que label est une liste
        if not isinstance(labels, list):
            labels = [labels]
        
        # Filtrer les valeurs valides (1-5)
        valid_labels = [l for l in labels if isinstance(l, (int, float)) and 1 <= l <= 5]
        
        if not valid_labels:
            continue
        
        # Optionnel: mélanger l'ordre des annotations
        if shuffle:
            valid_labels = valid_labels.copy()
            random.shuffle(valid_labels)
        
        # Créer un échantillon pour chaque annotation
        for label in valid_labels:
            output_records.append({
                "text": text,
                "label": float(label)
            })
    
    return output_records

def main():
    dev_df = load_dev_data(DEV_PATH)
    sol_df = load_solution_data(SOLUTION_PATH)
    
    # Convert join keys to string
    dev_df["sample_id"] = dev_df["sample_id"].astype(str)
    sol_df["sample_id"] = sol_df["sample_id"].astype(str)
    
    # Merge datasets
    merged = dev_df.merge(sol_df, on="sample_id", how="inner")
    print(f"✓ Merged dataset: {len(merged)} samples")
    
    # Warn about lost samples
    lost = len(dev_df) - len(merged)
    if lost > 0:
        print(f"⚠ Warning: lost {lost} samples during merge ({lost/len(dev_df)*100:.1f}%)")
    
    # Build text fields
    merged["input_text"] = merged.apply(build_input, axis=1)
    
    # Expand: créer un échantillon par annotation
    output_records = expand_annotations(
        merged, 
        shuffle=SHUFFLE_LABELS, 
        seed=RANDOM_SEED
    )
    
    print(f"✓ Expanded to {len(output_records)} training samples from {len(merged)} unique texts")
    print(f"  → Average {len(output_records)/len(merged):.1f} annotations per sample")
    
    # Save output
    try:
        with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
            for record in output_records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"✓ Saved {len(output_records)} samples → {OUTPUT_PATH}")
    except IOError as e:
        print(f"Error writing output file: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()