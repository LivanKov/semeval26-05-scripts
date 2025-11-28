import json
from collections import Counter

INPUT_FILE = "data/dev.json" 
OUTPUT_FILE = "data/dev_majority.json"

def choose_majority(choices, avg):
    
    counts = Counter(choices)
    max_count = max(counts.values())

    
    candidates = [r for r, c in counts.items() if c == max_count]

    
    if len(candidates) == 1:
        return candidates[0]

    
    candidates.sort(key=lambda x: abs(x - avg))

    return candidates[0]

def main():
    with open(INPUT_FILE, "r") as f:
        data = json.load(f)

    simplified = {}

    for key, sample in data.items():
        choices = sample["choices"]
        avg = sample["average"]

        majority = choose_majority(choices, avg)

        
        simplified[key] = {
            "homonym": sample["homonym"],
            "judged_meaning": sample["judged_meaning"],
            "precontext": sample["precontext"],
            "sentence": sample["sentence"],
            "ending": sample["ending"],
            "example_sentence": sample["example_sentence"],
            "majority_vote": majority
        }

    with open(OUTPUT_FILE, "w") as f:
        json.dump(simplified, f, indent=4)

    print(f"Saved simplified dataset to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()