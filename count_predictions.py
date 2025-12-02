import sys
import json
from collections import Counter

def count_file(path):
    counts = Counter()
    with open(path, "r") as f:
        for line in f:
            try:
                obj = json.loads(line)
                pred = int(obj["prediction"])
                counts[pred] += 1
            except:
                continue
    return counts

def merge_counts(list_of_counts):
    total = Counter()
    for c in list_of_counts:
        total.update(c)
    return total

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python3 count_predictions.py file1.jsonl")
        print("  python3 count_predictions.py file1.jsonl file2.jsonl file3.jsonl")
        sys.exit()

    files = sys.argv[1:]

    all_counts = []
    for f in files:
        c = count_file(f)
        all_counts.append(c)
        print(f"\nCounts for {f}:")
        for n in range(1, 6):
            print(f"  {n}: {c[n]}")

    if len(files) > 1:
        total = merge_counts(all_counts)
        print("\n===== TOTAL ACROSS ALL FILES =====")
        for n in range(1, 6):
            print(f"  {n}: {total[n]}")