import sys
import json
from collections import Counter
import matplotlib.pyplot as plt

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

def merge_counts(counts_list):
    total = Counter()
    for c in counts_list:
        total.update(c)
    return total

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python3 plot_predictions.py file1.jsonl file2.jsonl file3.jsonl")
        sys.exit()

    files = sys.argv[1:]
    all_counts = [count_file(f) for f in files]
    total = merge_counts(all_counts)

    # ---- Prepare data ----
    labels = [1, 2, 3, 4, 5]
    values = [total[x] for x in labels]
    total_sum = sum(values)
    percentages = [(v / total_sum) * 100 for v in values]

    # ---- Plot ----
    plt.figure(figsize=(8, 5))
    plt.bar(labels, percentages)
    plt.xlabel("Prediction")
    plt.ylabel("Percentage (%)")
    plt.title("Distribution of Predictions (Total Across All Files)")
    plt.xticks(labels)
    plt.ylim(0, max(percentages) * 1.2)

    plt.tight_layout()
    plt.savefig("prediction_distribution.png")
    plt.show()

    print("Saved chart as prediction_distribution.png")