import json
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-i', required=True, help='Path to JSONL file')
args = parser.parse_args()

counts = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}

with open(args.i, 'r') as f:
    for line in f:
        data = json.loads(line)
        for label in data['label']:
            counts[label] += 1

total = sum(counts.values())
percentages = [counts[i] / total * 100 for i in range(1, 6)]

plt.bar(range(1, 6), percentages, color='steelblue')
plt.xlabel('Label')
plt.ylabel('Percentage (%)')
plt.title('Distribution of Labels')
plt.xticks(range(1, 6))
plt.tight_layout()
plt.savefig('distribution.png', dpi=300, bbox_inches='tight')
plt.show()