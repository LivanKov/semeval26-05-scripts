import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-i', nargs='+', required=True, help='Paths to JSON files')
parser.add_argument('-o', required=True, help='Output file path')
args = parser.parse_args()

acc_sum = spear_sum = 0
for f in args.i:
    data = json.load(open(f))
    acc_sum += data['accuracy']
    spear_sum += data['spearman']

n = len(args.i)
result = {'accuracy': acc_sum / n, 'spearman': spear_sum / n}

with open(args.o, 'w') as out:
    json.dump(result, out)

print(f"✓ Averaged {n} files -> {args.o}")