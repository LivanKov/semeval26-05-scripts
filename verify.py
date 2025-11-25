import json
import argparse

def verify_jsonl(filename):
    with open(filename, 'r') as f:
        for i, line in enumerate(f):
            data = json.loads(line)
            assert data['id'] == str(i), f"Line {i}: id mismatch"
            assert 1 <= data['prediction'] <= 5, f"Line {i}: prediction out of range"
        assert i == 587, f"Expected 587 entries, got {i + 1}"
    print("✓ File valid")

parser = argparse.ArgumentParser()
parser.add_argument('-i', required=True, help='Path to JSONL file')
args = parser.parse_args()

verify_jsonl(args.i)