"""
verify.py
---------
Simple utility to validate a newline-delimited JSON (JSONL) predictions
file. The script ensures that each line is a valid JSON object with an
`id` string matching the expected sequence (starting at 0) and a numeric
`prediction` between 1 and 5. It also verifies that the file contains the
expected number of lines (currently 588 entries with index 0..587).

Usage:
    python verify.py -i predictions.jsonl
"""

import json
import argparse

def verify_jsonl(filename):
    """Verify a JSONL file for id and prediction format.

    The function opens the given filename and iterates each line. For every
    line it verifies two conditions using assertions:
      - data['id'] equals the expected index (stringified): ensures ordering
        and stable indexing (e.g., 0..N-1)
      - data['prediction'] is an integer in 1..5 (inclusive)

    At the end, it asserts that exactly 588 lines exist (index 0..587).
    On success, prints a short success message. Assertion failures raise
    AssertionError with an explanatory message.
    """
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