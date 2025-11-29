"""
make_majority_dev.py
--------------------
Create a simplified dataset that contains the majority vote for each
sample from the original `data/dev.json` file. The script reads `data/dev.json`,
computes a majority vote (ties broken by closeness to the original average),
and writes a reduced JSON mapping to `data/dev_majority.json` by default.

This file exposes two main functions:
- `choose_majority(choices, avg)` determines the majority vote from a list
    of numeric choices and breaks ties by selecting the candidate closest to
    the average value.
- `main()` orchestrates reading the input file, calling the majority
    selection, and writing the simplified JSON output file.
"""

import json
from collections import Counter

INPUT_FILE = "data/dev.json"
OUTPUT_FILE = "data/dev_majority.json"

def choose_majority(choices, avg):
        """Return the majority value among `choices`.

        If a single value occurs more frequently than others, return it. If
        there is a tie between multiple candidate values, choose the one whose
        numeric distance to `avg` (the sample's mean) is smallest.

        Parameters:
            - choices: sequence of numeric ratings (e.g., [1,2,3,3,4])
            - avg: numeric average value of the choices (float)

        Returns:
            The selected majority vote as a number (int/float depending on data).
        """
        counts = Counter(choices)
        max_count = max(counts.values())

        # Collect all candidates tied for the maximum count
        candidates = [r for r, c in counts.items() if c == max_count]

        # If a unique majority exists, return it immediately
        if len(candidates) == 1:
                return candidates[0]

        # Tie breaking: prefer the candidate that is closest to the sample average
        candidates.sort(key=lambda x: abs(x - avg))
        return candidates[0]

def main():
    """Read `INPUT_FILE`, compute the majority vote for each sample, and
    write a simplified dataset to `OUTPUT_FILE`.

    The resulting JSON contains only the essential fields and a
    `majority_vote` per sample. The script is intentionally simple and
    does not validate the input beyond expecting that `data/dev.json` is
    a mapping from ids to objects with `choices` and `average` fields.
    """

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
            "majority_vote": majority,
        }

    with open(OUTPUT_FILE, "w") as f:
        json.dump(simplified, f, indent=4)
    print(f"Saved simplified dataset to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()