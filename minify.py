import json
import sys
from pathlib import Path

def minify_json(input_path: str, output_path: str):
    with open(input_path, "r", encoding="utf-8") as f_in:
        data = json.load(f_in) 

    with open(output_path, "w", encoding="utf-8") as f_out:
        json.dump(data, f_out, separators=(',', ':'), ensure_ascii=False)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(f"Usage: {Path(sys.argv[0]).name} input.json output.json")
        sys.exit(1)

    minify_json(sys.argv[1], sys.argv[2])
