"""
minify.py
---------
Simple utility that removes unnecessary whitespace from a JSON file.
The script reads a JSON file and writes an equivalent JSON file with
no extraneous whitespace (compact JSON). The character encoding is
preserved (UTF-8) and non-ASCII characters are kept (no escaping).

Usage:
    python minify.py input.json output.json

This file intentionally keeps the implementation minimal: it reads the
source JSON to memory, then writes a compact JSON representation to
the output path using compact separators.
"""

import json
import sys
from pathlib import Path


def minify_json(input_path: str, output_path: str):
    """Read JSON from `input_path` and write a compact/`minified` version to
    `output_path`.

    This preserves the original data model while removing indent and
    separators (space after commas and colons), which is useful for
    minimizing disk usage or shipping smaller JSON payloads.
    """

    with open(input_path, "r", encoding="utf-8") as f_in:
        data = json.load(f_in)

    with open(output_path, "w", encoding="utf-8") as f_out:
        json.dump(data, f_out, separators=(",", ":"), ensure_ascii=False)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(f"Usage: {Path(sys.argv[0]).name} input.json output.json")
        sys.exit(1)

    minify_json(sys.argv[1], sys.argv[2])
