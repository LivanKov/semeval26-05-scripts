#!/usr/bin/env python3
"""
clean_dev_json.py
------------------
Helper script used to remove scoring-related fields from Semeval dev
JSON objects. This is useful when you want a "clean" version of the
dataset for sharing or for downstream tasks that should not rely on
annotator judgments.

By default this script reads `data/dev.json` and prints a cleaned JSON
to stdout. You can pass a different input path and/or an output path
using positional and optional arguments. You can also override the
default list of fields to remove with `-f`/`--fields`.

Example:
    python3 clean_dev_json.py data/dev.json -o data/dev.cleaned.json

The script does not modify the input; if an output path is supplied the
cleaned JSON is written there, otherwise the script prints to stdout.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, MutableMapping, Set


def drop_fields(entries: Dict[str, MutableMapping[str, object]], fields: Iterable[str]) -> None:
    """Remove the specified fields from every nested mapping in ``entries``.

    Parameters:
    - entries: Mapping from sample-id to a mapping of properties (the JSON
      objects loaded from dev.json). Each nested value is expected to be a
      MutableMapping (e.g., dict) and if not, the value is ignored.
    - fields: An iterable of field names (strings) that should be removed
      from each nested mapping. Removal is performed in-place.

    This function is intentionally permissive: it will skip entries whose
    values are not mappings and it will silently ignore field names that
    are not present in an entry.
    """

    field_set: Set[str] = set(fields)
    for entry in entries.values():
        if not isinstance(entry, MutableMapping):
            continue
        for field in field_set:
            entry.pop(field, None)


def main() -> None:
    """Command-line entrypoint for cleaning a dev.json file.

    This function parses CLI arguments, reads the input JSON file, removes
    the specified fields from each entry using `drop_fields`, and then
    writes the cleaned JSON to stdout or the provided output path.
    """

    # Create argument parser and configure accepted flags
    parser = argparse.ArgumentParser(
        description=(
            "Remove scoring-related fields from Semeval dev.json objects. "
            "Defaults to data/dev.json and prints to stdout."
        )
    )
    parser.add_argument(
        "input",
        nargs="?",
        default="data/dev.json",
        help="Path to the source dev.json file (default: data/dev.json)",
    )
    parser.add_argument(
        "-o",
        "--output",
        help="Destination for the cleaned JSON (default: stdout)",
    )
    parser.add_argument(
        "-f",
        "--fields",
        nargs="*",
        default=["choices", "average", "stdev", "nonsensical", "sample_id"],
        help="Optional explicit list of field names to strip",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        parser.error(f"Input file {input_path} does not exist")

    with input_path.open(encoding="utf-8") as fh:
        data = json.load(fh)

    drop_fields(data, args.fields)

    output_text = json.dumps(data, indent=4, ensure_ascii=False)

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(output_text + "\n", encoding="utf-8")
    else:
        print(output_text)


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
