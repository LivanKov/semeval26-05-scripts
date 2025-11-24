#!/usr/bin/env python3
"""Strip scoring fields from dev.json objects."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, MutableMapping, Set


def drop_fields(entries: Dict[str, MutableMapping[str, object]], fields: Iterable[str]) -> None:
    """Remove the specified fields from every nested mapping in ``entries``."""

    field_set: Set[str] = set(fields)
    for entry in entries.values():
        if not isinstance(entry, MutableMapping):
            continue
        for field in field_set:
            entry.pop(field, None)


def main() -> None:
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
