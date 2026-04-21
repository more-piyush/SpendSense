"""CLI validator — check feedback-event files against the schema contract.

Give this to the serving team so they can self-test their log output before
deploying. Exits non-zero if any record is invalid.

Accepts either a single JSON file (one event per file, as production writes)
or a JSONL file (one event per line, for test fixtures).

Usage:
    python validate_schema.py --file path/to/event.json  --type categorization_feedback
    python validate_schema.py --file path/to/events.jsonl --type categorization_feedback
    python validate_schema.py --file path/to/event.json  --type trend_feedback
"""
import argparse
import json
import sys
from pathlib import Path

from pydantic import ValidationError

from schemas import CategorizationFeedbackEvent, TrendFeedbackEvent


MODELS = {
    "categorization_feedback": CategorizationFeedbackEvent,
    "trend_feedback":          TrendFeedbackEvent,
}


def _load_records(path: Path):
    """Yield (lineno, record_dict). Auto-detects single-object .json vs JSONL."""
    text = path.read_text()
    stripped = text.lstrip()
    if stripped.startswith("{"):
        # Try whole-file single object first
        try:
            yield 1, json.loads(text)
            return
        except json.JSONDecodeError:
            pass
    # Fall back to JSONL
    for lineno, line in enumerate(text.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            yield lineno, json.loads(line)
        except json.JSONDecodeError as e:
            yield lineno, {"__decode_error__": str(e)}


def main():
    parser = argparse.ArgumentParser(
        description="Validate production-log feedback events against the schema."
    )
    parser.add_argument("--file", required=True, type=Path,
                        help="Path to the .json or .jsonl file to validate")
    parser.add_argument("--type", required=True, choices=MODELS.keys(),
                        help="Which event schema to validate against")
    parser.add_argument("--max-errors", type=int, default=20,
                        help="Max errors to print before stopping (default 20)")
    args = parser.parse_args()

    if not args.file.exists():
        print(f"ERROR: file not found: {args.file}", file=sys.stderr)
        sys.exit(2)

    model = MODELS[args.type]

    total = 0
    errors = []
    for lineno, rec in _load_records(args.file):
        total += 1
        if "__decode_error__" in rec:
            errors.append((lineno, f"invalid JSON: {rec['__decode_error__']}"))
            continue
        try:
            model(**rec)
        except ValidationError as e:
            first = e.errors()[0]
            errors.append((lineno, f"{first['loc']}: {first['msg']}"))

    print(f"Checked {total} records")
    print(f"Valid:   {total - len(errors)}")
    print(f"Invalid: {len(errors)}")

    if errors:
        print(f"\nFirst {min(args.max_errors, len(errors))} errors:")
        for lineno, msg in errors[:args.max_errors]:
            print(f"  line {lineno}: {msg}")
        sys.exit(1)

    print("\nAll records valid.")
    sys.exit(0)


if __name__ == "__main__":
    main()
