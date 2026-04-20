"""CLI validator — check a JSONL file against the schema contract.

Give this tool to the serving team so they can self-test their log output
before deploying. Exits non-zero if any line is invalid.

Usage:
    python validate_schema.py --file path/to/events.jsonl --type categorization
    python validate_schema.py --file path/to/feedback.jsonl --type anomaly_feedback
"""
import argparse
import json
import sys
from pathlib import Path

from pydantic import ValidationError

from schemas import AnomalyFeedbackEvent, CategorizationEvent


MODELS = {
    "categorization":   CategorizationEvent,
    "anomaly_feedback": AnomalyFeedbackEvent,
}


def main():
    parser = argparse.ArgumentParser(description="Validate production-log JSONL against the schema.")
    parser.add_argument("--file", required=True, type=Path,
                        help="Path to the .jsonl file to validate")
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
    with open(args.file) as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            total += 1
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as e:
                errors.append((lineno, f"invalid JSON: {e}"))
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
