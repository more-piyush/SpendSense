"""Entrypoint for scheduled retraining-data preparation jobs."""

from __future__ import annotations

import argparse
import subprocess
import sys


def run(script: str) -> int:
    return subprocess.call([sys.executable, script])


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build versioned retraining datasets from MinIO logs + base training parquet files."
    )
    parser.add_argument(
        "task",
        choices=["categorization", "trend", "both"],
        help="Which retraining-data dataset pipeline to run.",
    )
    args = parser.parse_args()

    if args.task == "categorization":
        return run("build_categorization_dataset.py")
    if args.task == "trend":
        return run("build_trend_detection_dataset.py")

    first = run("build_categorization_dataset.py")
    if first != 0:
        return first
    return run("build_trend_detection_dataset.py")


if __name__ == "__main__":
    raise SystemExit(main())
