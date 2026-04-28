"""Copy active model selections between deployment channels."""

import argparse
import json

from export_serving_artifacts import materialize_active_serving_artifacts
from utils import load_active_models, set_active_models


def main():
    parser = argparse.ArgumentParser(description="Copy active model selections between registry files")
    parser.add_argument("--registry-path", default="s3://mlflow/registry")
    parser.add_argument("--source-file", default="active_models.json")
    parser.add_argument("--target-file", default="active_models_canary.json")
    parser.add_argument(
        "--tasks",
        nargs="*",
        choices=["categorization", "trend"],
        default=["categorization", "trend"],
    )
    args = parser.parse_args()

    source = load_active_models(args.registry_path, active_models_filename=args.source_file)
    kwargs = {
        "registry_path": args.registry_path,
        "active_models_filename": args.target_file,
    }

    if "categorization" in args.tasks and source.get("categorization"):
        kwargs["active_categorization_registry_id"] = source["categorization"]["registry_id"]
    if "trend" in args.tasks and source.get("trend"):
        kwargs["active_trend_registry_id"] = source["trend"]["registry_id"]

    updated = set_active_models(**kwargs)
    updated = materialize_active_serving_artifacts(
        args.registry_path,
        active_models_filename=args.target_file,
    )
    print(json.dumps(updated, indent=2))


if __name__ == "__main__":
    main()
