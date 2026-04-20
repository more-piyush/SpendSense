"""Set active production model IDs for categorization and trend tasks."""

import argparse
import json

from utils import load_active_models, set_active_models


def main():
    parser = argparse.ArgumentParser(description="Update active model selections")
    parser.add_argument(
        "--registry-path",
        default="s3://mlflow/registry",
        help="Path to the shared model registry directory",
    )
    parser.add_argument(
        "--categorization",
        dest="active_categorization_model",
        help="Model ID to mark active for categorization",
    )
    parser.add_argument(
        "--trend",
        dest="active_trend_model",
        help="Model ID to mark active for trend detection",
    )
    args = parser.parse_args()

    updated = set_active_models(
        registry_path=args.registry_path,
        active_categorization_model=args.active_categorization_model,
        active_trend_model=args.active_trend_model,
    )
    print(json.dumps(updated, indent=2))
    print(json.dumps(load_active_models(args.registry_path), indent=2))


if __name__ == "__main__":
    main()
