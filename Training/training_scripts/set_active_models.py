"""Set active production model IDs or auto-select winners from the registry."""

import argparse
import json

from export_serving_artifacts import materialize_active_serving_artifacts
from utils import load_active_models, set_active_models, update_active_model_selection


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
        "--categorization-registry-id",
        dest="active_categorization_registry_id",
        help="Registry ID to mark active for categorization",
    )
    parser.add_argument(
        "--trend",
        dest="active_trend_model",
        help="Model ID to mark active for trend detection",
    )
    parser.add_argument(
        "--trend-registry-id",
        dest="active_trend_registry_id",
        help="Registry ID to mark active for trend detection",
    )
    parser.add_argument(
        "--auto-select",
        action="store_true",
        help="Pick the current best categorization and trend models from registry metrics",
    )
    args = parser.parse_args()

    if args.auto_select:
        updated = update_active_model_selection(args.registry_path)
    else:
        updated = set_active_models(
            registry_path=args.registry_path,
            active_categorization_model=args.active_categorization_model,
            active_trend_model=args.active_trend_model,
            active_categorization_registry_id=args.active_categorization_registry_id,
            active_trend_registry_id=args.active_trend_registry_id,
        )
    updated = materialize_active_serving_artifacts(args.registry_path)
    print(json.dumps(updated, indent=2))
    print(json.dumps(load_active_models(args.registry_path), indent=2))


if __name__ == "__main__":
    main()
