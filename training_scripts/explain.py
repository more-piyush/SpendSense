"""
explain.py — SHAP-based explainability for spending anomaly detection.

Generates human-readable explanations for anomaly alerts by computing
SHAP values for XGBoost predictions. Produces the top_factors output
described in the documentation (Section 4.2.3):
  - top_factors: Top 3 features by SHAP value, explaining the anomaly

Usage:
  python explain.py configs/explain.yaml --user-id <UUID>
  python explain.py configs/explain.yaml --all-users
  python explain.py configs/explain.yaml --anomaly-id <ID>

Also supports:
  - Batch explanation generation (nightly, alongside trend detection)
  - Single anomaly explanation (on-demand via API)
  - Global feature importance summary
"""

import sys
import os
import json
import argparse
from datetime import datetime

import numpy as np
import pandas as pd
import shap
import xgboost as xgb
import yaml

try:
    import mlflow
    from mlflow.tracking import MlflowClient
except ImportError:
    mlflow = None


# Feature names matching documentation Section 4.2.2
FEATURE_COLUMNS = [
    "current_spend", "rolling_mean_1m", "rolling_mean_3m", "rolling_mean_6m",
    "rolling_std_3m", "rolling_std_6m", "deviation_ratio", "share_of_wallet",
    "hist_share_of_wallet", "txn_count", "hist_txn_count_mean", "avg_txn_size",
    "hist_avg_txn_size", "days_since_last_txn", "month_of_year",
    "spending_velocity", "weekend_txn_ratio", "total_monthly_spend",
    "elevated_cat_count", "budget_utilization",
]

# Human-readable feature descriptions for alert text
FEATURE_DESCRIPTIONS = {
    "current_spend": "current spending",
    "rolling_mean_1m": "last month's average",
    "rolling_mean_3m": "3-month average",
    "rolling_mean_6m": "6-month average",
    "rolling_std_3m": "3-month spending variability",
    "rolling_std_6m": "6-month spending variability",
    "deviation_ratio": "deviation from 3-month average",
    "share_of_wallet": "share of total spending",
    "hist_share_of_wallet": "historical share of spending",
    "txn_count": "number of transactions",
    "hist_txn_count_mean": "typical transaction count",
    "avg_txn_size": "average transaction size",
    "hist_avg_txn_size": "typical transaction size",
    "days_since_last_txn": "days since last transaction",
    "month_of_year": "time of year",
    "spending_velocity": "spending pace this month",
    "weekend_txn_ratio": "weekend spending pattern",
    "total_monthly_spend": "total monthly spending",
    "elevated_cat_count": "number of elevated categories",
    "budget_utilization": "budget utilization",
}


# ============================================================
# MODEL LOADING
# ============================================================
def load_xgboost_model(config: dict) -> xgb.XGBRegressor:
    """Load the production XGBoost model."""
    model_path = config.get("model_path")

    if model_path and os.path.exists(model_path):
        model = xgb.XGBRegressor()
        model.load_model(model_path)
        print(f"[INFO] Loaded XGBoost model from {model_path}")
        return model

    # Try loading from MLflow
    if mlflow is not None:
        tracking_uri = config.get("mlflow_tracking_uri", "http://localhost:5000")
        run_id = config.get("production_run_id")
        if run_id:
            mlflow.set_tracking_uri(tracking_uri)
            model = mlflow.xgboost.load_model(f"runs:/{run_id}/model")
            print(f"[INFO] Loaded XGBoost model from MLflow run {run_id}")
            return model

    raise FileNotFoundError("No model found. Provide model_path or production_run_id in config.")


# ============================================================
# SHAP EXPLAINER
# ============================================================
class AnomalyExplainer:
    """SHAP-based explainer for spending anomalies."""

    def __init__(self, model: xgb.XGBRegressor, background_data: np.ndarray = None):
        """Initialize SHAP explainer.

        Args:
            model: Trained XGBoost model.
            background_data: Sample of training data for SHAP reference.
                If None, uses TreeExplainer (exact, no background needed).
        """
        self.model = model
        self.explainer = shap.TreeExplainer(model)
        print("[INFO] SHAP TreeExplainer initialized")

    def explain_single(self, features: np.ndarray, feature_names: list = None) -> dict:
        """Generate SHAP explanation for a single prediction.

        Args:
            features: 1D array of 20 features.
            feature_names: List of feature names.

        Returns:
            Dict with prediction, shap_values, top_factors.
        """
        if feature_names is None:
            feature_names = FEATURE_COLUMNS

        features_2d = features.reshape(1, -1)

        # Get prediction
        prediction = float(self.model.predict(features_2d)[0])

        # Get SHAP values
        shap_values = self.explainer.shap_values(features_2d)[0]
        base_value = float(self.explainer.expected_value)

        # Top 3 features by absolute SHAP value
        abs_shap = np.abs(shap_values)
        top_indices = np.argsort(abs_shap)[-3:][::-1]

        top_factors = []
        for idx in top_indices:
            fname = feature_names[idx]
            fval = float(features[idx])
            sv = float(shap_values[idx])
            direction = "increased" if sv > 0 else "decreased"
            description = FEATURE_DESCRIPTIONS.get(fname, fname)

            top_factors.append({
                "feature": fname,
                "feature_value": round(fval, 2),
                "shap_value": round(sv, 4),
                "direction": direction,
                "description": description,
                "impact": f"{description} {direction} the predicted amount by ${abs(sv):.2f}",
            })

        return {
            "predicted_spend": round(prediction, 2),
            "base_value": round(base_value, 2),
            "shap_values": {
                feature_names[i]: round(float(shap_values[i]), 4)
                for i in range(len(shap_values))
            },
            "top_factors": top_factors,
        }

    def explain_batch(
        self,
        features_df: pd.DataFrame,
        anomaly_scores: np.ndarray = None,
        threshold: float = 0.6,
    ) -> list:
        """Generate explanations for all anomalies in a batch.

        Args:
            features_df: DataFrame with feature columns + metadata.
            anomaly_scores: Anomaly scores (only explain those > threshold).
            threshold: Minimum anomaly score to explain.

        Returns:
            List of explanation dicts.
        """
        X = features_df[FEATURE_COLUMNS].values.astype(np.float32)

        # Filter to anomalies only
        if anomaly_scores is not None:
            mask = anomaly_scores > threshold
            indices = np.where(mask)[0]
        else:
            indices = range(len(X))

        print(f"[INFO] Generating explanations for {len(indices)} anomalies...")

        # Batch SHAP computation
        all_shap = self.explainer.shap_values(X[indices]) if len(indices) > 0 else np.array([])

        explanations = []
        for i, idx in enumerate(indices):
            shap_vals = all_shap[i]
            abs_shap = np.abs(shap_vals)
            top_3 = np.argsort(abs_shap)[-3:][::-1]

            top_factors = []
            for t in top_3:
                fname = FEATURE_COLUMNS[t]
                sv = float(shap_vals[t])
                fval = float(X[idx, t])
                direction = "increased" if sv > 0 else "decreased"
                desc = FEATURE_DESCRIPTIONS.get(fname, fname)

                top_factors.append({
                    "feature": fname,
                    "feature_value": round(fval, 2),
                    "shap_value": round(sv, 4),
                    "direction": direction,
                    "description": desc,
                })

            # Build metadata
            meta = {}
            for col in ["user_id", "category", "period"]:
                if col in features_df.columns:
                    meta[col] = str(features_df.iloc[idx][col])

            predicted = float(self.model.predict(X[idx:idx+1])[0])
            actual = float(X[idx, FEATURE_COLUMNS.index("current_spend")])
            direction = 1 if actual > predicted else -1
            magnitude_pct = abs(actual - predicted) / max(predicted, 1e-8) * 100

            explanation = {
                **meta,
                "anomaly_score": round(float(anomaly_scores[idx]), 4) if anomaly_scores is not None else None,
                "predicted_spend": round(predicted, 2),
                "actual_spend": round(actual, 2),
                "direction": direction,
                "magnitude_pct": round(magnitude_pct, 1),
                "top_factors": top_factors,
                "natural_language": generate_alert_text(
                    meta.get("category", "Unknown"),
                    actual, predicted, direction, magnitude_pct, top_factors,
                ),
            }
            explanations.append(explanation)

        print(f"[INFO] Generated {len(explanations)} explanations")
        return explanations

    def global_importance(self, X: np.ndarray) -> dict:
        """Compute global feature importance via SHAP."""
        shap_values = self.explainer.shap_values(X)
        mean_abs_shap = np.abs(shap_values).mean(axis=0)

        importance = {}
        for i, fname in enumerate(FEATURE_COLUMNS):
            importance[fname] = round(float(mean_abs_shap[i]), 4)

        # Sort by importance
        importance = dict(sorted(importance.items(), key=lambda x: -x[1]))
        return importance


# ============================================================
# NATURAL LANGUAGE GENERATION
# ============================================================

SEVERITY_THRESHOLDS = {
    "low": (0.6, 0.7),
    "medium": (0.7, 0.85),
    "high": (0.85, 1.0),
}


def get_severity(anomaly_score: float) -> str:
    """Map anomaly score to severity tier."""
    if anomaly_score >= 0.85:
        return "high"
    elif anomaly_score >= 0.7:
        return "medium"
    else:
        return "low"


def generate_alert_text(
    category: str,
    actual: float,
    predicted: float,
    direction: int,
    magnitude_pct: float,
    top_factors: list,
) -> str:
    """Generate human-readable alert text for an anomaly.

    Example output:
      "Your Dining spending this month ($485.30) is 45% higher than expected ($334.69).
       This appears driven by: higher average transaction size ($48.53 vs typical $33.47),
       more transactions (10 vs typical 7), and increased weekend spending."
    """
    direction_word = "higher" if direction > 0 else "lower"

    # Main alert
    text = (
        f"Your {category} spending this month (${actual:,.2f}) is "
        f"{magnitude_pct:.0f}% {direction_word} than expected (${predicted:,.2f})."
    )

    # Explain top factors
    if top_factors:
        factor_texts = []
        for f in top_factors[:3]:
            desc = f.get("description", f["feature"])
            if f["direction"] == "increased":
                factor_texts.append(f"higher {desc}")
            else:
                factor_texts.append(f"lower {desc}")

        if len(factor_texts) == 1:
            factors_str = factor_texts[0]
        elif len(factor_texts) == 2:
            factors_str = f"{factor_texts[0]} and {factor_texts[1]}"
        else:
            factors_str = f"{factor_texts[0]}, {factor_texts[1]}, and {factor_texts[2]}"

        text += f" This appears driven by: {factors_str}."

    return text


# ============================================================
# MAIN
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="SHAP explainability for anomaly detection")
    parser.add_argument("config", help="Path to YAML config file")
    parser.add_argument("--user-id", help="Explain anomalies for specific user")
    parser.add_argument("--all-users", action="store_true", help="Explain all user anomalies")
    parser.add_argument("--output", help="Output path for explanations JSON")
    parser.add_argument("--global-importance", action="store_true",
                        help="Compute global feature importance")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # Load model
    model = load_xgboost_model(config)
    explainer = AnomalyExplainer(model)

    # Load feature data
    features_path = config.get("features_path", "/data/trend_features.parquet")
    features_df = pd.read_parquet(features_path)

    if args.user_id:
        features_df = features_df[features_df["user_id"] == args.user_id]

    if features_df.empty:
        print("[WARN] No feature data found")
        return

    # Global importance
    if args.global_importance:
        X = features_df[FEATURE_COLUMNS].values.astype(np.float32)
        importance = explainer.global_importance(X)
        print(f"\n[GLOBAL IMPORTANCE]")
        for fname, imp in importance.items():
            print(f"  {fname}: {imp:.4f}")
        return

    # Compute anomaly scores (XGBoost residual + Isolation Forest)
    X = features_df[FEATURE_COLUMNS].values.astype(np.float32)
    predictions = model.predict(X)
    actuals = features_df["current_spend"].values

    residuals = np.abs(actuals - predictions)
    residual_scores = (residuals - residuals.min()) / (residuals.max() - residuals.min() + 1e-8)

    # Use residual scores as anomaly scores (Isolation Forest would be added in production)
    anomaly_scores = residual_scores

    # Generate explanations
    threshold = config.get("anomaly_threshold", 0.6)
    explanations = explainer.explain_batch(features_df, anomaly_scores, threshold)

    # Add severity
    for exp in explanations:
        if exp.get("anomaly_score") is not None:
            exp["severity"] = get_severity(exp["anomaly_score"])

    # Output
    output_path = args.output or config.get("output_path", "/data/anomaly_explanations.json")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(explanations, f, indent=2, default=str)

    print(f"\n[DONE] Saved {len(explanations)} explanations to {output_path}")

    # Print summary
    if explanations:
        print(f"\n--- Anomaly Summary ---")
        for exp in explanations[:5]:
            print(f"\n  {exp.get('category', '?')} ({exp.get('period', '?')}):")
            print(f"    {exp.get('natural_language', '')}")
            print(f"    Severity: {exp.get('severity', '?')} | Score: {exp.get('anomaly_score', '?')}")


if __name__ == "__main__":
    main()
