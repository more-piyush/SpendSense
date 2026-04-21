"""
train_trend_detection.py — Training script for the XGBoost spending trend detection model.

Supports multiple model types via config:
  - "random_forest": Random Forest baseline
  - "xgboost": XGBoost Gradient Boosted Regression
  - "xgboost_optuna": XGBoost with Optuna hyperparameter tuning

Usage:
  python train_trend_detection.py configs/trend_rf_baseline.yaml
  python train_trend_detection.py configs/trend_xgb_v1.yaml
  python train_trend_detection.py configs/trend_xgb_optuna.yaml

All hyperparameters come from the YAML config file.
All runs are logged to MLflow.
"""

import sys
import os
import json
import time
import warnings

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.xgboost
import mlflow.sklearn

from utils import (
    load_config,
    setup_mlflow,
    log_environment_info,
    compute_data_hash,
    load_parquet,
    TrainingTimer,
    log_peak_memory,
    register_candidate_model,
    tag_primary_model_artifact,
)

warnings.filterwarnings("ignore")

# The 20 input features for trend detection
FEATURE_COLUMNS = [
    "current_spend", "rolling_mean_1m", "rolling_mean_3m", "rolling_mean_6m",
    "rolling_std_3m", "rolling_std_6m", "deviation_ratio", "share_of_wallet",
    "hist_share_of_wallet", "txn_count", "hist_txn_count_mean", "avg_txn_size",
    "hist_avg_txn_size", "days_since_last_txn", "month_of_year",
    "spending_velocity", "weekend_txn_ratio", "total_monthly_spend",
    "elevated_cat_count", "budget_utilization",
]

TARGET_COLUMN = "next_month_spend"


# ============================================================
# DATA LOADING
# ============================================================
def load_data(config):
    """Load trend detection data with chronological splitting."""
    data_path = config["data_path"]
    print(f"[INFO] Loading data from {data_path}")

    if data_path.endswith(".parquet"):
        df = load_parquet(data_path, config)
    elif data_path.endswith(".csv"):
        if data_path.startswith("s3://"):
            raise ValueError("CSV read from s3:// not supported; use parquet")
        df = pd.read_csv(data_path)
    else:
        raise ValueError(f"Unsupported format: {data_path}")

    data_hash = compute_data_hash(data_path, config)
    mlflow.log_param("data_hash", data_hash)
    mlflow.log_param("data_rows", len(df))

    # Verify required columns exist
    missing = [c for c in FEATURE_COLUMNS + [TARGET_COLUMN] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in data: {missing}")

    X = df[FEATURE_COLUMNS].values.astype(np.float32)
    y = df[TARGET_COLUMN].values.astype(np.float32)

    # Chronological split (critical: no random splitting for time-series)
    if "split" in df.columns:
        train_mask = df["split"] == "train"
        val_mask = df["split"] == "val"
        test_mask = df["split"] == "test"
    elif "period" in df.columns:
        # Sort by period and split chronologically
        df = df.sort_values("period").reset_index(drop=True)
        n = len(df)
        train_end = int(n * 0.7)
        val_end = int(n * 0.85)
        train_mask = pd.Series([True] * train_end + [False] * (n - train_end))
        val_mask = pd.Series(
            [False] * train_end + [True] * (val_end - train_end) +
            [False] * (n - val_end)
        )
        test_mask = pd.Series([False] * val_end + [True] * (n - val_end))
    else:
        # Fallback: 70/15/15 chronological
        n = len(df)
        train_end = int(n * 0.7)
        val_end = int(n * 0.85)
        train_mask = np.arange(n) < train_end
        val_mask = (np.arange(n) >= train_end) & (np.arange(n) < val_end)
        test_mask = np.arange(n) >= val_end

    X_train, y_train = X[train_mask], y[train_mask]
    X_val, y_val = X[val_mask], y[val_mask]
    X_test, y_test = X[test_mask], y[test_mask]

    mlflow.log_params({
        "train_size": len(y_train),
        "val_size": len(y_val),
        "test_size": len(y_test),
        "n_features": len(FEATURE_COLUMNS),
    })
    print(f"[INFO] Split: train={len(y_train)}, val={len(y_val)}, test={len(y_test)}")

    return X_train, y_train, X_val, y_val, X_test, y_test, df


# ============================================================
# BASELINE: Random Forest
# ============================================================
def train_random_forest(config, X_train, y_train, X_val, y_val, X_test, y_test):
    """Train Random Forest baseline."""
    print("\n" + "=" * 60)
    print("TRAINING BASELINE: Random Forest Regressor")
    print("=" * 60)

    with TrainingTimer("rf_training"):
        model = RandomForestRegressor(
            n_estimators=config.get("n_estimators", 100),
            max_depth=config.get("max_depth", 10),
            min_samples_split=config.get("min_samples_split", 5),
            min_samples_leaf=config.get("min_samples_leaf", 2),
            n_jobs=-1,
            random_state=42,
        )
        model.fit(X_train, y_train)

    metrics = evaluate_regression(model, X_val, y_val, X_test, y_test, "rf")

    # Feature importance
    importances = dict(zip(FEATURE_COLUMNS, model.feature_importances_))
    top_features = sorted(importances.items(), key=lambda x: -x[1])[:5]
    mlflow.log_param("top_5_features",
                     json.dumps([f[0] for f in top_features]))
    print(f"[INFO] Top 5 features: {[f'{k}: {v:.4f}' for k, v in top_features]}")

    model_info = mlflow.sklearn.log_model(model, "model")
    tag_primary_model_artifact(model_info)

    # --- Rich Artifacts ---
    # Feature importance bar chart
    fig_feat, ax_feat = plt.subplots(figsize=(10, 8))
    sorted_imp = sorted(importances.items(), key=lambda x: x[1])
    ax_feat.barh([x[0] for x in sorted_imp], [x[1] for x in sorted_imp], color="forestgreen")
    ax_feat.set_xlabel("Feature Importance (Gini)")
    ax_feat.set_title("Random Forest Feature Importance")
    fig_feat.tight_layout()
    mlflow.log_figure(fig_feat, "feature_importance.png")
    plt.close(fig_feat)

    # Predicted vs Actual scatter
    _log_pred_vs_actual(model, X_test, y_test, "rf")

    return metrics


# ============================================================
# XGBOOST TRAINING
# ============================================================
def train_xgboost(config, X_train, y_train, X_val, y_val, X_test, y_test):
    """Train XGBoost model with given hyperparameters."""
    print("\n" + "=" * 60)
    print("TRAINING: XGBoost Gradient Boosted Regression")
    print("=" * 60)

    params = {
        "objective": config.get("objective", "reg:squarederror"),
        "n_estimators": config.get("n_estimators", 300),
        "max_depth": config.get("max_depth", 6),
        "learning_rate": config.get("learning_rate", 0.05),
        "min_child_weight": config.get("min_child_weight", 5),
        "subsample": config.get("subsample", 0.8),
        "colsample_bytree": config.get("colsample_bytree", 0.8),
        "reg_alpha": config.get("reg_alpha", 0.1),
        "reg_lambda": config.get("reg_lambda", 1.0),
        "gamma": config.get("gamma", 0.1),
        "random_state": 42,
        "n_jobs": -1,
    }

    with TrainingTimer("xgb_training"):
        model = xgb.XGBRegressor(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=config.get("verbose", 10),
        )

    # Get best iteration
    best_iter = model.best_iteration if hasattr(model, "best_iteration") else params["n_estimators"]
    mlflow.log_metric("best_iteration", best_iter)

    metrics = evaluate_regression(model, X_val, y_val, X_test, y_test, "xgb")

    # Feature importance (gain-based)
    importances = model.get_booster().get_score(importance_type="gain")
    feature_imp = {
        FEATURE_COLUMNS[int(k.replace("f", ""))]: v
        for k, v in importances.items()
        if k.startswith("f") and int(k.replace("f", "")) < len(FEATURE_COLUMNS)
    }
    top_features = sorted(feature_imp.items(), key=lambda x: -x[1])[:5]
    mlflow.log_param("top_5_features",
                     json.dumps([f[0] for f in top_features]))
    print(f"[INFO] Top 5 features: {[f'{k}: {v:.1f}' for k, v in top_features]}")

    model_info = mlflow.xgboost.log_model(model, "model")
    tag_primary_model_artifact(model_info)

    # --- Rich Artifacts ---
    # Feature importance bar chart (gain-based)
    fig_feat, ax_feat = plt.subplots(figsize=(10, 8))
    sorted_imp = sorted(feature_imp.items(), key=lambda x: x[1])
    ax_feat.barh([x[0] for x in sorted_imp], [x[1] for x in sorted_imp], color="darkorange")
    ax_feat.set_xlabel("Feature Importance (Gain)")
    ax_feat.set_title("XGBoost Feature Importance")
    fig_feat.tight_layout()
    mlflow.log_figure(fig_feat, "feature_importance.png")
    plt.close(fig_feat)

    # Predicted vs Actual scatter
    _log_pred_vs_actual(model, X_test, y_test, "xgb")

    # Also train Isolation Forest ensemble member
    if config.get("train_isolation_forest", True):
        train_isolation_forest(config, X_train, X_test, y_test, model)

    return metrics


def train_isolation_forest(config, X_train, X_test, y_test, xgb_model):
    """Train Isolation Forest as ensemble member alongside XGBoost."""
    print("\n[INFO] Training Isolation Forest ensemble member...")

    with TrainingTimer("isolation_forest_training"):
        iso_model = IsolationForest(
            n_estimators=config.get("iso_n_estimators", 100),
            contamination=config.get("iso_contamination", 0.1),
            max_samples=config.get("iso_max_samples", 256),
            random_state=42,
            n_jobs=-1,
        )
        iso_model.fit(X_train)

    # Score on test set
    iso_scores = -iso_model.score_samples(X_test)  # Higher = more anomalous
    iso_scores_norm = (iso_scores - iso_scores.min()) / (
        iso_scores.max() - iso_scores.min() + 1e-8
    )

    # XGBoost residual-based scores
    xgb_preds = xgb_model.predict(X_test)
    residuals = np.abs(y_test - xgb_preds)
    residual_scores = (residuals - residuals.min()) / (
        residuals.max() - residuals.min() + 1e-8
    )

    # Ensemble: weighted combination
    xgb_weight = config.get("xgb_ensemble_weight", 0.6)
    iso_weight = 1 - xgb_weight
    ensemble_scores = xgb_weight * residual_scores + iso_weight * iso_scores_norm

    mlflow.log_params({
        "xgb_ensemble_weight": xgb_weight,
        "iso_ensemble_weight": iso_weight,
        "iso_n_estimators": config.get("iso_n_estimators", 100),
        "iso_contamination": config.get("iso_contamination", 0.1),
    })
    mlflow.log_metric("ensemble_mean_score", round(float(ensemble_scores.mean()), 4))
    mlflow.sklearn.log_model(iso_model, "isolation_forest_model")
    print(f"[INFO] Isolation Forest trained. Ensemble mean score: {ensemble_scores.mean():.4f}")


# ============================================================
# XGBOOST WITH OPTUNA TUNING
# ============================================================
def train_xgboost_optuna(config, X_train, y_train, X_val, y_val, X_test, y_test):
    """Train XGBoost with Optuna Bayesian hyperparameter optimization."""
    import optuna

    print("\n" + "=" * 60)
    print("TRAINING: XGBoost with Optuna Hyperparameter Tuning")
    print("=" * 60)

    n_trials = config.get("optuna_n_trials", 50)
    print(f"[INFO] Running {n_trials} Optuna trials...")

    def objective(trial):
        params = {
            "objective": "reg:squarederror",
            "n_estimators": trial.suggest_int("n_estimators", 50, 500),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 2.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 5.0),
            "gamma": trial.suggest_float("gamma", 0.0, 1.0),
            "random_state": 42,
            "n_jobs": -1,
        }

        model = xgb.XGBRegressor(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=0,
        )

        val_preds = model.predict(X_val)
        rmse = np.sqrt(mean_squared_error(y_val, val_preds))
        return rmse

    with TrainingTimer("optuna_search"):
        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    # Log Optuna results
    best_params = study.best_params
    mlflow.log_params({f"best_{k}": v for k, v in best_params.items()})
    mlflow.log_metric("optuna_best_rmse", study.best_value)
    mlflow.log_metric("optuna_n_trials", n_trials)
    print(f"[INFO] Best Optuna RMSE: {study.best_value:.4f}")
    print(f"[INFO] Best params: {json.dumps(best_params, indent=2)}")

    # Train final model with best params
    best_params["objective"] = "reg:squarederror"
    best_params["random_state"] = 42
    best_params["n_jobs"] = -1

    with TrainingTimer("xgb_final_training"):
        best_model = xgb.XGBRegressor(**best_params)
        best_model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=10,
        )

    metrics = evaluate_regression(best_model, X_val, y_val, X_test, y_test, "xgb_optuna")
    model_info = mlflow.xgboost.log_model(best_model, "model")
    tag_primary_model_artifact(model_info)

    # --- Rich Artifacts ---
    # Optuna optimization history
    fig_optuna, ax_optuna = plt.subplots(figsize=(10, 6))
    trials = study.trials
    trial_nums = [t.number for t in trials]
    trial_vals = [t.value for t in trials]
    ax_optuna.plot(trial_nums, trial_vals, "o-", alpha=0.5, markersize=4, color="purple")
    ax_optuna.axhline(study.best_value, color="red", linestyle="--", label=f"Best RMSE: {study.best_value:.4f}")
    ax_optuna.set_xlabel("Trial")
    ax_optuna.set_ylabel("RMSE")
    ax_optuna.set_title("Optuna Optimization History")
    ax_optuna.legend()
    fig_optuna.tight_layout()
    mlflow.log_figure(fig_optuna, "optuna_history.png")
    plt.close(fig_optuna)

    # Predicted vs Actual scatter
    _log_pred_vs_actual(best_model, X_test, y_test, "xgb_optuna")

    # Best params as artifact
    mlflow.log_text(json.dumps(best_params, indent=2, default=str), "best_hyperparameters.json")

    # Train Isolation Forest with best model
    if config.get("train_isolation_forest", True):
        train_isolation_forest(config, X_train, X_test, y_test, best_model)

    return metrics


# ============================================================
# ARTIFACT HELPERS
# ============================================================
def _log_pred_vs_actual(model, X_test, y_test, prefix):
    """Log predicted vs actual scatter plot."""
    test_preds = model.predict(X_test)
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(y_test, test_preds, alpha=0.3, s=10, color="steelblue")
    min_val = min(y_test.min(), test_preds.min())
    max_val = max(y_test.max(), test_preds.max())
    ax.plot([min_val, max_val], [min_val, max_val], "r--", lw=2, label="Perfect prediction")
    ax.set_xlabel("Actual Spend")
    ax.set_ylabel("Predicted Spend")
    ax.set_title(f"{prefix.upper()} — Predicted vs Actual (Test Set)")
    ax.legend()
    fig.tight_layout()
    mlflow.log_figure(fig, "predicted_vs_actual.png")
    plt.close(fig)

    # Residual distribution
    residuals = y_test - test_preds
    fig_res, ax_res = plt.subplots(figsize=(10, 6))
    ax_res.hist(residuals, bins=50, color="salmon", edgecolor="black", alpha=0.7)
    ax_res.axvline(0, color="black", linestyle="--", lw=1.5)
    ax_res.set_xlabel("Residual (Actual - Predicted)")
    ax_res.set_ylabel("Count")
    ax_res.set_title(f"{prefix.upper()} — Residual Distribution (Test Set)")
    fig_res.tight_layout()
    mlflow.log_figure(fig_res, "residual_distribution.png")
    plt.close(fig_res)

    # Sample predictions CSV
    sample_size = min(50, len(y_test))
    idx = np.random.RandomState(42).choice(len(y_test), sample_size, replace=False)
    sample_df = pd.DataFrame({
        "actual": y_test[idx],
        "predicted": np.round(test_preds[idx], 2),
        "residual": np.round(y_test[idx] - test_preds[idx], 2),
        "abs_error_pct": np.round(np.abs(y_test[idx] - test_preds[idx]) / (np.abs(y_test[idx]) + 1e-8) * 100, 1),
    })
    mlflow.log_text(sample_df.to_csv(index=False), "sample_predictions.csv")

    # Data stats
    data_stats = {
        "test_size": len(y_test),
        "actual_mean": float(round(y_test.mean(), 2)),
        "actual_std": float(round(y_test.std(), 2)),
        "pred_mean": float(round(test_preds.mean(), 2)),
        "pred_std": float(round(test_preds.std(), 2)),
        "residual_mean": float(round(residuals.mean(), 2)),
        "residual_std": float(round(residuals.std(), 2)),
    }
    mlflow.log_text(json.dumps(data_stats, indent=2), "data_stats.json")


# ============================================================
# EVALUATION
# ============================================================
def evaluate_regression(model, X_val, y_val, X_test, y_test, prefix):
    """Evaluate regression model and log metrics."""
    val_preds = model.predict(X_val)
    test_preds = model.predict(X_test)

    metrics = {
        f"{prefix}_val_rmse": float(round(np.sqrt(mean_squared_error(y_val, val_preds)), 4)),
        f"{prefix}_val_mae": float(round(mean_absolute_error(y_val, val_preds), 4)),
        f"{prefix}_val_r2": float(round(r2_score(y_val, val_preds), 4)),
        f"{prefix}_test_rmse": float(round(np.sqrt(mean_squared_error(y_test, test_preds)), 4)),
        f"{prefix}_test_mae": float(round(mean_absolute_error(y_test, test_preds), 4)),
        f"{prefix}_test_r2": float(round(r2_score(y_test, test_preds), 4)),
    }

    # Anomaly detection quality (simulated)
    # Precision@5: of top-5 largest residuals, how many are true anomalies?
    test_residuals = np.abs(y_test - test_preds)
    top5_indices = np.argsort(test_residuals)[-5:]
    # Using a simple heuristic: true anomaly if residual > 2 * std
    threshold = y_test.std() * 2
    true_anomalies = test_residuals > threshold
    precision_at_5 = true_anomalies[top5_indices].mean()
    metrics[f"{prefix}_precision_at_5"] = float(round(precision_at_5, 4))

    mlflow.log_metrics(metrics)
    log_peak_memory()

    print(f"\n[RESULTS] {prefix} metrics: {json.dumps(metrics, indent=2)}")
    return metrics


# ============================================================
# MAIN
# ============================================================
def main():
    if len(sys.argv) < 2:
        print("Usage: python train_trend_detection.py <config.yaml>")
        sys.exit(1)

    config_path = sys.argv[1]
    config = load_config(config_path)
    config["_config_path"] = config_path

    setup_mlflow(config)

    with mlflow.start_run(run_name=config.get("run_name", os.path.basename(config_path))) as run:
        # Log all config params
        for k, v in config.items():
            if not k.startswith("_"):
                mlflow.log_param(k, v)

        mlflow.set_tags({
            "task_type": config.get("task_type", "trend"),
            "model_id": config.get("model_id", config.get("run_name", "unknown_model")),
            "model_family": config.get("model_family", config.get("model_type", "unknown")),
            "training_mode": config.get("training_mode", "initial"),
        })

        log_environment_info()

        X_train, y_train, X_val, y_val, X_test, y_test, df = load_data(config)

        model_type = config.get("model_type", "xgboost")

        if model_type == "random_forest":
            metrics = train_random_forest(
                config, X_train, y_train, X_val, y_val, X_test, y_test
            )
        elif model_type == "xgboost":
            metrics = train_xgboost(
                config, X_train, y_train, X_val, y_val, X_test, y_test
            )
        elif model_type == "xgboost_optuna":
            metrics = train_xgboost_optuna(
                config, X_train, y_train, X_val, y_val, X_test, y_test
            )
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

        mlflow.log_artifact(config_path)
        register_candidate_model(config, run.info.run_id, metrics)
        print("\n[DONE] Training complete. Check MLflow for full results.")


if __name__ == "__main__":
    main()
