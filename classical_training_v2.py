#!/usr/bin/env python
"""
Classical ML — Training Pipeline (v2)
=======================================

Canonical training script that supersedes classical_training.ipynb.

Key improvements over v1:
- **Temporal split** (train/val/calibration/test) — no future data leakage
- **TimeSeriesSplit** cross-validation instead of StratifiedKFold
- **Train vs val score monitoring** for overfitting detection
- **Learning curves** during initial training
- **MCC** (Matthews Correlation Coefficient) in all evaluations
- **Separate regression logging** (R² never logged as roc_auc)
- **Dynamic scale_pos_weight** (computed from training data)
- **9 models** including LightGBM, CatBoost, SVM, L1/ElasticNet LR

Usage:
    # Full pipeline with all models:
    python classical_training_v2.py

    # Specific models only:
    python classical_training_v2.py --models xgboost lightgbm catboost

    # Smoke test (small subset):
    python classical_training_v2.py --max-rows 5000

    # Skip learning curves (faster):
    python classical_training_v2.py --no-learning-curves
"""

import argparse
import os
import sys

import joblib
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import RocCurveDisplay
from sklearn.model_selection import cross_validate
from tqdm.auto import tqdm

from classical_models import (
    FEATURE_COLUMNS,
    TARGET_RAIN,
    TARGET_TEMP,
    compute_scale_pos_weight,
    evaluate_model,
    evaluate_regressor,
    build_temp_regressor,
    get_all_models,
)
from data_splitting import (
    load_and_split,
    build_temporal_cv,
    print_split_info,
)
from hyperparameter_tuning import plot_learning_curves
from training_logger import (
    clean_all_csvs,
    log_cv_results,
    log_model_save,
    log_regression_metrics,
    log_training_metrics,
    log_train_val_comparison,
    setup_logger,
)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

ALL_MODELS = [
    "logistic_regression",
    "logistic_regression_l1",
    "logistic_regression_elasticnet",
    "decision_tree",
    "random_forest",
    "xgboost",
    "lightgbm",
    "catboost",
    "svm",
]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Classical ML training pipeline (v2 — temporal split).",
    )
    parser.add_argument(
        "--models", "-m", nargs="+", choices=ALL_MODELS, default=ALL_MODELS,
        help="Model(s) to train. Default: all models.",
    )
    parser.add_argument(
        "--cv-folds", type=int, default=5,
        help="Number of TimeSeriesSplit folds (default: 5).",
    )
    parser.add_argument(
        "--no-learning-curves", action="store_true",
        help="Skip learning curve generation.",
    )
    parser.add_argument(
        "--no-temp-regressor", action="store_true",
        help="Skip temperature regressor training.",
    )
    parser.add_argument(
        "--max-rows", type=int, default=None,
        help="Max rows to load (for smoke testing).",
    )
    parser.add_argument(
        "--models-dir", default="saved_models/",
        help="Directory for saved models (default: saved_models/).",
    )
    parser.add_argument(
        "--artifacts-dir", default="artifacts/",
        help="Directory for plots (default: artifacts/).",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv=None):
    cfg = parse_args(argv)

    os.makedirs(cfg.models_dir, exist_ok=True)
    os.makedirs(cfg.artifacts_dir, exist_ok=True)

    # Clean old CSV logs before new run
    clean_all_csvs("logs")

    logger = setup_logger("classical_training_v2", level=cfg.log_level)
    logger.info("=== Classical Training v2 — Session Started ===")

    # ── 1. Data Loading & Temporal Split ─────────────────────────────────────
    print("\n" + "=" * 60)
    print("  Phase 1: Data Loading & Temporal Split")
    print("=" * 60)

    split = load_and_split(max_rows=cfg.max_rows)
    print_split_info(split)

    X_train = split["X_train"]
    y_train = split["y_train"]
    X_val = split["X_val"]
    y_val = split["y_val"]
    X_test = split["X_test"]
    y_test = split["y_test"]

    logger.info(f"Data loaded: {split['split_info']}")

    # Dynamic scale_pos_weight
    spw = compute_scale_pos_weight(y_train)
    print(f"  Dynamic scale_pos_weight: {spw:.4f}")
    logger.info(f"scale_pos_weight (dynamic): {spw:.4f}")

    # ── 2. Model Instantiation ───────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  Phase 2: Model Instantiation")
    print("=" * 60)

    all_models = get_all_models(scale_pos_weight=spw)
    models = {name: all_models[name] for name in cfg.models if name in all_models}

    for name, pipeline in models.items():
        step_name = pipeline.steps[-1][0] if hasattr(pipeline, 'steps') else type(pipeline).__name__
        clf_name = type(pipeline.steps[-1][1]).__name__ if hasattr(pipeline, 'steps') else step_name
        print(f"  ✅ {name}: {clf_name}")
        logger.info(f"Model instantiated: {name} ({clf_name})")

    cv_strategy = build_temporal_cv(n_splits=cfg.cv_folds)

    # ── 3. Cross-Validation with Train vs Val Monitoring ─────────────────────
    print("\n" + "=" * 60)
    print("  Phase 3: Cross-Validation (TimeSeriesSplit)")
    print("=" * 60)

    scoring = ["roc_auc", "f1", "accuracy", "precision", "recall"]
    cv_results = {}

    for name, pipeline in tqdm(models.items(), desc="Cross-Validation", unit="model"):
        print(f"\n  ⏳ CV for {name}...")
        logger.info(f"CV: {name}")

        scores = cross_validate(
            pipeline, X_train, y_train,
            cv=cv_strategy, scoring=scoring,
            n_jobs=-1, return_train_score=True,
            verbose=0,
        )
        cv_results[name] = scores

        # Print val scores
        print(f"    Val ROC-AUC:  {scores['test_roc_auc'].mean():.4f} ± {scores['test_roc_auc'].std():.4f}")
        print(f"    Val F1:       {scores['test_f1'].mean():.4f} ± {scores['test_f1'].std():.4f}")
        print(f"    Val Accuracy: {scores['test_accuracy'].mean():.4f} ± {scores['test_accuracy'].std():.4f}")

        # Print train scores for overfitting check
        train_auc = scores["train_roc_auc"].mean()
        val_auc = scores["test_roc_auc"].mean()
        gap = train_auc - val_auc
        risk = "🔴 HIGH" if gap > 0.05 else "🟡 MODERATE" if gap > 0.02 else "🟢 LOW"
        print(f"    Train ROC-AUC: {train_auc:.4f}")
        print(f"    Overfit gap:   {gap:.4f} → {risk}")

        # Structured logging
        log_cv_results(logger, name, scores)
        log_train_val_comparison(
            logger, name,
            scores["train_roc_auc"], scores["test_roc_auc"],
            metric_name="roc_auc",
        )

    print("\n✅ Cross-validation complete for all models.")

    # ── Summary Table ────────────────────────────────────────────────────────
    print("\n── CV Results Summary (TimeSeriesSplit) ──")
    summary_rows = []
    for name in models:
        s = cv_results[name]
        summary_rows.append({
            "Model": name,
            "Val ROC-AUC": f"{s['test_roc_auc'].mean():.4f} ± {s['test_roc_auc'].std():.4f}",
            "Train ROC-AUC": f"{s['train_roc_auc'].mean():.4f}",
            "Gap": f"{s['train_roc_auc'].mean() - s['test_roc_auc'].mean():.4f}",
            "Val F1": f"{s['test_f1'].mean():.4f}",
        })
    print(pd.DataFrame(summary_rows).set_index("Model").to_markdown())

    # ── 4. Model Training & Evaluation ───────────────────────────────────────
    print("\n" + "=" * 60)
    print("  Phase 4: Full Training & Evaluation")
    print("=" * 60)

    trained_models = {}
    all_test_metrics = {}

    for name, pipeline in tqdm(models.items(), desc="Training models", unit="model"):
        print(f"\n  ⏳ Training {name}...")
        logger.info(f"Training {name}...")
        pipeline.fit(X_train, y_train)
        print(f"    ✅ {name} fitted.")

        # Evaluate on test set
        metrics = evaluate_model(pipeline, X_test, y_test)
        trained_models[name] = pipeline
        all_test_metrics[name] = metrics

        print(f"    Accuracy:  {metrics['accuracy']:.4f}")
        print(f"    ROC-AUC:   {metrics['roc_auc']:.4f}")
        print(f"    F1:        {metrics['f1_score']:.4f}")
        print(f"    Precision: {metrics['precision']:.4f}")
        print(f"    Recall:    {metrics['recall']:.4f}")
        print(f"    MCC:       {metrics['mcc']:.4f}")

        # Structured logging
        log_training_metrics(logger, name, metrics, phase="evaluation")

        # Save model
        model_path = os.path.join(cfg.models_dir, f"{name}.joblib")
        joblib.dump(pipeline, model_path)
        print(f"    Saved → {model_path}")
        log_model_save(logger, name, model_path)

    print("\n✅ All classification models trained and saved!")

    # ── Test Results Summary ─────────────────────────────────────────────────
    print("\n── Test Set Results Summary ──")
    test_rows = []
    for name, m in all_test_metrics.items():
        test_rows.append({
            "Model": name,
            "ROC-AUC": f"{m['roc_auc']:.4f}",
            "F1": f"{m['f1_score']:.4f}",
            "Accuracy": f"{m['accuracy']:.4f}",
            "Precision": f"{m['precision']:.4f}",
            "Recall": f"{m['recall']:.4f}",
            "MCC": f"{m['mcc']:.4f}",
        })
    print(pd.DataFrame(test_rows).set_index("Model").to_markdown())

    # ── 5. ROC Curves ────────────────────────────────────────────────────────
    print("\n  Generating ROC curves...")
    n_models = len(trained_models)
    n_cols = min(4, n_models)
    n_rows = (n_models + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))
    if n_models == 1:
        axes = np.array([axes])
    axes = axes.ravel()

    for ax, (name, pipeline) in zip(axes, trained_models.items()):
        RocCurveDisplay.from_estimator(pipeline, X_test, y_test, ax=ax, name=name)
        ax.set_title(name)
        ax.grid(True, alpha=0.3)

    # Hide unused axes
    for ax in axes[n_models:]:
        ax.set_visible(False)

    plt.tight_layout()
    roc_path = os.path.join(cfg.artifacts_dir, "roc_curves_classical_v2.png")
    plt.savefig(roc_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✅ ROC curves saved → {roc_path}")

    # ── 6. Learning Curves ───────────────────────────────────────────────────
    if not cfg.no_learning_curves:
        print("\n" + "=" * 60)
        print("  Phase 6: Learning Curves (Overfitting Detection)")
        print("=" * 60)

        # Re-instantiate unfitted models for learning curves
        fresh_models = get_all_models(scale_pos_weight=spw)
        lc_models = {n: fresh_models[n] for n in cfg.models if n in fresh_models}

        n_cols_lc = min(3, len(lc_models))
        n_rows_lc = (len(lc_models) + n_cols_lc - 1) // n_cols_lc
        fig, axes = plt.subplots(n_rows_lc, n_cols_lc,
                                 figsize=(6 * n_cols_lc, 5 * n_rows_lc))
        if len(lc_models) == 1:
            axes = np.array([axes])
        axes = axes.ravel()

        for ax, (name, pipeline) in tqdm(
            zip(axes, lc_models.items()), desc="Learning curves", total=len(lc_models)
        ):
            logger.info(f"Learning curve: {name}")
            plot_learning_curves(
                pipeline, X_train, y_train,
                cv=cv_strategy, scoring="roc_auc",
                n_jobs=-1, ax=ax,
            )
            ax.set_title(name)

        for ax in axes[len(lc_models):]:
            ax.set_visible(False)

        plt.tight_layout()
        lc_path = os.path.join(cfg.artifacts_dir, "learning_curves_v2.png")
        plt.savefig(lc_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"\n✅ Learning curves saved → {lc_path}")

    # ── 7. Temperature Regressor ─────────────────────────────────────────────
    if not cfg.no_temp_regressor:
        print("\n" + "=" * 60)
        print("  Phase 7: Temperature Regressor (HistGradientBoosting)")
        print("=" * 60)

        # Load full dataset for temp features
        from data_splitting import load_dataset
        df = load_dataset(max_rows=cfg.max_rows)

        temp_features = [c for c in FEATURE_COLUMNS if c != "MaxTemp"]
        df_temp = df.dropna(subset=["MaxTemp"]).copy()
        df_temp = df_temp.sort_values("Date").reset_index(drop=True)

        # Chronological 80/20 split
        split_idx = int(len(df_temp) * 0.8)
        X_temp_train = df_temp[temp_features].iloc[:split_idx]
        X_temp_test = df_temp[temp_features].iloc[split_idx:]
        y_temp_train = df_temp[TARGET_TEMP].iloc[:split_idx]
        y_temp_test = df_temp[TARGET_TEMP].iloc[split_idx:]

        print(f"  Temp Train: {X_temp_train.shape}, Temp Test: {X_temp_test.shape}")
        logger.info(f"Temp regressor data: train={X_temp_train.shape}, test={X_temp_test.shape}")

        temp_model = build_temp_regressor()
        print("  ⏳ Fitting temperature regressor (HistGradientBoosting)...")
        temp_model.fit(X_temp_train, y_temp_train)
        print("  ✅ Temperature regressor fitted.")

        temp_metrics = evaluate_regressor(temp_model, X_temp_test, y_temp_test)
        print(f"  MAE  : {temp_metrics['mae']:.4f}")
        print(f"  RMSE : {temp_metrics['rmse']:.4f}")
        print(f"  R²   : {temp_metrics['r2']:.4f}")

        # Use separate regression logger (NOT classification logger!)
        log_regression_metrics(logger, "temp_regressor", temp_metrics, phase="evaluation")

        temp_path = os.path.join(cfg.models_dir, "temp_regressor.joblib")
        joblib.dump(temp_model, temp_path)
        print(f"  Saved → {temp_path}")
        log_model_save(logger, "temp_regressor", temp_path)

    # ── 8. Identify Best Model ───────────────────────────────────────────────
    if all_test_metrics:
        best_name = max(all_test_metrics, key=lambda n: all_test_metrics[n]["roc_auc"])
        best_score = all_test_metrics[best_name]["roc_auc"]
        print(f"\n{'='*60}")
        print(f"  🏆 Best classifier: {best_name} (ROC-AUC = {best_score:.4f})")
        print(f"{'='*60}")

        best_path = os.path.join(cfg.models_dir, "best_classifier.joblib")
        joblib.dump(trained_models[best_name], best_path)
        print(f"  Saved best model → {best_path}")
        log_model_save(logger, f"best_classifier ({best_name})", best_path)

    logger.info("=== Classical Training v2 — Session Complete ===")
    print("\n🎉 Training session complete.")


if __name__ == "__main__":
    main()
