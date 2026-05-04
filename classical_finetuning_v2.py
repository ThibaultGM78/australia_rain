#!/usr/bin/env python
"""
Classical ML — Hyperparameter Tuning & Calibration (v2)
========================================================

Canonical finetuning script that supersedes classical_finetuning.ipynb.

Key improvements over v1:
- **TimeSeriesSplit** CV (no future data leakage)
- **Dedicated calibration set** (never calibrate on test set)
- **RandomSearch n_iter=50** (up from 10)
- **Optuna with MedianPruner** for early stopping
- **Saves best model globally** across all methods and models
- **Train-vs-val gap analysis** for every tuning result
- **Support for 9 models** including LightGBM, CatBoost, SVM

Usage:
    # Full pipeline:
    python classical_finetuning_v2.py

    # Specific models and methods:
    python classical_finetuning_v2.py --model xgboost lightgbm --method optuna

    # Quick Optuna-only with 30 trials:
    python classical_finetuning_v2.py --method optuna --n-trials 30

    # Skip calibration:
    python classical_finetuning_v2.py --no-calibrate
"""

import argparse
import os
import sys

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from classical_models import (
    FEATURE_COLUMNS,
    TARGET_RAIN,
    compute_scale_pos_weight,
    evaluate_model,
    get_all_models,
)
from data_splitting import (
    load_and_split,
    build_temporal_cv,
    print_split_info,
)
from hyperparameter_tuning import (
    calibrate_model,
    compare_calibration,
    extract_train_val_gap,
    get_param_distributions,
    get_param_grids,
    plot_calibration_curves,
    plot_learning_curves,
    run_grid_search,
    run_optuna_study,
    run_random_search,
)
from training_logger import (
    log_calibration_results,
    log_model_save,
    log_training_metrics,
    log_tuning_results,
    setup_logger,
)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

VALID_MODELS = [
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
VALID_METHODS = ["grid_search", "random_search", "optuna"]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Hyperparameter tuning & calibration (v2 — temporal split).",
    )
    parser.add_argument(
        "--model", "-m", dest="models", nargs="+",
        choices=VALID_MODELS, default=VALID_MODELS, metavar="MODEL",
        help=f"Model(s) to tune. Default: all. Choices: {', '.join(VALID_MODELS)}",
    )
    parser.add_argument(
        "--method", dest="methods", nargs="+",
        choices=VALID_METHODS, default=VALID_METHODS, metavar="METHOD",
        help=f"Tuning method(s). Default: all. Choices: {', '.join(VALID_METHODS)}",
    )
    parser.add_argument("--n-iter", type=int, default=50,
                        help="RandomizedSearchCV iterations (default: 50).")
    parser.add_argument("--n-trials", type=int, default=50,
                        help="Optuna trials (default: 50).")
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--scoring", default="roc_auc")
    parser.add_argument("--no-calibrate", action="store_true")
    parser.add_argument("--calibration-method", choices=["isotonic", "sigmoid"],
                        default="isotonic")
    parser.add_argument("--no-learning-curves", action="store_true")
    parser.add_argument("--models-dir", default="saved_models/")
    parser.add_argument("--artifacts-dir", default="artifacts/")
    parser.add_argument("--no-save", action="store_true")
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--n-jobs", type=int, default=-1)
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

    logger = setup_logger("classical_finetuning_v2", level=cfg.log_level)
    logger.info("=== Fine-Tuning v2 — Session Started ===")
    logger.info(f"Models   : {cfg.models}")
    logger.info(f"Methods  : {cfg.methods}")

    print(f"\n{'='*60}")
    print(f"  Fine-Tuning v2 (Temporal Split)")
    print(f"  Models  : {', '.join(cfg.models)}")
    print(f"  Methods : {', '.join(cfg.methods)}")
    print(f"  CV folds: {cfg.cv_folds} (TimeSeriesSplit)")
    print(f"{'='*60}\n")

    # ── Data ─────────────────────────────────────────────────────────────────
    split = load_and_split(max_rows=cfg.max_rows)
    print_split_info(split)

    X_train = split["X_train"]
    y_train = split["y_train"]
    X_val = split["X_val"]
    y_val = split["y_val"]
    X_cal = split["X_cal"]
    y_cal = split["y_cal"]
    X_test = split["X_test"]
    y_test = split["y_test"]

    logger.info(f"Data loaded: {split['split_info']}")

    spw = compute_scale_pos_weight(y_train)
    print(f"  Dynamic scale_pos_weight: {spw:.4f}")

    # ── Models ───────────────────────────────────────────────────────────────
    all_models = get_all_models(scale_pos_weight=spw)
    models = {name: all_models[name] for name in cfg.models if name in all_models}

    cv_strategy = build_temporal_cv(n_splits=cfg.cv_folds)

    # ── Global best tracker ──────────────────────────────────────────────────
    global_best = {"model": None, "score": -1.0, "method": None, "name": None}

    def update_global_best(name, method, score, model_obj):
        if score > global_best["score"]:
            global_best["model"] = model_obj
            global_best["score"] = score
            global_best["method"] = method
            global_best["name"] = name

    # ── Learning curves ──────────────────────────────────────────────────────
    if not cfg.no_learning_curves:
        print("── Learning Curves (TimeSeriesSplit) ────────────────────")
        n_cols = min(3, len(models))
        n_rows = (len(models) + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols,
                                 figsize=(6 * n_cols, 5 * n_rows), squeeze=False)
        axes = axes.ravel()

        for ax, (name, pipeline) in tqdm(
            zip(axes, models.items()), desc="Learning curves", total=len(models)
        ):
            logger.info(f"Learning curve: {name}")
            plot_learning_curves(
                pipeline, X_train, y_train,
                cv=cv_strategy, scoring=cfg.scoring,
                n_jobs=cfg.n_jobs, ax=ax,
            )
            ax.set_title(name)

        for ax in axes[len(models):]:
            ax.set_visible(False)

        plt.tight_layout()
        out_path = os.path.join(cfg.artifacts_dir, "learning_curves_finetuning_v2.png")
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"✅ Learning curves saved → {out_path}\n")

    # ── Tuning ───────────────────────────────────────────────────────────────
    grid_results = {}
    random_results = {}
    optuna_results = {}

    # GridSearch ──────────────────────────────────────────────────────────────
    if "grid_search" in cfg.methods:
        print("── GridSearchCV (TimeSeriesSplit) ────────────────────────")
        param_grids = get_param_grids()
        for name in tqdm(cfg.models, desc="GridSearchCV", unit="model"):
            if name not in param_grids:
                print(f"  ⚠️  No grid defined for {name}, skipping.")
                continue
            # Fresh unfitted model
            fresh = get_all_models(scale_pos_weight=spw)
            if name not in fresh:
                continue
            print(f"\n  Model: {name}")
            logger.info(f"GridSearch: {name}")
            gs = run_grid_search(
                fresh[name], param_grids[name],
                X_train, y_train,
                cv=cv_strategy, scoring=cfg.scoring, n_jobs=cfg.n_jobs,
            )
            grid_results[name] = gs

            # Train vs Val gap
            gap_info = extract_train_val_gap(gs)
            print(f"  Best {cfg.scoring}: {gs.best_score_:.4f}")
            print(f"  Train-Val gap: {gap_info['gap']:.4f} ({gap_info['overfitting_risk']})")
            print(f"  Best params : {gs.best_params_}")

            log_tuning_results(logger, name, "grid_search", gs.best_params_, gs.best_score_)
            update_global_best(name, "grid_search", gs.best_score_, gs.best_estimator_)
        print("✅ GridSearchCV complete.\n")

    # RandomizedSearch ────────────────────────────────────────────────────────
    if "random_search" in cfg.methods:
        print("── RandomizedSearchCV (TimeSeriesSplit) ──────────────────")
        param_distributions = get_param_distributions()
        for name in tqdm(cfg.models, desc="RandomizedSearchCV", unit="model"):
            if name not in param_distributions:
                print(f"  ⚠️  No distribution defined for {name}, skipping.")
                continue
            fresh = get_all_models(scale_pos_weight=spw)
            if name not in fresh:
                continue
            print(f"\n  Model: {name}")
            logger.info(f"RandomSearch: {name}")
            rs = run_random_search(
                fresh[name], param_distributions[name],
                X_train, y_train,
                n_iter=cfg.n_iter, cv=cv_strategy,
                scoring=cfg.scoring, n_jobs=cfg.n_jobs,
            )
            random_results[name] = rs

            gap_info = extract_train_val_gap(rs)
            print(f"  Best {cfg.scoring}: {rs.best_score_:.4f}")
            print(f"  Train-Val gap: {gap_info['gap']:.4f} ({gap_info['overfitting_risk']})")
            print(f"  Best params : {rs.best_params_}")

            log_tuning_results(logger, name, "random_search", rs.best_params_, rs.best_score_)
            update_global_best(name, "random_search", rs.best_score_, rs.best_estimator_)
        print("✅ RandomizedSearchCV complete.\n")

    # Optuna ──────────────────────────────────────────────────────────────────
    if "optuna" in cfg.methods:
        print("── Optuna Bayesian Optimisation (MedianPruner) ───────────")
        for name in tqdm(cfg.models, desc="Optuna", unit="model"):
            print(f"\n  Model: {name}  ({cfg.n_trials} trials)")
            logger.info(f"Optuna: {name}")
            study = run_optuna_study(
                name, X_train, y_train,
                n_trials=cfg.n_trials, cv=cv_strategy,
                scale_pos_weight=spw,
            )
            optuna_results[name] = study
            print(f"  Best {cfg.scoring}: {study.best_value:.4f}")
            print(f"  Best params : {study.best_params}")
            log_tuning_results(logger, name, "optuna", study.best_params, study.best_value)
            update_global_best(name, "optuna", study.best_value, None)
        print("✅ Optuna complete.\n")

    # ── Comparison table ─────────────────────────────────────────────────────
    print("── Tuning Method Comparison ──────────────────────────────")
    rows = []
    for name in cfg.models:
        row = {"Model": name}
        if name in grid_results:
            row["GridSearch"] = f"{grid_results[name].best_score_:.4f}"
        if name in random_results:
            row["RandomSearch"] = f"{random_results[name].best_score_:.4f}"
        if name in optuna_results:
            row["Optuna"] = f"{optuna_results[name].best_value:.4f}"
        rows.append(row)
    comp_df = pd.DataFrame(rows).set_index("Model")
    print(comp_df.to_markdown())
    print()

    # ── Probability calibration (on dedicated calibration set!) ──────────────
    calibrated_models = {}
    if not cfg.no_calibrate:
        print("── Probability Calibration (dedicated cal set) ──────────")
        print(f"  ⚠️  Using dedicated calibration set ({len(X_cal)} rows)")
        print(f"       NOT the test set — this is the correct approach.\n")

        # Find the best fitted model per name
        best_fitted = {}
        for name in cfg.models:
            if name in grid_results:
                best_fitted[name] = grid_results[name].best_estimator_
            elif name in random_results:
                best_fitted[name] = random_results[name].best_estimator_

        for name in tqdm(list(best_fitted.keys()), desc="Calibrating", unit="model"):
            logger.info(f"Calibrating: {name}")
            best_model = best_fitted[name]
            cal_model = calibrate_model(
                best_model, X_cal, y_cal, method=cfg.calibration_method
            )
            calibrated_models[name] = cal_model

            # Evaluate on test set (compare only, never fit!)
            stats = compare_calibration(best_model, cal_model, X_test, y_test)
            print(
                f"  {name}\n"
                f"    Raw        → ROC-AUC: {stats['raw']['roc_auc']:.4f}  "
                f"Brier: {stats['raw']['brier_score']:.4f}  "
                f"MCC: {stats['raw']['mcc']:.4f}\n"
                f"    Calibrated → ROC-AUC: {stats['calibrated']['roc_auc']:.4f}  "
                f"Brier: {stats['calibrated']['brier_score']:.4f}  "
                f"MCC: {stats['calibrated']['mcc']:.4f}"
            )
            log_calibration_results(logger, name, stats)

        # Calibration curves
        if calibrated_models:
            cal_plot_models = {}
            for name in calibrated_models:
                if name in best_fitted:
                    cal_plot_models[f"{name} (raw)"] = best_fitted[name]
                cal_plot_models[f"{name} (cal)"] = calibrated_models[name]

            fig, ax = plt.subplots(figsize=(10, 8))
            plot_calibration_curves(cal_plot_models, X_test, y_test, ax=ax)
            plt.tight_layout()
            out_path = os.path.join(cfg.artifacts_dir, "calibration_curves_v2.png")
            plt.savefig(out_path, dpi=150, bbox_inches="tight")
            plt.close()
            print(f"\n✅ Calibration curves saved → {out_path}")

    # ── Save best models ─────────────────────────────────────────────────────
    if not cfg.no_save:
        print("\n── Saving Tuned Models ──────────────────────────────────")

        # Save per-model best
        for name in cfg.models:
            best_model = None
            if name in grid_results:
                best_model = grid_results[name].best_estimator_
            elif name in random_results:
                best_model = random_results[name].best_estimator_

            if best_model is not None:
                model_path = os.path.join(cfg.models_dir, f"{name}_tuned.joblib")
                joblib.dump(best_model, model_path)
                print(f"  Saved {name} → {model_path}")
                log_model_save(logger, f"{name}_tuned", model_path)

        # Save calibrated models
        for name, cal_model in calibrated_models.items():
            cal_path = os.path.join(cfg.models_dir, f"{name}_calibrated.joblib")
            joblib.dump(cal_model, cal_path)
            print(f"  Saved {name} (calibrated) → {cal_path}")
            log_model_save(logger, f"{name}_calibrated", cal_path)

        # Save global best
        if global_best["model"] is not None:
            best_path = os.path.join(cfg.models_dir, "best_tuned_overall.joblib")
            joblib.dump(global_best["model"], best_path)
            print(f"\n  🏆 Best overall: {global_best['name']} ({global_best['method']}) "
                  f"→ {cfg.scoring}={global_best['score']:.4f}")
            print(f"     Saved → {best_path}")
            log_model_save(logger, f"best_overall ({global_best['name']})", best_path)

        print("✅ All tuned models saved.\n")

    logger.info("=== Fine-Tuning v2 — Session Complete ===")
    print("🎉 Fine-tuning session complete.")


if __name__ == "__main__":
    main()
