#!/usr/bin/env python
# coding: utf-8

"""
Classical ML — Hyperparameter Tuning & Calibration
====================================================

Run with arguments to control which models and tuning methods to use.

Examples
--------
# Tune all models with all methods (default behaviour):
    python classical_finetuning.py

# Tune a single model with a specific method:
    python classical_finetuning.py --model random_forest --method optuna

# Multiple models, single method, custom iterations:
    python classical_finetuning.py --model logistic_regression random_forest \
        --method random_search --n-iter 20

# Skip calibration, change output directory:
    python classical_finetuning.py --model xgboost --method grid_search \
        --no-calibrate --models-dir my_models/

# Optuna-only run with 100 trials and 3 CV folds:
    python classical_finetuning.py --method optuna --n-trials 100 --cv-folds 3

Available models  : logistic_regression  decision_tree  random_forest  xgboost
Available methods : grid_search  random_search  optuna
"""

# ---------------------------------------------------------------------------
# 0 — CLI argument parsing (must come before heavy imports so --help is fast)
# ---------------------------------------------------------------------------

import argparse
import sys

VALID_MODELS = ["logistic_regression", "decision_tree", "random_forest", "xgboost"]
VALID_METHODS = ["grid_search", "random_search", "optuna"]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Hyperparameter tuning & calibration for classical ML models.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # ── Model selection ──────────────────────────────────────────────────────
    parser.add_argument(
        "--model", "-m",
        dest="models",
        nargs="+",
        choices=VALID_MODELS,
        default=VALID_MODELS,
        metavar="MODEL",
        help=(
            "Model(s) to tune. Pass one or more names separated by spaces. "
            f"Choices: {', '.join(VALID_MODELS)}. Default: all models."
        ),
    )

    # ── Tuning method ────────────────────────────────────────────────────────
    parser.add_argument(
        "--method",
        dest="methods",
        nargs="+",
        choices=VALID_METHODS,
        default=VALID_METHODS,
        metavar="METHOD",
        help=(
            "Tuning method(s) to run. Pass one or more names separated by spaces. "
            f"Choices: {', '.join(VALID_METHODS)}. Default: all methods."
        ),
    )

    # ── Tuning knobs ─────────────────────────────────────────────────────────
    parser.add_argument(
        "--n-iter",
        type=int,
        default=10,
        help="Number of iterations for RandomizedSearchCV (default: 10).",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=50,
        help="Number of trials for Optuna Bayesian optimisation (default: 50).",
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=5,
        help="Number of stratified K-fold cross-validation splits (default: 5).",
    )
    parser.add_argument(
        "--scoring",
        default="roc_auc",
        help="Sklearn scoring metric used by GridSearch / RandomSearch (default: roc_auc).",
    )

    # ── Calibration ──────────────────────────────────────────────────────────
    parser.add_argument(
        "--no-calibrate",
        action="store_true",
        default=False,
        help="Skip probability calibration step.",
    )
    parser.add_argument(
        "--calibration-method",
        choices=["isotonic", "sigmoid"],
        default="isotonic",
        help="Calibration method: 'isotonic' or 'sigmoid' (Platt). Default: isotonic.",
    )

    # ── Learning curves ───────────────────────────────────────────────────────
    parser.add_argument(
        "--no-learning-curves",
        action="store_true",
        default=False,
        help="Skip learning-curve plotting.",
    )

    # ── Output ───────────────────────────────────────────────────────────────
    parser.add_argument(
        "--models-dir",
        default="saved_models/",
        help="Directory where tuned models are saved (default: saved_models/).",
    )
    parser.add_argument(
        "--artifacts-dir",
        default="artifacts/",
        help="Directory where plots are saved (default: artifacts/).",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        default=False,
        help="Do not save tuned models to disk.",
    )

    # ── Misc ─────────────────────────────────────────────────────────────────
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=-1,
        help="Parallelism for sklearn searchers (default: -1 → all cores).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Global random seed (default: 42).",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging verbosity (default: INFO).",
    )

    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# 1 — Imports & Setup
# ---------------------------------------------------------------------------

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.model_selection import train_test_split, StratifiedKFold

from classical_models import (
    get_all_models, evaluate_model,
    FEATURE_COLUMNS, TARGET_RAIN, LOCATION_COLUMN, DATA_PATH,
)
from hyperparameter_tuning import (
    get_param_grids, get_param_distributions,
    run_grid_search, run_random_search,
    run_optuna_study,
    calibrate_model, compare_calibration,
    plot_learning_curves, plot_calibration_curves,
)
from training_logger import (
    setup_logger, log_training_metrics, log_tuning_results,
    log_calibration_results, log_model_save,
)
from tqdm.auto import tqdm


# ---------------------------------------------------------------------------
# 2 — Main entry point
# ---------------------------------------------------------------------------

def main(args=None):
    cfg = parse_args(args)

    # ── Directories ──────────────────────────────────────────────────────────
    os.makedirs(cfg.models_dir, exist_ok=True)
    os.makedirs(cfg.artifacts_dir, exist_ok=True)

    # ── Logger ───────────────────────────────────────────────────────────────
    logger = setup_logger("classical_finetuning", level=cfg.log_level)
    logger.info("=== Fine-Tuning Session Started ===")
    logger.info(f"Models   : {cfg.models}")
    logger.info(f"Methods  : {cfg.methods}")
    logger.info(f"CV folds : {cfg.cv_folds}  |  Seed: {cfg.seed}")

    print(f"\n{'='*60}")
    print(f"  Models  : {', '.join(cfg.models)}")
    print(f"  Methods : {', '.join(cfg.methods)}")
    print(f"  CV folds: {cfg.cv_folds}  |  Seed: {cfg.seed}")
    print(f"{'='*60}\n")

    # ── Data ─────────────────────────────────────────────────────────────────
    df = pd.read_csv(DATA_PATH)
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_RAIN]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=cfg.seed, stratify=y
    )
    print(f"Train : {X_train.shape}   Test : {X_test.shape}")
    print(f"Rain prevalence — Train: {y_train.mean():.2%}, Test: {y_test.mean():.2%}\n")
    logger.info(f"Data loaded: train={X_train.shape}, test={X_test.shape}")

    # ── Models ───────────────────────────────────────────────────────────────
    all_models = get_all_models()
    models = {name: all_models[name] for name in cfg.models}

    cv_strategy = StratifiedKFold(
        n_splits=cfg.cv_folds, shuffle=True, random_state=cfg.seed
    )

    # ── Learning curves ───────────────────────────────────────────────────────
    if not cfg.no_learning_curves:
        print("── Learning Curves ──────────────────────────────────────")
        fig, axes = plt.subplots(
            1, len(models), figsize=(6 * len(models), 5), squeeze=False
        )
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
        plt.tight_layout()
        out_path = os.path.join(cfg.artifacts_dir, "learning_curves.png")
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.show()
        print(f"Learning curves saved → {out_path}\n")

    # ── Tuning ───────────────────────────────────────────────────────────────
    grid_results   = {}   # name → GridSearchCV
    random_results = {}   # name → RandomizedSearchCV
    optuna_results = {}   # name → optuna.Study

    # Grid Search ─────────────────────────────────────────────────────────────
    if "grid_search" in cfg.methods:
        print("── GridSearchCV ─────────────────────────────────────────")
        param_grids = get_param_grids()
        for name, pipeline in tqdm(models.items(), desc="GridSearchCV", unit="model"):
            print(f"\n  Model: {name}")
            logger.info(f"GridSearch: {name}")
            gs = run_grid_search(
                pipeline, param_grids[name],
                X_train, y_train,
                cv=cv_strategy, scoring=cfg.scoring, n_jobs=cfg.n_jobs,
            )
            grid_results[name] = gs
            print(f"  Best {cfg.scoring}: {gs.best_score_:.4f}")
            print(f"  Best params : {gs.best_params_}")
            log_tuning_results(logger, name, "grid_search", gs.best_params_, gs.best_score_)
        print("GridSearchCV complete.\n")

    # Randomized Search ───────────────────────────────────────────────────────
    if "random_search" in cfg.methods:
        print("── RandomizedSearchCV ───────────────────────────────────")
        param_distributions = get_param_distributions()
        for name in tqdm(cfg.models, desc="RandomizedSearchCV", unit="model"):
            print(f"\n  Model: {name}")
            logger.info(f"RandomSearch: {name}")
            fresh_models = get_all_models()
            rs = run_random_search(
                fresh_models[name], param_distributions[name],
                X_train, y_train,
                n_iter=cfg.n_iter, cv=cv_strategy,
                scoring=cfg.scoring, n_jobs=cfg.n_jobs,
            )
            random_results[name] = rs
            print(f"  Best {cfg.scoring}: {rs.best_score_:.4f}")
            print(f"  Best params : {rs.best_params_}")
            log_tuning_results(logger, name, "random_search", rs.best_params_, rs.best_score_)
        print("RandomizedSearchCV complete.\n")

    # Optuna ──────────────────────────────────────────────────────────────────
    if "optuna" in cfg.methods:
        print("── Optuna Bayesian Optimisation ─────────────────────────")
        for name in tqdm(cfg.models, desc="Optuna", unit="model"):
            print(f"\n  Model: {name}  ({cfg.n_trials} trials)")
            logger.info(f"Optuna: {name}")
            study = run_optuna_study(
                name, X_train, y_train,
                n_trials=cfg.n_trials, cv=cv_strategy,
            )
            optuna_results[name] = study
            print(f"  Best {cfg.scoring}: {study.best_value:.4f}")
            print(f"  Best params : {study.best_params}")
            log_tuning_results(logger, name, "optuna", study.best_params, study.best_value)
        print("Optuna complete.\n")

    # ── Comparison table ─────────────────────────────────────────────────────
    if len(cfg.methods) > 1:
        print("── Tuning Method Comparison ─────────────────────────────")
        rows = []
        for name in cfg.models:
            row = {"Model": name}
            if name in grid_results:
                row["GridSearch"] = grid_results[name].best_score_
            if name in random_results:
                row["RandomSearch"] = random_results[name].best_score_
            if name in optuna_results:
                row["Optuna"] = optuna_results[name].best_value
            rows.append(row)
        comp_df = pd.DataFrame(rows).set_index("Model")
        print(comp_df.to_markdown())
        print()

    # ── Probability calibration ───────────────────────────────────────────────
    calibrated_models = {}
    if not cfg.no_calibrate and grid_results:
        print("── Probability Calibration ──────────────────────────────")
        for name in tqdm(cfg.models, desc="Calibrating", unit="model"):
            if name not in grid_results:
                print(f"  Skipping {name} — no GridSearch result (calibration uses GridSearch best estimator).")
                continue
            logger.info(f"Calibrating: {name}")
            best_model = grid_results[name].best_estimator_
            cal_model = calibrate_model(
                best_model, X_test, y_test, method=cfg.calibration_method
            )
            calibrated_models[name] = cal_model
            stats = compare_calibration(best_model, cal_model, X_test, y_test)
            print(
                f"  {name}\n"
                f"    Raw        → ROC-AUC: {stats['raw']['roc_auc']:.4f}  "
                f"Brier: {stats['raw']['brier_score']:.4f}\n"
                f"    Calibrated → ROC-AUC: {stats['calibrated']['roc_auc']:.4f}  "
                f"Brier: {stats['calibrated']['brier_score']:.4f}"
            )
            log_calibration_results(logger, name, stats)

        # Calibration curves
        if calibrated_models:
            cal_plot_models = {}
            for name in cfg.models:
                if name in grid_results:
                    cal_plot_models[f"{name} (raw)"] = grid_results[name].best_estimator_
                if name in calibrated_models:
                    cal_plot_models[f"{name} (cal)"] = calibrated_models[name]

            fig, ax = plt.subplots(figsize=(10, 8))
            plot_calibration_curves(cal_plot_models, X_test, y_test, ax=ax)
            plt.tight_layout()
            out_path = os.path.join(cfg.artifacts_dir, "calibration_curves.png")
            plt.savefig(out_path, dpi=150, bbox_inches="tight")
            plt.show()
            print(f"Calibration curves saved → {out_path}\n")
    elif not grid_results:
        print(" Skipping calibration — requires grid_search to be included in --method.\n")

    # ── Save best models ──────────────────────────────────────────────────────
    if not cfg.no_save and grid_results:
        print("── Saving Tuned Models ──────────────────────────────────")
        for name in cfg.models:
            if name not in grid_results:
                continue
            best_model = grid_results[name].best_estimator_
            model_path = os.path.join(cfg.models_dir, f"{name}_tuned.joblib")
            joblib.dump(best_model, model_path)
            print(f"  Saved {name} → {model_path}")
            log_model_save(logger, f"{name}_tuned", model_path)
        print("All tuned models saved.\n")

    logger.info("=== Fine-Tuning Session Complete ===")
    print("🎉 Fine-tuning session complete.")


if __name__ == "__main__":
    main()
