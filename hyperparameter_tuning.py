"""
Hyperparameter Tuning, Overfitting Monitoring & Probability Calibration
========================================================================

Provides utilities for:
- GridSearchCV / RandomizedSearchCV parameter grids
- Optuna-based Bayesian optimisation with MedianPruner
- Probability calibration via CalibratedClassifierCV
- Learning curve plotting for overfitting detection
- Raw vs calibrated model comparison
- Train-vs-validation gap analysis

Usage:
    from hyperparameter_tuning import (
        get_param_grids, get_param_distributions,
        build_optuna_objective, run_optuna_study,
        calibrate_model, compare_calibration,
        plot_learning_curves,
    )
"""

import numpy as np
from tqdm.auto import tqdm
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    brier_score_loss,
    f1_score,
    log_loss,
    matthews_corrcoef,
    roc_auc_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    TimeSeriesSplit,
    learning_curve,
)
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner

from classical_models import build_preprocessor, FEATURE_COLUMNS, TARGET_RAIN


# ---------------------------------------------------------------------------
# Default CV strategy — TimeSeriesSplit (replaces StratifiedKFold)
# ---------------------------------------------------------------------------

def _default_cv(n_splits=5):
    """Return a TimeSeriesSplit cross-validator.

    For weather data, temporal ordering must be respected.
    StratifiedKFold would leak future data into training folds.
    """
    return TimeSeriesSplit(n_splits=n_splits)


# ---------------------------------------------------------------------------
# 1. Parameter grids — for GridSearchCV
# ---------------------------------------------------------------------------

def get_param_grids() -> dict:
    """Return parameter grids for GridSearchCV, keyed by model name.

    Keys use the sklearn Pipeline convention: ``classifier__<param>``.
    """
    return {
        "logistic_regression": {
            "classifier__C": [0.01, 0.1, 1.0, 10.0],
            "classifier__solver": ["lbfgs", "liblinear"],
            "classifier__penalty": ["l2"],
        },
        "logistic_regression_l1": {
            "classifier__C": [0.01, 0.1, 1.0, 10.0],
            "classifier__solver": ["liblinear"],
            "classifier__penalty": ["l1"],
        },
        "logistic_regression_elasticnet": {
            "classifier__C": [0.01, 0.1, 1.0, 10.0],
            "classifier__solver": ["saga"],
            "classifier__penalty": ["elasticnet"],
            "classifier__l1_ratio": [0.2, 0.5, 0.8],
        },
        "decision_tree": {
            "classifier__max_depth": [5, 10, 15, 20],
            "classifier__min_samples_leaf": [20, 50, 100],
            "classifier__min_samples_split": [2, 5, 10],
        },
        "random_forest": {
            "classifier__n_estimators": [100, 200, 300],
            "classifier__max_depth": [10, 15, 20, None],
            "classifier__min_samples_leaf": [10, 20, 50],
        },
        "xgboost": {
            "classifier__n_estimators": [100, 200, 300],
            "classifier__max_depth": [4, 6, 8],
            "classifier__learning_rate": [0.01, 0.05, 0.1],
            "classifier__subsample": [0.7, 0.8, 0.9],
            "classifier__colsample_bytree": [0.7, 0.8, 0.9],
        },
        "lightgbm": {
            "classifier__n_estimators": [100, 200, 300],
            "classifier__max_depth": [6, 8, 10],
            "classifier__learning_rate": [0.01, 0.05, 0.1],
            "classifier__subsample": [0.7, 0.8, 0.9],
            "classifier__colsample_bytree": [0.7, 0.8, 0.9],
        },
        "catboost": {
            "classifier__iterations": [100, 200, 300],
            "classifier__depth": [4, 6, 8],
            "classifier__learning_rate": [0.01, 0.05, 0.1],
        },
        "svm": {
            "classifier__C": [0.1, 1.0, 10.0],
            "classifier__gamma": ["scale", "auto"],
            "classifier__kernel": ["rbf", "poly"],
        },
    }


# ---------------------------------------------------------------------------
# 2. Parameter distributions — for RandomizedSearchCV
# ---------------------------------------------------------------------------

def get_param_distributions() -> dict:
    """Return parameter distributions for RandomizedSearchCV, keyed by model name.

    Uses scipy-compatible distributions for continuous params and lists for
    discrete params.
    """
    from scipy.stats import loguniform, randint, uniform

    return {
        "logistic_regression": {
            "classifier__C": loguniform(1e-3, 1e2),
            "classifier__solver": ["lbfgs", "liblinear"],
            "classifier__penalty": ["l2"],
        },
        "logistic_regression_l1": {
            "classifier__C": loguniform(1e-3, 1e2),
            "classifier__solver": ["liblinear"],
            "classifier__penalty": ["l1"],
        },
        "logistic_regression_elasticnet": {
            "classifier__C": loguniform(1e-3, 1e2),
            "classifier__solver": ["saga"],
            "classifier__penalty": ["elasticnet"],
            "classifier__l1_ratio": uniform(0.1, 0.8),
        },
        "decision_tree": {
            "classifier__max_depth": randint(3, 30),
            "classifier__min_samples_leaf": randint(10, 200),
            "classifier__min_samples_split": randint(2, 20),
        },
        "random_forest": {
            "classifier__n_estimators": randint(50, 500),
            "classifier__max_depth": randint(5, 30),
            "classifier__min_samples_leaf": randint(5, 100),
        },
        "xgboost": {
            "classifier__n_estimators": randint(50, 500),
            "classifier__max_depth": randint(3, 12),
            "classifier__learning_rate": loguniform(1e-3, 0.3),
            "classifier__subsample": uniform(0.6, 0.4),
            "classifier__colsample_bytree": uniform(0.6, 0.4),
        },
        "lightgbm": {
            "classifier__n_estimators": randint(50, 500),
            "classifier__max_depth": randint(4, 15),
            "classifier__learning_rate": loguniform(1e-3, 0.3),
            "classifier__subsample": uniform(0.6, 0.4),
            "classifier__colsample_bytree": uniform(0.6, 0.4),
            "classifier__num_leaves": randint(20, 100),
        },
        "catboost": {
            "classifier__iterations": randint(50, 500),
            "classifier__depth": randint(3, 10),
            "classifier__learning_rate": loguniform(1e-3, 0.3),
            "classifier__l2_leaf_reg": loguniform(1, 10),
        },
        "svm": {
            "classifier__C": loguniform(1e-2, 1e2),
            "classifier__gamma": ["scale", "auto"],
            "classifier__kernel": ["rbf", "poly"],
        },
    }


# ---------------------------------------------------------------------------
# 3. GridSearchCV / RandomizedSearchCV runners
# ---------------------------------------------------------------------------

def run_grid_search(pipeline, param_grid, X_train, y_train,
                    cv=None, scoring="roc_auc", n_jobs=-1):
    """Run GridSearchCV and return the fitted searcher.

    Parameters
    ----------
    pipeline : sklearn Pipeline (unfitted)
    param_grid : dict of parameter name → list of values
    X_train, y_train : training data
    cv : cross-validation strategy (default: 5-fold TimeSeriesSplit)
    scoring : metric to optimise (default: roc_auc)
    n_jobs : parallelism

    Returns
    -------
    GridSearchCV (fitted)
    """
    if cv is None:
        cv = _default_cv()

    gs = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=cv,
        scoring=scoring,
        n_jobs=n_jobs,
        refit=True,
        return_train_score=True,
        verbose=2,
    )
    n_splits = cv.get_n_splits() if hasattr(cv, 'get_n_splits') else cv
    print(f"  ⏳ GridSearchCV fitting × {n_splits} folds (TimeSeriesSplit)...")
    gs.fit(X_train, y_train)
    print(f"  ✅ GridSearchCV done — best score: {gs.best_score_:.4f}")
    return gs


def run_random_search(pipeline, param_distributions, X_train, y_train,
                      n_iter=50, cv=None, scoring="roc_auc", n_jobs=-1):
    """Run RandomizedSearchCV and return the fitted searcher.

    Parameters
    ----------
    pipeline : sklearn Pipeline (unfitted)
    param_distributions : dict of parameter name → distribution
    n_iter : number of random combinations to try (default: 50, up from 10)
    """
    if cv is None:
        cv = _default_cv()

    rs = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_distributions,
        n_iter=n_iter,
        cv=cv,
        scoring=scoring,
        n_jobs=n_jobs,
        refit=True,
        return_train_score=True,
        random_state=42,
        verbose=2,
    )
    n_splits = cv.get_n_splits() if hasattr(cv, 'get_n_splits') else cv
    print(f"  ⏳ RandomizedSearchCV fitting ({n_iter} iterations × {n_splits} folds, TimeSeriesSplit)...")
    rs.fit(X_train, y_train)
    print(f"  ✅ RandomizedSearchCV done — best score: {rs.best_score_:.4f}")
    return rs


# ---------------------------------------------------------------------------
# Train vs Validation gap analysis
# ---------------------------------------------------------------------------

def extract_train_val_gap(search_cv):
    """Extract train vs validation score gap from a fitted GridSearchCV
    or RandomizedSearchCV.

    Parameters
    ----------
    search_cv : fitted GridSearchCV or RandomizedSearchCV
        Must have been run with ``return_train_score=True``.

    Returns
    -------
    dict with best train/val scores and the gap
    """
    results = search_cv.cv_results_
    best_idx = search_cv.best_index_

    train_score = results["mean_train_score"][best_idx]
    val_score = results["mean_test_score"][best_idx]
    gap = train_score - val_score

    return {
        "train_score": float(train_score),
        "val_score": float(val_score),
        "gap": float(gap),
        "overfitting_risk": "HIGH" if gap > 0.05 else "MODERATE" if gap > 0.02 else "LOW",
    }


# ---------------------------------------------------------------------------
# 4. Optuna — Bayesian hyperparameter optimisation with pruning
# ---------------------------------------------------------------------------

def build_optuna_objective(model_name, X_train, y_train, cv=None,
                           scale_pos_weight=3.5):
    """Return an Optuna objective function for the given model.

    Parameters
    ----------
    model_name : one of the supported model names
    X_train, y_train : training data
    cv : cross-validation strategy
    scale_pos_weight : float
        Class imbalance ratio for boosting models.

    Returns
    -------
    callable : objective(trial) → float (ROC-AUC)
    """
    from sklearn.model_selection import cross_val_score
    from sklearn.pipeline import Pipeline

    if cv is None:
        cv = _default_cv()

    def objective(trial):
        preprocessor = build_preprocessor()

        if model_name == "logistic_regression":
            classifier = LogisticRegression(
                C=trial.suggest_float("C", 1e-3, 100.0, log=True),
                solver=trial.suggest_categorical("solver", ["lbfgs", "liblinear"]),
                class_weight="balanced",
                max_iter=2000,
                random_state=42,
            )
        elif model_name == "logistic_regression_l1":
            classifier = LogisticRegression(
                C=trial.suggest_float("C", 1e-3, 100.0, log=True),
                solver="liblinear",
                penalty="l1",
                class_weight="balanced",
                max_iter=2000,
                random_state=42,
            )
        elif model_name == "logistic_regression_elasticnet":
            classifier = LogisticRegression(
                C=trial.suggest_float("C", 1e-3, 100.0, log=True),
                solver="saga",
                penalty="elasticnet",
                l1_ratio=trial.suggest_float("l1_ratio", 0.1, 0.9),
                class_weight="balanced",
                max_iter=2000,
                random_state=42,
            )
        elif model_name == "decision_tree":
            classifier = DecisionTreeClassifier(
                max_depth=trial.suggest_int("max_depth", 3, 30),
                min_samples_leaf=trial.suggest_int("min_samples_leaf", 10, 200),
                min_samples_split=trial.suggest_int("min_samples_split", 2, 20),
                class_weight="balanced",
                random_state=42,
            )
        elif model_name == "random_forest":
            classifier = RandomForestClassifier(
                n_estimators=trial.suggest_int("n_estimators", 50, 500),
                max_depth=trial.suggest_int("max_depth", 5, 30),
                min_samples_leaf=trial.suggest_int("min_samples_leaf", 5, 100),
                class_weight="balanced",
                n_jobs=-1,
                random_state=42,
            )
        elif model_name == "xgboost":
            classifier = XGBClassifier(
                n_estimators=trial.suggest_int("n_estimators", 50, 500),
                max_depth=trial.suggest_int("max_depth", 3, 12),
                learning_rate=trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
                subsample=trial.suggest_float("subsample", 0.6, 1.0),
                colsample_bytree=trial.suggest_float("colsample_bytree", 0.6, 1.0),
                scale_pos_weight=scale_pos_weight,
                eval_metric="logloss",
                random_state=42,
            )
        elif model_name == "lightgbm":
            classifier = LGBMClassifier(
                n_estimators=trial.suggest_int("n_estimators", 50, 500),
                max_depth=trial.suggest_int("max_depth", 4, 15),
                learning_rate=trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
                subsample=trial.suggest_float("subsample", 0.6, 1.0),
                colsample_bytree=trial.suggest_float("colsample_bytree", 0.6, 1.0),
                num_leaves=trial.suggest_int("num_leaves", 20, 100),
                scale_pos_weight=scale_pos_weight,
                random_state=42,
                verbose=-1,
            )
        elif model_name == "catboost":
            classifier = CatBoostClassifier(
                iterations=trial.suggest_int("iterations", 50, 500),
                depth=trial.suggest_int("depth", 3, 10),
                learning_rate=trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
                l2_leaf_reg=trial.suggest_float("l2_leaf_reg", 1.0, 10.0, log=True),
                scale_pos_weight=scale_pos_weight,
                random_seed=42,
                verbose=0,
                allow_writing_files=False,
            )
        elif model_name == "svm":
            classifier = SVC(
                C=trial.suggest_float("C", 1e-2, 100.0, log=True),
                gamma=trial.suggest_categorical("gamma", ["scale", "auto"]),
                kernel=trial.suggest_categorical("kernel", ["rbf", "poly"]),
                class_weight="balanced",
                probability=True,
                random_state=42,
            )
        else:
            raise ValueError(f"Unknown model: {model_name}")

        pipe = Pipeline([("preprocessor", preprocessor), ("classifier", classifier)])
        scores = cross_val_score(pipe, X_train, y_train, cv=cv,
                                 scoring="roc_auc", n_jobs=-1)
        return scores.mean()

    return objective


def run_optuna_study(model_name, X_train, y_train,
                     n_trials=50, cv=None, direction="maximize",
                     scale_pos_weight=3.5):
    """Create and run an Optuna study for the given model.

    Includes MedianPruner for early stopping of unpromising trials.

    Returns
    -------
    optuna.Study
    """
    objective = build_optuna_objective(
        model_name, X_train, y_train, cv=cv,
        scale_pos_weight=scale_pos_weight,
    )
    study = optuna.create_study(
        direction=direction,
        sampler=TPESampler(seed=42),
        pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=3),
        study_name=f"tune_{model_name}",
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    return study


# ---------------------------------------------------------------------------
# 5. Probability calibration
# ---------------------------------------------------------------------------

def calibrate_model(fitted_pipeline, X_cal, y_cal, method="isotonic"):
    """Wrap a fitted pipeline with CalibratedClassifierCV (prefit mode).

    Parameters
    ----------
    fitted_pipeline : an already-fitted sklearn Pipeline
    X_cal, y_cal : **dedicated calibration data** — must NOT be the test set!
    method : "isotonic" or "sigmoid" (Platt scaling)

    Returns
    -------
    CalibratedClassifierCV (fitted)

    .. warning::
        Pass a dedicated calibration set here, never the test set.
        Calibrating on the test set biases evaluation.
    """
    calibrated = CalibratedClassifierCV(
        estimator=fitted_pipeline,
        method=method,
        cv="prefit",
    )
    calibrated.fit(X_cal, y_cal)
    return calibrated


def calibrate_model_cv(unfitted_pipeline, X_train, y_train, method="isotonic", cv=5):
    """Calibrate via internal cross-validation (when no dedicated set exists).

    Parameters
    ----------
    unfitted_pipeline : sklearn Pipeline (NOT fitted)
    X_train, y_train : training data
    method : calibration method
    cv : number of folds for internal calibration

    Returns
    -------
    CalibratedClassifierCV (fitted)
    """
    calibrated = CalibratedClassifierCV(
        estimator=unfitted_pipeline,
        method=method,
        cv=cv,
    )
    calibrated.fit(X_train, y_train)
    return calibrated


def compare_calibration(raw_model, calibrated_model, X_test, y_test):
    """Compare raw vs calibrated model on test data.

    Returns
    -------
    dict with metrics for both raw and calibrated models
    """
    raw_proba = raw_model.predict_proba(X_test)[:, 1]
    cal_proba = calibrated_model.predict_proba(X_test)[:, 1]

    raw_pred = raw_model.predict(X_test)
    cal_pred = calibrated_model.predict(X_test)

    results = {
        "raw": {
            "roc_auc": float(roc_auc_score(y_test, raw_proba)),
            "brier_score": float(brier_score_loss(y_test, raw_proba)),
            "log_loss": float(log_loss(y_test, raw_proba)),
            "f1_score": float(f1_score(y_test, raw_pred, average="binary")),
            "mcc": float(matthews_corrcoef(y_test, raw_pred)),
        },
        "calibrated": {
            "roc_auc": float(roc_auc_score(y_test, cal_proba)),
            "brier_score": float(brier_score_loss(y_test, cal_proba)),
            "log_loss": float(log_loss(y_test, cal_proba)),
            "f1_score": float(f1_score(y_test, cal_pred, average="binary")),
            "mcc": float(matthews_corrcoef(y_test, cal_pred)),
        },
    }
    return results


# ---------------------------------------------------------------------------
# 6. Overfitting monitoring — learning curves
# ---------------------------------------------------------------------------

def plot_learning_curves(pipeline, X_train, y_train, cv=None,
                         scoring="roc_auc", n_jobs=-1, ax=None):
    """Plot learning curves (train vs validation) to detect overfitting.

    Parameters
    ----------
    pipeline : sklearn Pipeline (unfitted)
    X_train, y_train : training data
    cv : cross-validation strategy (default: TimeSeriesSplit)
    scoring : metric
    ax : matplotlib Axes (optional)

    Returns
    -------
    matplotlib Axes
    """
    import matplotlib.pyplot as plt

    if cv is None:
        cv = _default_cv()

    n_splits = cv.get_n_splits() if hasattr(cv, 'get_n_splits') else cv
    print(f"  ⏳ Computing learning curve (10 train sizes × {n_splits} folds, TimeSeriesSplit)...")
    train_sizes, train_scores, val_scores = learning_curve(
        pipeline, X_train, y_train,
        cv=cv,
        scoring=scoring,
        n_jobs=n_jobs,
        train_sizes=np.linspace(0.1, 1.0, 10),
        shuffle=False,  # Preserve temporal order
        verbose=1,
    )
    print("  ✅ Learning curve computed.")

    train_mean = train_scores.mean(axis=1)
    train_std = train_scores.std(axis=1)
    val_mean = val_scores.mean(axis=1)
    val_std = val_scores.std(axis=1)

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))

    ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std,
                    alpha=0.15, color="tab:blue")
    ax.fill_between(train_sizes, val_mean - val_std, val_mean + val_std,
                    alpha=0.15, color="tab:orange")
    ax.plot(train_sizes, train_mean, "o-", color="tab:blue", label="Training")
    ax.plot(train_sizes, val_mean, "o-", color="tab:orange", label="Validation")
    ax.set_xlabel("Training set size")
    ax.set_ylabel(scoring.replace("_", " ").title())
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)

    # Annotate final gap
    gap = train_mean[-1] - val_mean[-1]
    ax.annotate(f"Δ = {gap:.4f}",
                xy=(train_sizes[-1], val_mean[-1]),
                xytext=(train_sizes[-1] * 0.8, val_mean[-1] + 0.02),
                arrowprops=dict(arrowstyle="->", color="red"),
                fontsize=9, color="red")

    return ax


def plot_calibration_curves(models_dict, X_test, y_test, n_bins=10, ax=None):
    """Plot calibration curves for multiple models.

    Parameters
    ----------
    models_dict : dict of {name: fitted_model}
    X_test, y_test : test data
    n_bins : number of bins for calibration curve
    ax : matplotlib Axes (optional)

    Returns
    -------
    matplotlib Axes
    """
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 6))

    ax.plot([0, 1], [0, 1], "k--", label="Perfectly calibrated", alpha=0.5)

    for name, model in models_dict.items():
        proba = model.predict_proba(X_test)[:, 1]
        fraction_pos, mean_predicted = calibration_curve(
            y_test, proba, n_bins=n_bins, strategy="uniform"
        )
        brier = brier_score_loss(y_test, proba)
        ax.plot(mean_predicted, fraction_pos, "s-",
                label=f"{name} (Brier={brier:.4f})")

    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.set_title("Calibration Curves")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)

    return ax
