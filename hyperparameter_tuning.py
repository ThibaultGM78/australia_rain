"""
Hyperparameter Tuning, Overfitting Monitoring & Probability Calibration
========================================================================

Provides utilities for:
- GridSearchCV / RandomizedSearchCV parameter grids
- Optuna-based Bayesian optimisation
- Probability calibration via CalibratedClassifierCV
- Learning curve plotting for overfitting detection
- Raw vs calibrated model comparison

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
    roc_auc_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    StratifiedKFold,
    learning_curve,
)
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

import optuna
from optuna.samplers import TPESampler

from classical_models import build_preprocessor, FEATURE_COLUMNS, TARGET_RAIN


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
    cv : cross-validation strategy (default: 5-fold stratified)
    scoring : metric to optimise (default: roc_auc)
    n_jobs : parallelism

    Returns
    -------
    GridSearchCV (fitted)
    """
    if cv is None:
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

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
    print(f"  ⏳ GridSearchCV fitting ({len(gs.get_params()['param_grid'] or param_grid)} param combos × {cv.get_n_splits()} folds)...")
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
    n_iter : number of random combinations to try
    """
    if cv is None:
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

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
    print(f"  ⏳ RandomizedSearchCV fitting ({n_iter} iterations × {cv.get_n_splits()} folds)...")
    rs.fit(X_train, y_train)
    print(f"  ✅ RandomizedSearchCV done — best score: {rs.best_score_:.4f}")
    return rs


# ---------------------------------------------------------------------------
# 4. Optuna — Bayesian hyperparameter optimisation
# ---------------------------------------------------------------------------

def build_optuna_objective(model_name, X_train, y_train, cv=None):
    """Return an Optuna objective function for the given model.

    Parameters
    ----------
    model_name : one of "logistic_regression", "decision_tree",
                 "random_forest", "xgboost"
    X_train, y_train : training data
    cv : cross-validation strategy

    Returns
    -------
    callable : objective(trial) → float (ROC-AUC)
    """
    from sklearn.model_selection import cross_val_score
    from sklearn.pipeline import Pipeline

    if cv is None:
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    def objective(trial):
        preprocessor = build_preprocessor()

        if model_name == "logistic_regression":
            classifier = LogisticRegression(
                C=trial.suggest_float("C", 1e-3, 100.0, log=True),
                solver=trial.suggest_categorical("solver", ["lbfgs", "liblinear"]),
                class_weight="balanced",
                max_iter=1000,
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
                scale_pos_weight=3.5,
                eval_metric="logloss",
                random_state=42,
                use_label_encoder=False,
            )
        else:
            raise ValueError(f"Unknown model: {model_name}")

        pipe = Pipeline([("preprocessor", preprocessor), ("classifier", classifier)])
        scores = cross_val_score(pipe, X_train, y_train, cv=cv,
                                 scoring="roc_auc", n_jobs=-1)
        return scores.mean()

    return objective


def run_optuna_study(model_name, X_train, y_train,
                     n_trials=50, cv=None, direction="maximize"):
    """Create and run an Optuna study for the given model.

    Returns
    -------
    optuna.Study
    """
    objective = build_optuna_objective(model_name, X_train, y_train, cv=cv)
    study = optuna.create_study(
        direction=direction,
        sampler=TPESampler(seed=42),
        study_name=f"tune_{model_name}",
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    return study


# ---------------------------------------------------------------------------
# 5. Probability calibration
# ---------------------------------------------------------------------------

def calibrate_model(fitted_pipeline, X_cal, y_cal, method="isotonic", cv=5):
    """Wrap a fitted pipeline with CalibratedClassifierCV.

    Parameters
    ----------
    fitted_pipeline : an already-fitted sklearn Pipeline
    X_cal, y_cal : calibration data (can be same as validation set)
    method : "isotonic" or "sigmoid" (Platt scaling)
    cv : int or "prefit" — use "prefit" when pipeline is already fitted

    Returns
    -------
    CalibratedClassifierCV (fitted)
    """
    calibrated = CalibratedClassifierCV(
        estimator=fitted_pipeline,
        method=method,
        cv="prefit",
    )
    calibrated.fit(X_cal, y_cal)
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
        },
        "calibrated": {
            "roc_auc": float(roc_auc_score(y_test, cal_proba)),
            "brier_score": float(brier_score_loss(y_test, cal_proba)),
            "log_loss": float(log_loss(y_test, cal_proba)),
            "f1_score": float(f1_score(y_test, cal_pred, average="binary")),
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
    cv : cross-validation strategy
    scoring : metric
    ax : matplotlib Axes (optional)

    Returns
    -------
    matplotlib Axes
    """
    import matplotlib.pyplot as plt

    if cv is None:
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    print(f"  ⏳ Computing learning curve (10 train sizes × {cv.get_n_splits()} folds)...")
    train_sizes, train_scores, val_scores = learning_curve(
        pipeline, X_train, y_train,
        cv=cv,
        scoring=scoring,
        n_jobs=n_jobs,
        train_sizes=np.linspace(0.1, 1.0, 10),
        random_state=42,
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
