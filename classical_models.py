"""
Classical ML Models for Rain in Australia Prediction
=====================================================

Provides classical model pipelines along with a shared preprocessing step and
evaluation utilities.

**Models (classification):**
- Logistic Regression (L2, L1, ElasticNet variants)
- Decision Tree
- Random Forest
- XGBoost
- LightGBM
- CatBoost
- SVM (RBF kernel)
- VotingClassifier (soft voting)
- StackingClassifier (LogisticRegression meta-learner)

**Models (regression):**
- HistGradientBoostingRegressor (MaxTemp prediction)

Usage:
    from classical_models import get_all_models, evaluate_model
    from classical_models import FEATURE_COLUMNS, TARGET_RAIN
"""

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    StackingClassifier,
    VotingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
import joblib


# ---------------------------------------------------------------------------
# Constants — columns available in weatherAUS_clean_features_v2.csv
# ---------------------------------------------------------------------------

# Path to the clean feature-engineered dataset (v2 with Date column)
DATA_PATH = "weatherAUS_clean_features_v2.csv"

# The raw Location column (string) — preserved for filtering / display
LOCATION_COLUMN = "Location"

# Date column — preserved for temporal splitting
DATE_COLUMN = "Date"

FEATURE_COLUMNS = [
    "MinTemp",
    "MaxTemp",
    "Rainfall",
    "WindGustSpeed",
    "WindSpeed9am",
    "WindSpeed3pm",
    "Humidity9am",
    "Humidity3pm",
    "Pressure9am",
    "Pressure3pm",
    "Temp9am",
    "Temp3pm",
    "TempRange",
    "HumidityMean",
    "PressureChange",
    "Month_sin",
    "Month_cos",
    "WindDir3pm_sin",
    "WindDir3pm_cos",
    "RainToday_bin",
    "Location_rainrate",
]

TARGET_RAIN = "RainTomorrow_enc"
TARGET_TEMP = "MaxTemp"

# Columns that are already encoded / cyclical — no scaling needed
_PASSTHROUGH_COLUMNS = [
    "Month_sin",
    "Month_cos",
    "WindDir3pm_sin",
    "WindDir3pm_cos",
    "RainToday_bin",
    "Location_rainrate",
]

# Numeric columns that need standardisation
_NUMERIC_COLUMNS = [c for c in FEATURE_COLUMNS if c not in _PASSTHROUGH_COLUMNS]


# ---------------------------------------------------------------------------
# Dynamic class weight computation
# ---------------------------------------------------------------------------

def compute_scale_pos_weight(y):
    """Compute scale_pos_weight dynamically from the target vector.

    Parameters
    ----------
    y : array-like
        Binary target vector (0/1).

    Returns
    -------
    float — ratio of negatives to positives
    """
    y = np.asarray(y)
    n_neg = (y == 0).sum()
    n_pos = (y == 1).sum()
    if n_pos == 0:
        return 1.0
    return float(n_neg / n_pos)


# ---------------------------------------------------------------------------
# Preprocessor
# ---------------------------------------------------------------------------

def build_preprocessor(exclude=None) -> ColumnTransformer:
    """Return a ColumnTransformer that scales numeric features and passes
    through already-encoded features.

    Parameters
    ----------
    exclude : list[str] or None
        Column names to exclude from the preprocessor (e.g. ["MaxTemp"]
        when predicting MaxTemp to avoid data leakage).
    """
    exclude = set(exclude or [])
    num_cols = [c for c in _NUMERIC_COLUMNS if c not in exclude]
    pass_cols = [c for c in _PASSTHROUGH_COLUMNS if c not in exclude]
    return ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), num_cols),
            ("passthrough", "passthrough", pass_cols),
        ],
        remainder="drop",
    )


# ---------------------------------------------------------------------------
# Model builders — each returns a full sklearn Pipeline
# ---------------------------------------------------------------------------

def build_logistic_regression(penalty="l2", C=1.0) -> Pipeline:
    """Logistic Regression pipeline with class-weight balancing.

    Parameters
    ----------
    penalty : str
        Regularization type: 'l2', 'l1', or 'elasticnet'.
    C : float
        Inverse regularization strength.
    """
    solver_map = {"l2": "lbfgs", "l1": "liblinear", "elasticnet": "saga"}
    solver = solver_map.get(penalty, "lbfgs")
    kwargs = {}
    if penalty == "elasticnet":
        kwargs["l1_ratio"] = 0.5

    return Pipeline(
        [
            ("preprocessor", build_preprocessor()),
            (
                "classifier",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    random_state=42,
                    solver=solver,
                    penalty=penalty,
                    C=C,
                    **kwargs,
                ),
            ),
        ]
    )


def build_decision_tree() -> Pipeline:
    """Decision Tree pipeline with depth / leaf constraints."""
    return Pipeline(
        [
            ("preprocessor", build_preprocessor()),
            (
                "classifier",
                DecisionTreeClassifier(
                    max_depth=10,
                    min_samples_leaf=50,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )


def build_random_forest() -> Pipeline:
    """Random Forest pipeline with 200 estimators."""
    return Pipeline(
        [
            ("preprocessor", build_preprocessor()),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=200,
                    max_depth=15,
                    min_samples_leaf=20,
                    class_weight="balanced",
                    n_jobs=-1,
                    random_state=42,
                ),
            ),
        ]
    )


def build_xgboost(scale_pos_weight=3.5) -> Pipeline:
    """XGBoost pipeline with scale_pos_weight for class imbalance.

    Parameters
    ----------
    scale_pos_weight : float
        Class imbalance ratio. Pass ``compute_scale_pos_weight(y_train)``
        for a data-driven value instead of the default.
    """
    return Pipeline(
        [
            ("preprocessor", build_preprocessor()),
            (
                "classifier",
                XGBClassifier(
                    n_estimators=200,
                    max_depth=6,
                    learning_rate=0.05,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    scale_pos_weight=scale_pos_weight,
                    eval_metric="logloss",
                    random_state=42,
                ),
            ),
        ]
    )


def build_lightgbm(scale_pos_weight=3.5) -> Pipeline:
    """LightGBM classifier pipeline.

    Parameters
    ----------
    scale_pos_weight : float
        Class imbalance ratio.
    """
    return Pipeline(
        [
            ("preprocessor", build_preprocessor()),
            (
                "classifier",
                LGBMClassifier(
                    n_estimators=200,
                    max_depth=8,
                    learning_rate=0.05,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    scale_pos_weight=scale_pos_weight,
                    random_state=42,
                    verbose=-1,
                ),
            ),
        ]
    )


def build_catboost(scale_pos_weight=3.5) -> Pipeline:
    """CatBoost classifier pipeline.

    Parameters
    ----------
    scale_pos_weight : float
        Class imbalance ratio. Converted to ``auto_class_weights`` or
        explicit ``class_weights``.
    """
    return Pipeline(
        [
            ("preprocessor", build_preprocessor()),
            (
                "classifier",
                CatBoostClassifier(
                    iterations=200,
                    depth=6,
                    learning_rate=0.05,
                    scale_pos_weight=scale_pos_weight,
                    random_seed=42,
                    verbose=0,
                    allow_writing_files=False,
                ),
            ),
        ]
    )


def build_svm() -> Pipeline:
    """SVM with RBF kernel. Uses probability=True for ROC-AUC scoring."""
    return Pipeline(
        [
            ("preprocessor", build_preprocessor()),
            (
                "classifier",
                SVC(
                    kernel="rbf",
                    C=1.0,
                    gamma="scale",
                    class_weight="balanced",
                    probability=True,
                    random_state=42,
                ),
            ),
        ]
    )


def build_voting_classifier(estimators_dict=None) -> VotingClassifier:
    """Soft VotingClassifier over the provided estimators.

    Parameters
    ----------
    estimators_dict : dict[str, Pipeline] or None
        Dict of name → fitted/unfitted Pipeline. If None, builds default
        set (LR, RF, XGBoost, LightGBM).

    Returns
    -------
    VotingClassifier (unfitted)
    """
    if estimators_dict is None:
        estimators_dict = {
            "lr": build_logistic_regression(),
            "rf": build_random_forest(),
            "xgb": build_xgboost(),
            "lgbm": build_lightgbm(),
        }
    estimators = list(estimators_dict.items())
    return VotingClassifier(estimators=estimators, voting="soft", n_jobs=-1)


def build_stacking_classifier(estimators_dict=None, meta_learner=None) -> StackingClassifier:
    """StackingClassifier with a LogisticRegression meta-learner.

    Parameters
    ----------
    estimators_dict : dict[str, Pipeline] or None
        Base estimators. If None, uses default set.
    meta_learner : estimator or None
        Final estimator. Defaults to LogisticRegression.

    Returns
    -------
    StackingClassifier (unfitted)
    """
    if estimators_dict is None:
        estimators_dict = {
            "lr": build_logistic_regression(),
            "rf": build_random_forest(),
            "xgb": build_xgboost(),
            "lgbm": build_lightgbm(),
        }
    if meta_learner is None:
        meta_learner = LogisticRegression(
            max_iter=2000, class_weight="balanced", random_state=42
        )
    estimators = list(estimators_dict.items())
    return StackingClassifier(
        estimators=estimators,
        final_estimator=meta_learner,
        cv=5,
        n_jobs=-1,
        passthrough=False,
    )


# ---------------------------------------------------------------------------
# Convenience accessors
# ---------------------------------------------------------------------------

def get_all_models(
    include_ensemble=False,
    load_saved=False,
    scale_pos_weight=3.5,
) -> dict:
    """Return a dict of model-name → Pipeline for classical models.

    Parameters
    ----------
    include_ensemble : bool
        If True, include VotingClassifier and StackingClassifier.
    load_saved : bool
        If True, load pre-trained models from disk.
    scale_pos_weight : float
        Class imbalance ratio for XGBoost / LightGBM / CatBoost.
    """
    if load_saved:
        return {
            "logistic_regression": joblib.load("saved_models/logistic_regression.joblib"),
            "decision_tree": joblib.load("saved_models/decision_tree.joblib"),
            "random_forest": joblib.load("saved_models/random_forest.joblib"),
            "xgboost": joblib.load("saved_models/xgboost.joblib"),
            "temp_regressor": joblib.load("saved_models/temp_regressor.joblib"),
        }

    models = {
        "logistic_regression": build_logistic_regression(),
        "logistic_regression_l1": build_logistic_regression(penalty="l1"),
        "logistic_regression_elasticnet": build_logistic_regression(penalty="elasticnet"),
        "decision_tree": build_decision_tree(),
        "random_forest": build_random_forest(),
        "xgboost": build_xgboost(scale_pos_weight=scale_pos_weight),
        "lightgbm": build_lightgbm(scale_pos_weight=scale_pos_weight),
        "catboost": build_catboost(scale_pos_weight=scale_pos_weight),
        "svm": build_svm(),
    }

    if include_ensemble:
        base = {
            "lr": models["logistic_regression"],
            "rf": models["random_forest"],
            "xgb": models["xgboost"],
            "lgbm": models["lightgbm"],
        }
        models["voting"] = build_voting_classifier(base)
        models["stacking"] = build_stacking_classifier(base)

    return models


# ---------------------------------------------------------------------------
# Evaluation utility — Classification (assumes model is already fitted)
# ---------------------------------------------------------------------------

def evaluate_model(model, X_test, y_test) -> dict:
    """Compute standard classification metrics on a fitted model.

    Parameters
    ----------
    model : a *fitted* sklearn Pipeline or estimator
    X_test : feature DataFrame / array
    y_test : true labels (0/1)

    Returns
    -------
    dict with accuracy, roc_auc, f1_score, precision, recall, mcc,
         confusion_matrix
    """
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    return {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "roc_auc": float(roc_auc_score(y_test, y_proba)),
        "f1_score": float(f1_score(y_test, y_pred, average="binary")),
        "precision": float(precision_score(y_test, y_pred)),
        "recall": float(recall_score(y_test, y_pred)),
        "mcc": float(matthews_corrcoef(y_test, y_pred)),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }


# ---------------------------------------------------------------------------
# Evaluation utility — Regression
# ---------------------------------------------------------------------------

def evaluate_regressor(model, X_test, y_test) -> dict:
    """MAE, RMSE, and R² for a fitted regressor pipeline.

    Parameters
    ----------
    model : a *fitted* sklearn Pipeline
    X_test : feature DataFrame / array
    y_test : true target values

    Returns
    -------
    dict with mae, rmse, r2
    """
    y_pred = model.predict(X_test)
    return {
        "mae": float(mean_absolute_error(y_test, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
        "r2": float(r2_score(y_test, y_pred)),
    }


# ---------------------------------------------------------------------------
# Temperature Regressor
# ---------------------------------------------------------------------------

def build_temp_regressor() -> Pipeline:
    """HistGradientBoosting regressor for MaxTemp prediction.

    Uses a preprocessor that *excludes* MaxTemp to avoid data leakage
    (MaxTemp is the prediction target).

    Note: HistGradientBoostingRegressor replaces the older
    GradientBoostingRegressor for better performance on large datasets
    (native histogram-based binning, NaN support, lower memory).
    """
    return Pipeline(
        [
            ("preprocessor", build_preprocessor(exclude=["MaxTemp"])),
            (
                "regressor",
                HistGradientBoostingRegressor(
                    max_iter=200,
                    max_depth=5,
                    learning_rate=0.05,
                    random_state=42,
                ),
            ),
        ]
    )
