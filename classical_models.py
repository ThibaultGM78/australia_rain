"""
Classical ML Models for Rain in Australia Prediction
=====================================================

Provides four classical model pipelines (Logistic Regression, Decision Tree,
Random Forest, XGBoost) along with a shared preprocessing step and an
evaluation utility.

Usage:
    from classical_models import get_all_models, evaluate_model
    from classical_models import FEATURE_COLUMNS, TARGET_RAIN
"""

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

# ---------------------------------------------------------------------------
# Constants — columns available in data/clean_data.csv
# ---------------------------------------------------------------------------

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
    "Humidity_Avg",
    "Pressure_Diff",
    "Month_sin",
    "Month_cos",
    "WindDir3pm_sin",
    "WindDir3pm_cos",
    "RainToday",
    "City_Encoded",
]

TARGET_RAIN = "RainTomorrow"
TARGET_TEMP = "MaxTemp"

# Columns that are already encoded / cyclical — no scaling needed
_PASSTHROUGH_COLUMNS = [
    "Month_sin",
    "Month_cos",
    "WindDir3pm_sin",
    "WindDir3pm_cos",
    "RainToday",
    "City_Encoded",
]

# Numeric columns that need standardisation
_NUMERIC_COLUMNS = [c for c in FEATURE_COLUMNS if c not in _PASSTHROUGH_COLUMNS]

# Approximate class ratio (No / Yes ≈ 3.5) used for XGBoost's scale_pos_weight
_SCALE_POS_WEIGHT = 3.5


# ---------------------------------------------------------------------------
# Preprocessor
# ---------------------------------------------------------------------------

def build_preprocessor() -> ColumnTransformer:
    """Return a ColumnTransformer that scales numeric features and passes
    through already-encoded features."""
    return ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), _NUMERIC_COLUMNS),
            ("passthrough", "passthrough", _PASSTHROUGH_COLUMNS),
        ],
        remainder="drop",
    )


# ---------------------------------------------------------------------------
# Model builders — each returns a full sklearn Pipeline
# ---------------------------------------------------------------------------

def build_logistic_regression() -> Pipeline:
    """Logistic Regression pipeline with class-weight balancing."""
    return Pipeline(
        [
            ("preprocessor", build_preprocessor()),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    random_state=42,
                    solver="lbfgs",
                    C=1.0,
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


def build_xgboost() -> Pipeline:
    """XGBoost pipeline with scale_pos_weight for class imbalance."""
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
                    scale_pos_weight=_SCALE_POS_WEIGHT,
                    eval_metric="logloss",
                    random_state=42,
                    use_label_encoder=False,
                ),
            ),
        ]
    )


# ---------------------------------------------------------------------------
# Convenience accessors
# ---------------------------------------------------------------------------

def get_all_models() -> dict:
    """Return a dict of model-name → Pipeline for all 4 classical models."""
    return {
        "logistic_regression": build_logistic_regression(),
        "decision_tree": build_decision_tree(),
        "random_forest": build_random_forest(),
        "xgboost": build_xgboost(),
    }


# ---------------------------------------------------------------------------
# Evaluation utility (assumes model is already fitted)
# ---------------------------------------------------------------------------

def evaluate_model(model: Pipeline, X_test, y_test) -> dict:
    """Compute standard classification metrics on a fitted model.

    Parameters
    ----------
    model : a *fitted* sklearn Pipeline
    X_test : feature DataFrame / array
    y_test : true labels (0/1)

    Returns
    -------
    dict with accuracy, roc_auc, f1_score, precision, recall, confusion_matrix
    """
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    return {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "roc_auc": float(roc_auc_score(y_test, y_proba)),
        "f1_score": float(f1_score(y_test, y_pred, average="binary")),
        "precision": float(precision_score(y_test, y_pred)),
        "recall": float(recall_score(y_test, y_pred)),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }
