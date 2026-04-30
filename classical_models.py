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
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
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
import joblib
# ---------------------------------------------------------------------------
# Constants — columns available in weatherAUS_clean_features.csv
# ---------------------------------------------------------------------------

# Path to the clean feature-engineered dataset
DATA_PATH = "weatherAUS_clean_features.csv"

# The raw Location column (string) — preserved for filtering / display
LOCATION_COLUMN = "Location"

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

def get_all_models(load_saved=False) -> dict:
    """Return a dict of model-name → Pipeline for all 4 classical models."""
    if load_saved:
        return {
            "logistic_regression": joblib.load("saved_models/logistic_regression.joblib"),
            "decision_tree": joblib.load("saved_models/decision_tree.joblib"),
            "random_forest": joblib.load("saved_models/random_forest.joblib"),
            "xgboost": joblib.load("saved_models/xgboost.joblib"),
            "temp_regressor": joblib.load("saved_models/temp_regressor.joblib"),
        }
    else:
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


# ---------------------------------------------------------------------------
# Temperature Regressor
# ---------------------------------------------------------------------------

def build_temp_regressor() -> Pipeline:
    """Random Forest regressor for MaxTemp prediction."""
    return Pipeline(
        [
            ("preprocessor", build_preprocessor()),
            (
                "regressor",
                GradientBoostingRegressor(
                    n_estimators=200,
                    max_depth=5,
                    learning_rate=0.05,
                    subsample=0.8,
                    random_state=42,
                ),
            ),
        ]
    )


def evaluate_regressor(model: Pipeline, X_test, y_test) -> dict:
    """MAE and R² for a fitted regressor pipeline."""
    y_pred = model.predict(X_test)
    return {
        "mae": float(mean_absolute_error(y_test, y_pred)),
        "r2":  float(r2_score(y_test, y_pred)),
    }
