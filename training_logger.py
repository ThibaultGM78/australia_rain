"""
Training Logger — Structured Logging for Training & Fine-Tuning Metrics
=========================================================================

Provides utilities for:
- Python logging with file + console output
- Persisting training metrics to CSV files in logs/
- Logging cross-validation results, tuning results, calibration results
- **Separate** classification vs regression metric logging
- Train-vs-validation gap logging
- CSV deduplication for clean outputs

Usage:
    from training_logger import setup_logger, log_training_metrics, log_cv_results
"""

import csv
import logging
import os
from datetime import datetime

import numpy as np


# ---------------------------------------------------------------------------
# Logger setup
# ---------------------------------------------------------------------------

def setup_logger(name="training", log_dir="logs", level="INFO"):
    """Configure and return a logger that writes to both console and file.

    Parameters
    ----------
    name : str
        Logger name (also used for the log filename)
    log_dir : str
        Directory for log files (created if it doesn't exist)
    level : str or int
        Logging level for the console handler.
        Accepts standard level names: "DEBUG", "INFO", "WARNING", "ERROR"
        or their integer equivalents. The file handler always logs DEBUG.
        Default: "INFO"

    Returns
    -------
    logging.Logger
    """
    os.makedirs(log_dir, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # Avoid duplicate handlers on repeated calls
    if logger.handlers:
        # If logger already exists, update the console handler level in case
        # the caller changed it (e.g. re-running with a different --log-level)
        numeric_level = _parse_level(level)
        for handler in logger.handlers:
            if isinstance(handler, logging.StreamHandler) and not isinstance(
                handler, logging.FileHandler
            ):
                handler.setLevel(numeric_level)
        return logger

    numeric_level = _parse_level(level)

    # File handler — always DEBUG so every detail is captured on disk
    fh = logging.FileHandler(
        os.path.join(log_dir, f"{name}.log"),
        mode="a",
        encoding="utf-8",
    )
    fh.setLevel(logging.DEBUG)
    file_fmt = logging.Formatter(
        "%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    fh.setFormatter(file_fmt)

    # Console handler — level controlled by caller
    ch = logging.StreamHandler()
    ch.setLevel(numeric_level)
    console_fmt = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    ch.setFormatter(console_fmt)

    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger


def _parse_level(level):
    """Convert a level name string or int to a logging int constant.

    Parameters
    ----------
    level : str or int
        e.g. "DEBUG", "INFO", "WARNING", "ERROR", or 10, 20, 30, 40

    Returns
    -------
    int
    """
    if isinstance(level, int):
        return level
    numeric = getattr(logging, level.upper(), None)
    if numeric is None:
        raise ValueError(
            f"Invalid log level: {level!r}. "
            "Choose from DEBUG, INFO, WARNING, ERROR."
        )
    return numeric


# ---------------------------------------------------------------------------
# CSV persistence helpers
# ---------------------------------------------------------------------------

def _csv_path(log_dir, filename):
    """Return full path for a metrics CSV file."""
    os.makedirs(log_dir, exist_ok=True)
    return os.path.join(log_dir, filename)


def _timestamp():
    """Return current ISO timestamp."""
    return datetime.now().isoformat()


def _append_row_to_csv(filepath, row_dict):
    """Append a dict as a row to a CSV file, creating headers if needed."""
    file_exists = os.path.exists(filepath)

    with open(filepath, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row_dict.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row_dict)


# ---------------------------------------------------------------------------
# Classification metrics logging
# ---------------------------------------------------------------------------

def log_training_metrics(logger, model_name, metrics_dict, phase="training",
                         log_dir="logs"):
    """Log classification training/evaluation metrics and persist to CSV.

    Parameters
    ----------
    logger : logging.Logger
    model_name : str
    metrics_dict : dict — e.g. {"accuracy": 0.85, "roc_auc": 0.87, "mcc": 0.55, ...}
    phase : str — "training", "evaluation", "test", etc.
    log_dir : str
    """
    logger.info(f"[{phase.upper()}] {model_name} (classification) metrics:")
    for key, value in metrics_dict.items():
        if key == "confusion_matrix":
            logger.info(f"  {key}: {value}")
        elif isinstance(value, float):
            logger.info(f"  {key}: {value:.4f}")
        else:
            logger.info(f"  {key}: {value}")

    row = {
        "timestamp": _timestamp(),
        "phase": phase,
        "model_type": "classification",
        "model": model_name,
    }
    for key, value in metrics_dict.items():
        if key == "confusion_matrix":
            row[key] = str(value)
        else:
            row[key] = value

    _append_row_to_csv(_csv_path(log_dir, "training_metrics.csv"), row)


# ---------------------------------------------------------------------------
# Regression metrics logging (separate from classification!)
# ---------------------------------------------------------------------------

def log_regression_metrics(logger, model_name, metrics_dict, phase="evaluation",
                           log_dir="logs"):
    """Log regression metrics and persist to a **separate** CSV.

    This prevents R² from being mixed up with roc_auc in the same file.

    Parameters
    ----------
    logger : logging.Logger
    model_name : str
    metrics_dict : dict — e.g. {"mae": 0.44, "rmse": 0.65, "r2": 0.99}
    phase : str
    log_dir : str
    """
    logger.info(f"[{phase.upper()}] {model_name} (regression) metrics:")
    for key, value in metrics_dict.items():
        if isinstance(value, float):
            logger.info(f"  {key}: {value:.4f}")
        else:
            logger.info(f"  {key}: {value}")

    row = {
        "timestamp": _timestamp(),
        "phase": phase,
        "model_type": "regression",
        "model": model_name,
    }
    row.update(metrics_dict)

    _append_row_to_csv(_csv_path(log_dir, "regression_metrics.csv"), row)


# ---------------------------------------------------------------------------
# Cross-validation results logging
# ---------------------------------------------------------------------------

def log_cv_results(logger, model_name, cv_scores, log_dir="logs"):
    """Log cross-validation results and persist to CSV.

    Now also logs **train scores** when available for overfitting monitoring.

    Parameters
    ----------
    logger : logging.Logger
    model_name : str
    cv_scores : dict — output of sklearn cross_validate
    log_dir : str
    """
    logger.info(f"[CV] {model_name} cross-validation results:")

    row = {
        "timestamp": _timestamp(),
        "phase": "cross_validation",
        "model": model_name,
    }

    for key in cv_scores:
        if key.startswith("test_"):
            metric_name = key.replace("test_", "")
            mean_val = float(cv_scores[key].mean())
            std_val = float(cv_scores[key].std())
            row[f"{metric_name}_mean"] = round(mean_val, 6)
            row[f"{metric_name}_std"] = round(std_val, 6)
            logger.info(f"  val_{metric_name}: {mean_val:.4f} ± {std_val:.4f}")

    # Also log train scores if available (for overfitting monitoring)
    for key in cv_scores:
        if key.startswith("train_"):
            metric_name = key.replace("train_", "")
            mean_val = float(cv_scores[key].mean())
            std_val = float(cv_scores[key].std())
            row[f"train_{metric_name}_mean"] = round(mean_val, 6)
            row[f"train_{metric_name}_std"] = round(std_val, 6)
            logger.info(f"  train_{metric_name}: {mean_val:.4f} ± {std_val:.4f}")

            # Compute gap
            val_mean_key = f"{metric_name}_mean"
            if val_mean_key in row:
                gap = mean_val - row[val_mean_key]
                row[f"{metric_name}_gap"] = round(gap, 6)
                risk = "HIGH" if gap > 0.05 else "MODERATE" if gap > 0.02 else "LOW"
                logger.info(f"  {metric_name} gap (train-val): {gap:.4f} → overfit risk: {risk}")

    _append_row_to_csv(_csv_path(log_dir, "cv_results.csv"), row)


# ---------------------------------------------------------------------------
# Train vs Validation comparison logging
# ---------------------------------------------------------------------------

def log_train_val_comparison(logger, model_name, train_scores, val_scores,
                              metric_name="roc_auc", log_dir="logs"):
    """Log train vs validation score comparison for overfitting monitoring.

    Parameters
    ----------
    logger : logging.Logger
    model_name : str
    train_scores : array-like — scores from training folds
    val_scores : array-like — scores from validation folds
    metric_name : str
    log_dir : str
    """
    train_mean = float(np.mean(train_scores))
    val_mean = float(np.mean(val_scores))
    gap = train_mean - val_mean

    risk = "HIGH" if gap > 0.05 else "MODERATE" if gap > 0.02 else "LOW"

    logger.info(f"[OVERFIT CHECK] {model_name}:")
    logger.info(f"  Train {metric_name}: {train_mean:.4f}")
    logger.info(f"  Val   {metric_name}: {val_mean:.4f}")
    logger.info(f"  Gap: {gap:.4f} → risk: {risk}")

    row = {
        "timestamp": _timestamp(),
        "phase": "overfit_check",
        "model": model_name,
        "metric": metric_name,
        "train_score": round(train_mean, 6),
        "val_score": round(val_mean, 6),
        "gap": round(gap, 6),
        "risk": risk,
    }
    _append_row_to_csv(_csv_path(log_dir, "train_val_comparison.csv"), row)


# ---------------------------------------------------------------------------
# Hyperparameter tuning results logging
# ---------------------------------------------------------------------------

def log_tuning_results(logger, model_name, method, best_params, best_score,
                       log_dir="logs"):
    """Log hyperparameter tuning results and persist to CSV.

    Parameters
    ----------
    logger : logging.Logger
    model_name : str
    method : str — "grid_search", "random_search", "optuna"
    best_params : dict
    best_score : float
    log_dir : str
    """
    logger.info(f"[TUNING] {model_name} ({method}):")
    logger.info(f"  Best score: {best_score:.4f}")
    logger.info(f"  Best params: {best_params}")

    row = {
        "timestamp": _timestamp(),
        "phase": "tuning",
        "method": method,
        "model": model_name,
        "best_score": round(float(best_score), 6),
        "best_params": str(best_params),
    }
    _append_row_to_csv(_csv_path(log_dir, "tuning_results.csv"), row)


# ---------------------------------------------------------------------------
# Calibration results logging
# ---------------------------------------------------------------------------

def log_calibration_results(logger, model_name, raw_vs_calibrated,
                            log_dir="logs"):
    """Log raw vs calibrated model comparison and persist to CSV.

    Parameters
    ----------
    logger : logging.Logger
    model_name : str
    raw_vs_calibrated : dict — output of compare_calibration()
    log_dir : str
    """
    logger.info(f"[CALIBRATION] {model_name} — raw vs calibrated:")

    for variant, metrics in raw_vs_calibrated.items():
        logger.info(f"  {variant}:")
        row = {
            "timestamp": _timestamp(),
            "phase": "calibration",
            "model": model_name,
            "variant": variant,
        }
        for key, value in metrics.items():
            row[key] = round(float(value), 6)
            logger.info(f"    {key}: {value:.4f}")

        _append_row_to_csv(_csv_path(log_dir, "calibration_results.csv"), row)


# ---------------------------------------------------------------------------
# Model save logging
# ---------------------------------------------------------------------------

def log_model_save(logger, model_name, path, log_dir="logs"):
    """Log model serialization event.

    Parameters
    ----------
    logger : logging.Logger
    model_name : str
    path : str — path where model was saved
    log_dir : str
    """
    logger.info(f"[SAVE] {model_name} → {path}")

    row = {
        "timestamp": _timestamp(),
        "phase": "model_save",
        "model": model_name,
        "path": path,
    }
    # Save to its own CSV instead of mixing with training metrics
    _append_row_to_csv(_csv_path(log_dir, "model_saves.csv"), row)


# ---------------------------------------------------------------------------
# CSV deduplication
# ---------------------------------------------------------------------------

def clean_csv_duplicates(filepath, key_columns=None):
    """Deduplicate a CSV file, keeping the latest entry per key.

    Parameters
    ----------
    filepath : str
        Path to the CSV file.
    key_columns : list[str] or None
        Columns that define a unique row. If None, defaults to
        ["model", "phase"].
    """
    import pandas as pd

    if not os.path.exists(filepath):
        return

    try:
        df = pd.read_csv(filepath, on_bad_lines="skip")
    except Exception:
        # If the CSV is too corrupt to read, skip it
        return

    if df.empty:
        return

    if key_columns is None:
        key_columns = [c for c in ["model", "phase"] if c in df.columns]

    if not key_columns:
        return

    before = len(df)
    df = df.drop_duplicates(subset=key_columns, keep="last")
    after = len(df)

    if before != after:
        df.to_csv(filepath, index=False)
        print(f"  Cleaned {filepath}: {before} → {after} rows (removed {before - after} duplicates)")


def clean_all_csvs(log_dir="logs"):
    """Clean all CSV files in the log directory."""
    import glob

    for path in glob.glob(os.path.join(log_dir, "*.csv")):
        clean_csv_duplicates(path)