"""
Data Splitting — Temporal Train / Val / Calibration / Test
===========================================================

Provides utilities for:
- Chronological 4-way split (no future leakage)
- TimeSeriesSplit cross-validation builder
- Data loading with Date parsing and sorting

Usage:
    from data_splitting import load_and_split, build_temporal_cv
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

from classical_models import FEATURE_COLUMNS, TARGET_RAIN, LOCATION_COLUMN

# Default dataset path (v2 with Date column)
DATA_PATH_V2 = "weatherAUS_clean_features_v2.csv"
DATE_COLUMN = "Date"


# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------

def load_dataset(path=None, max_rows=None):
    """Load the feature-engineered dataset with Date column.

    Parameters
    ----------
    path : str or None
        Path to CSV. Defaults to ``DATA_PATH_V2``.
    max_rows : int or None
        If set, only load the first N rows (for smoke testing).

    Returns
    -------
    pd.DataFrame sorted by Date
    """
    path = path or DATA_PATH_V2
    df = pd.read_csv(path, parse_dates=[DATE_COLUMN], nrows=max_rows)
    df = df.sort_values(DATE_COLUMN).reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Temporal 4-Way Split
# ---------------------------------------------------------------------------

def temporal_train_val_cal_test_split(
    df,
    feature_columns=None,
    target_column=None,
    train_frac=0.60,
    val_frac=0.15,
    cal_frac=0.10,
):
    """Split data chronologically into train / val / calibration / test.

    Parameters
    ----------
    df : pd.DataFrame
        Must be sorted by Date and contain ``feature_columns`` and
        ``target_column``.
    feature_columns : list[str] or None
        Columns to use as features. Defaults to ``FEATURE_COLUMNS``.
    target_column : str or None
        Column name for the target. Defaults to ``TARGET_RAIN``.
    train_frac, val_frac, cal_frac : float
        Proportions for train, validation, and calibration sets.
        Test set gets the remaining fraction ``1 - train - val - cal``.

    Returns
    -------
    dict with keys:
        X_train, X_val, X_cal, X_test,
        y_train, y_val, y_cal, y_test,
        dates_train, dates_val, dates_cal, dates_test,
        split_info  (dict of split metadata)
    """
    feature_columns = feature_columns or FEATURE_COLUMNS
    target_column = target_column or TARGET_RAIN

    # Ensure sorted by date
    df = df.sort_values(DATE_COLUMN).reset_index(drop=True)
    n = len(df)

    # Compute cut indices
    i_train = int(n * train_frac)
    i_val = int(n * (train_frac + val_frac))
    i_cal = int(n * (train_frac + val_frac + cal_frac))

    # Split
    df_train = df.iloc[:i_train]
    df_val = df.iloc[i_train:i_val]
    df_cal = df.iloc[i_val:i_cal]
    df_test = df.iloc[i_cal:]

    # Extract X, y, dates
    result = {}
    for name, subset in [
        ("train", df_train),
        ("val", df_val),
        ("cal", df_cal),
        ("test", df_test),
    ]:
        result[f"X_{name}"] = subset[feature_columns].reset_index(drop=True)
        result[f"y_{name}"] = subset[target_column].reset_index(drop=True)
        result[f"dates_{name}"] = subset[DATE_COLUMN].reset_index(drop=True)

    # Split metadata
    test_frac = 1.0 - train_frac - val_frac - cal_frac
    result["split_info"] = {
        "n_total": n,
        "n_train": len(df_train),
        "n_val": len(df_val),
        "n_cal": len(df_cal),
        "n_test": len(df_test),
        "date_train": f"{df_train[DATE_COLUMN].min()} → {df_train[DATE_COLUMN].max()}",
        "date_val": f"{df_val[DATE_COLUMN].min()} → {df_val[DATE_COLUMN].max()}",
        "date_cal": f"{df_cal[DATE_COLUMN].min()} → {df_cal[DATE_COLUMN].max()}",
        "date_test": f"{df_test[DATE_COLUMN].min()} → {df_test[DATE_COLUMN].max()}",
        "rain_pct_train": f"{df_train[target_column].mean():.2%}",
        "rain_pct_val": f"{df_val[target_column].mean():.2%}",
        "rain_pct_cal": f"{df_cal[target_column].mean():.2%}",
        "rain_pct_test": f"{df_test[target_column].mean():.2%}",
        "fractions": f"train={train_frac:.0%} val={val_frac:.0%} cal={cal_frac:.0%} test={test_frac:.0%}",
    }

    return result


# ---------------------------------------------------------------------------
# Temporal Cross-Validation Builder
# ---------------------------------------------------------------------------

def build_temporal_cv(n_splits=5):
    """Return a TimeSeriesSplit cross-validator.

    Parameters
    ----------
    n_splits : int
        Number of CV splits.

    Returns
    -------
    TimeSeriesSplit
    """
    return TimeSeriesSplit(n_splits=n_splits)


# ---------------------------------------------------------------------------
# Convenience: load + split in one call
# ---------------------------------------------------------------------------

def load_and_split(path=None, max_rows=None, **split_kwargs):
    """Load dataset and perform temporal split in one call.

    Parameters
    ----------
    path : str or None
        CSV path.
    max_rows : int or None
        Limit rows for smoke testing.
    **split_kwargs
        Forwarded to ``temporal_train_val_cal_test_split``.

    Returns
    -------
    dict (same as ``temporal_train_val_cal_test_split``)
    """
    df = load_dataset(path=path, max_rows=max_rows)
    return temporal_train_val_cal_test_split(df, **split_kwargs)


def print_split_info(split):
    """Pretty-print the split metadata."""
    info = split["split_info"]
    print(f"\n{'='*60}")
    print(f"  Temporal Data Split ({info['fractions']})")
    print(f"{'='*60}")
    print(f"  Total rows : {info['n_total']:,}")
    print(f"  Train      : {info['n_train']:>7,}  ({info['date_train']})  rain={info['rain_pct_train']}")
    print(f"  Validation : {info['n_val']:>7,}  ({info['date_val']})  rain={info['rain_pct_val']}")
    print(f"  Calibration: {info['n_cal']:>7,}  ({info['date_cal']})  rain={info['rain_pct_cal']}")
    print(f"  Test       : {info['n_test']:>7,}  ({info['date_test']})  rain={info['rain_pct_test']}")
    print(f"{'='*60}\n")
