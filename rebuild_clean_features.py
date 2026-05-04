#!/usr/bin/env python
"""
Rebuild Clean Features — Add Date Column
==========================================

Reads the raw ``data/weatherAUS.csv`` and reproduces the exact same feature-
engineering pipeline as ``rain_australia_analysis.ipynb``, **but preserves the
Date column** in the output so that temporal splits are possible.

Outputs ``weatherAUS_clean_features_v2.csv`` at the project root.

Usage:
    python rebuild_clean_features.py
"""

import numpy as np
import pandas as pd

RAW_PATH = "data/weatherAUS.csv"
OUTPUT_PATH = "weatherAUS_clean_features_v2.csv"

# ── Wind direction mapping ────────────────────────────────────────────────────
WD_MAP = {
    "N": 0, "NNE": 22.5, "NE": 45, "ENE": 67.5,
    "E": 90, "ESE": 112.5, "SE": 135, "SSE": 157.5,
    "S": 180, "SSW": 202.5, "SW": 225, "WSW": 247.5,
    "W": 270, "WNW": 292.5, "NW": 315, "NNW": 337.5,
}


def main():
    print("Loading raw data …")
    df = pd.read_csv(RAW_PATH)
    print(f"  Raw rows: {len(df)}")

    # ── 1. Drop high-missing columns (>30%) ──────────────────────────────────
    pct_missing = (df.isnull().sum() / len(df)) * 100
    cols_to_drop = pct_missing[pct_missing > 30].index.tolist()
    print(f"  Dropping columns with >30% missing: {cols_to_drop}")
    df = df.drop(columns=cols_to_drop)

    # ── 2. Impute missing values ─────────────────────────────────────────────
    num_cols = df.select_dtypes(include=["float64", "int64"]).columns
    for col in num_cols:
        df[col] = df[col].fillna(df[col].median())
    cat_cols = df.select_dtypes(include=["object"]).columns
    for col in cat_cols:
        if col != "Date":
            df[col] = df[col].fillna(df[col].mode()[0])

    # ── 3. Parse Date ────────────────────────────────────────────────────────
    df["Date"] = pd.to_datetime(df["Date"])

    # ── 4. Feature Engineering ───────────────────────────────────────────────
    df["Month"] = df["Date"].dt.month
    df["DayOfYear"] = df["Date"].dt.dayofyear
    df["WeekOfYear"] = df["Date"].dt.isocalendar().week.astype(int)
    df["Quarter"] = df["Date"].dt.quarter

    # Cyclical month
    df["Month_sin"] = np.sin(2 * np.pi * df["Month"] / 12)
    df["Month_cos"] = np.cos(2 * np.pi * df["Month"] / 12)

    # Cyclical day
    df["Day_sin"] = np.sin(2 * np.pi * df["DayOfYear"] / 365.25)
    df["Day_cos"] = np.cos(2 * np.pi * df["DayOfYear"] / 365.25)

    # Temperature features
    df["TempRange"] = df["MaxTemp"] - df["MinTemp"]
    df["TempMean"] = (df["MaxTemp"] + df["MinTemp"]) / 2
    df["TempDiff_9_3"] = df["Temp3pm"] - df["Temp9am"]

    # Wind chill (simplified)
    df["WindChill9am"] = df["Temp9am"] - 0.5 * df["WindSpeed9am"]
    df["WindChill3pm"] = df["Temp3pm"] - 0.5 * df["WindSpeed3pm"]

    # Humidity features
    df["HumidityChange"] = df["Humidity3pm"] - df["Humidity9am"]
    df["HumidityMean"] = (df["Humidity9am"] + df["Humidity3pm"]) / 2

    # Pressure features
    df["PressureChange"] = df["Pressure3pm"] - df["Pressure9am"]
    df["PressureMean"] = (df["Pressure9am"] + df["Pressure3pm"]) / 2

    # Dew point approximation
    df["DewPoint9am"] = df["Temp9am"] - ((100 - df["Humidity9am"]) / 5)
    df["DewPoint3pm"] = df["Temp3pm"] - ((100 - df["Humidity3pm"]) / 5)

    # Heat index approximation
    df["HeatIndex9am"] = df["Temp9am"] + 0.5 * (df["Humidity9am"] - 50)
    df["HeatIndex3pm"] = df["Temp3pm"] + 0.5 * (df["Humidity3pm"] - 50)

    # Wind direction encoding (gust, 9am, 3pm)
    for wdir_col, prefix in [
        ("WindGustDir", "WindDirGust"),
        ("WindDir9am", "WindDir9am"),
        ("WindDir3pm", "WindDir3pm"),
    ]:
        if wdir_col in df.columns:
            deg = df[wdir_col].map(WD_MAP)
            df[f"{prefix}_sin"] = np.sin(np.deg2rad(deg))
            df[f"{prefix}_cos"] = np.cos(np.deg2rad(deg))
            df.drop(columns=[wdir_col], inplace=True)

    # Wind speed features
    df["WindSpeedChange"] = df["WindSpeed3pm"] - df["WindSpeed9am"]
    df["WindSpeedMean"] = (df["WindSpeed9am"] + df["WindSpeed3pm"]) / 2

    # Storm flag
    df["StormFlag"] = (
        (df["WindGustSpeed"] > 60)
        & (df["Humidity3pm"] > 70)
        & (df["PressureChange"] < -2)
    ).astype(int)

    # Binary RainToday
    df["RainToday_bin"] = df["RainToday"].map({"No": 0, "Yes": 1})

    # Rain risk score
    df["RainRiskScore"] = (
        df["Humidity3pm"] * 0.4
        + df["Rainfall"] * 0.3
        + (100 - df["Pressure3pm"] / 10) * 0.3
    )

    # Temperature-humidity interaction
    df["TempHumidity_interact"] = df["Temp3pm"] * df["Humidity3pm"] / 100

    # Lag features (sort by Location then Date first)
    df = df.sort_values(["Location", "Date"]).reset_index(drop=True)

    for col_name, lags in [
        ("Rainfall", [1, 2, 7]),
        ("Humidity3pm", [1, 2, 7]),
        ("MaxTemp", [1, 2, 7]),
        ("MinTemp", [1, 2, 7]),
    ]:
        for lag in lags:
            df[f"{col_name}_lag{lag}"] = df.groupby("Location")[col_name].shift(lag)
        df[f"{col_name}_roll3"] = (
            df.groupby("Location")[col_name]
            .transform(lambda x: x.rolling(3, min_periods=1).mean())
        )
        df[f"{col_name}_roll7"] = (
            df.groupby("Location")[col_name]
            .transform(lambda x: x.rolling(7, min_periods=1).mean())
        )

    # Season dummies
    season_map = {12: "Summer", 1: "Summer", 2: "Summer",
                  3: "Autumn",  4: "Autumn",  5: "Autumn",
                  6: "Winter",  7: "Winter",  8: "Winter",
                  9: "Spring", 10: "Spring", 11: "Spring"}
    df["Season"] = df["Month"].map(season_map)
    season_dummies = pd.get_dummies(df["Season"], prefix="Season")
    # Keep only 3 dummies (drop Autumn as reference)
    for s in ["Spring", "Summer", "Winter"]:
        col = f"Season_{s}"
        if col in season_dummies.columns:
            df[col] = season_dummies[col].astype(int)
    df.drop(columns=["Season"], inplace=True)

    # RainToday_Yes
    df["RainToday_Yes"] = df["RainToday_bin"]

    # Location rain rate (target encoding — frequency of rain per location)
    rain_rate = df.groupby("Location")["RainToday_bin"].mean()
    df["Location_rainrate"] = df["Location"].map(rain_rate)

    # Target encoding
    df["RainTomorrow_enc"] = df["RainTomorrow"].map({"No": 0, "Yes": 1})

    # ── 5. Drop original categorical columns we've already encoded ───────────
    cols_to_drop_final = ["RainToday", "RainTomorrow"]
    df.drop(columns=[c for c in cols_to_drop_final if c in df.columns], inplace=True)

    # ── 6. Drop rows with NaN from lag features ──────────────────────────────
    before = len(df)
    df = df.dropna().reset_index(drop=True)
    print(f"  Dropped {before - len(df)} rows with NaN (lag features)")

    # ── 7. Sort by Date globally for temporal split ──────────────────────────
    df = df.sort_values("Date").reset_index(drop=True)

    # ── 8. Save ──────────────────────────────────────────────────────────────
    print(f"  Final shape: {df.shape}")
    print(f"  Date range: {df['Date'].min()} → {df['Date'].max()}")
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"✅ Saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
