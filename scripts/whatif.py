"""
whatif_analysis.py
What-if scenario analysis for the rain prediction XGBoost model.

Techniques:
  1. Partial Dependence Plots (PDP)  — how P(rain) evolves as one feature varies
  2. 2D sensitivity heatmap          — temp × humidity joint effect
  3. Tornado chart                   — which feature moves P(rain) the most (±1 std)
  4. ICE curves                      — per-sample variation (Individual Cond. Expectation)
  5. Interactive scenario function   — predict_scenario(MaxTemp=35, Humidity3pm=90)
  6. Dashboard                       — all panels in one figure

Usage:
  python whatif_analysis.py                     # runs all analyses
  python whatif_analysis.py --interactive       # launches the CLI what-if tool

Requirements:
  pip install pandas numpy matplotlib scikit-learn joblib
"""

import argparse
import os
import warnings

import joblib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from sklearn.inspection import partial_dependence

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────────
MODEL_PATH  = "saved_models/xgboost.joblib"
DATA_PATH   = None          # optional: path to your original CSV for ICE/tornado
OUTPUT_DIR  = "."

# ── Style ──────────────────────────────────────────────────────────────────────
BLUE   = "#378ADD"
TEAL   = "#1D9E75"
CORAL  = "#D85A30"
AMBER  = "#BA7517"
PURPLE = "#7F77DD"
GRAY   = "#888780"
LIGHT  = "#E6F1FB"

plt.rcParams.update({
    "font.family":       "sans-serif",
    "font.size":         11,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.color":        "#e0e0e0",
    "grid.linewidth":    0.6,
    "figure.dpi":        130,
})

# ── Australian weather feature catalogue ──────────────────────────────────────
# Adjust ranges if your preprocessing differs.
FEATURE_CATALOG = {
    "MinTemp":        {"range": (-10, 45),  "unit": "°C",   "label": "Min temp"},
    "MaxTemp":        {"range": (-5,  50),  "unit": "°C",   "label": "Max temp"},
    "Rainfall":       {"range": (0,   100), "unit": "mm",   "label": "Rainfall"},
    "Evaporation":    {"range": (0,   30),  "unit": "mm",   "label": "Evaporation"},
    "Sunshine":       {"range": (0,   14),  "unit": "h",    "label": "Sunshine"},
    "WindGustSpeed":  {"range": (0,   140), "unit": "km/h", "label": "Wind gust speed"},
    "WindSpeed9am":   {"range": (0,   80),  "unit": "km/h", "label": "Wind speed 9am"},
    "WindSpeed3pm":   {"range": (0,   80),  "unit": "km/h", "label": "Wind speed 3pm"},
    "Humidity9am":    {"range": (0,   100), "unit": "%",    "label": "Humidity 9am"},
    "Humidity3pm":    {"range": (0,   100), "unit": "%",    "label": "Humidity 3pm"},
    "Pressure9am":    {"range": (980, 1040),"unit": "hPa",  "label": "Pressure 9am"},
    "Pressure3pm":    {"range": (980, 1040),"unit": "hPa",  "label": "Pressure 3pm"},
    "Cloud9am":       {"range": (0,   8),   "unit": "oktas","label": "Cloud 9am"},
    "Cloud3pm":       {"range": (0,   8),   "unit": "oktas","label": "Cloud 3pm"},
    "Temp9am":        {"range": (-5,  45),  "unit": "°C",   "label": "Temp 9am"},
    "Temp3pm":        {"range": (-5,  50),  "unit": "°C",   "label": "Temp 3pm"},
}

# Default baseline (Australian median-ish values for a typical day)
BASELINE = {
    "MinTemp":       12.0,
    "MaxTemp":       23.0,
    "Rainfall":       0.0,
    "Evaporation":    4.8,
    "Sunshine":       8.0,
    "WindGustSpeed": 39.0,
    "WindSpeed9am":  14.0,
    "WindSpeed3pm":  19.0,
    "Humidity9am":   69.0,
    "Humidity3pm":   51.0,
    "Pressure9am":  1017.6,
    "Pressure3pm":  1015.3,
    "Cloud9am":       4.0,
    "Cloud3pm":       4.0,
    "Temp9am":       16.9,
    "Temp3pm":       21.7,
}


# ── Model helpers ──────────────────────────────────────────────────────────────
def load_model(path=MODEL_PATH):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Model not found at '{path}'.\n"
            "Make sure you run the training script first and that\n"
            "MODEL_PATH points to the saved .joblib file."
        )
    model = joblib.load(path)
    print(f"  Model loaded from: {path}")
    return model


def get_feature_names(model):
    """Extract feature names from a sklearn Pipeline or raw estimator."""
    # sklearn Pipeline: look at the last preprocessor step
    if hasattr(model, "named_steps"):
        for step_name, step in model.named_steps.items():
            if hasattr(step, "get_feature_names_out"):
                return list(step.get_feature_names_out())
            if hasattr(step, "feature_names_in_"):
                return list(step.feature_names_in_)
    if hasattr(model, "feature_names_in_"):
        return list(model.feature_names_in_)
    return None


def make_baseline_df(model, feature_names=None):
    """
    Build a one-row DataFrame at baseline values.
    Falls back to zeros for features not in BASELINE.
    """
    if feature_names is None:
        feature_names = get_feature_names(model) or list(BASELINE.keys())

    row = {}
    for f in feature_names:
        # strip pipeline prefixes like "num__MaxTemp" → "MaxTemp"
        key = f.split("__")[-1] if "__" in f else f
        row[f] = BASELINE.get(key, 0.0)
    return pd.DataFrame([row])


def predict_proba(model, df):
    """Return P(rain=1) for each row in df."""
    return model.predict_proba(df)[:, 1]


# ── Scenario function (public API) ────────────────────────────────────────────
def predict_scenario(model, feature_names=None, verbose=True, **kwargs):
    """
    Predict rain probability for a custom scenario.

    Usage:
        predict_scenario(model, MaxTemp=35, Humidity3pm=90, Pressure3pm=1005)

    All unspecified features take their baseline values.
    Returns P(rain) as a float.
    """
    baseline_df = make_baseline_df(model, feature_names)
    if feature_names is None:
        feature_names = list(baseline_df.columns)

    for key, val in kwargs.items():
        # handle both "MaxTemp" and "num__MaxTemp" style column names
        matched = [f for f in feature_names if f == key or f.endswith("__" + key)]
        if not matched:
            print(f"  WARNING: feature '{key}' not found in model. Skipped.")
            continue
        baseline_df.loc[0, matched[0]] = val

    prob = float(predict_proba(model, baseline_df)[0])

    if verbose:
        changed = {k: v for k, v in kwargs.items()}
        print(f"\n  Scenario: {changed}")
        print(f"  P(rain)  = {prob:.1%}  ({'RAIN' if prob >= 0.5 else 'NO RAIN'})")
    return prob


# ── Figure 1 — Partial Dependence Plots ───────────────────────────────────────
def plot_pdp(model, feature_names=None, save=True):
    """
    PDP for temperature and humidity features: how P(rain) varies
    as each feature sweeps its full range, others held at baseline.
    """
    if feature_names is None:
        feature_names = get_feature_names(model) or list(BASELINE.keys())

    targets = ["MaxTemp", "Humidity3pm", "MinTemp", "Humidity9am",
               "Pressure3pm", "Cloud3pm"]
    # map short names → actual column names
    col_map = {f.split("__")[-1]: f for f in feature_names}
    found = [(t, col_map[t]) for t in targets if t in col_map]

    if not found:
        print("  WARNING: none of the target PDP features found in model. Skipping fig1.")
        return None

    baseline_df = make_baseline_df(model, feature_names)
    n = len(found)
    ncols = 3
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 4 * nrows))
    axes = np.array(axes).flatten()

    for i, (short_name, col_name) in enumerate(found):
        ax = axes[i]
        info = FEATURE_CATALOG.get(short_name, {})
        lo, hi = info.get("range", (0, 100))
        unit    = info.get("unit", "")
        label   = info.get("label", short_name)

        grid = np.linspace(lo, hi, 80)
        probs = []
        for val in grid:
            row = baseline_df.copy()
            row.loc[0, col_name] = val
            probs.append(float(predict_proba(model, row)[0]))

        probs = np.array(probs)

        # Shade under curve — colour by probability level
        ax.fill_between(grid, probs, alpha=0.15, color=BLUE)
        ax.plot(grid, probs, color=BLUE, lw=2)

        # Baseline marker
        base_val = baseline_df.loc[0, col_name]
        base_prob = float(predict_proba(model, baseline_df)[0])
        ax.axvline(base_val, color=GRAY, lw=1, ls="--", alpha=0.7, label=f"Baseline: {base_val:.0f}{unit}")
        ax.axhline(base_prob, color=GRAY, lw=0.8, ls=":", alpha=0.5)
        ax.scatter([base_val], [base_prob], color=CORAL, zorder=5, s=50)

        ax.set_xlabel(f"{label} ({unit})" if unit else label)
        ax.set_ylabel("P(rain)" if i % ncols == 0 else "")
        ax.set_ylim(0, 1)
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
        ax.legend(fontsize=9, frameon=False)

        # Direction annotation
        trend = "↑ as increases" if probs[-1] > probs[0] else "↓ as increases"
        ax.set_title(f"{label}  —  P(rain) {trend}", fontsize=11, pad=6)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Partial Dependence Plots — P(rain) vs individual features",
                 fontsize=13, y=1.01)
    fig.tight_layout()
    if save:
        path = os.path.join(OUTPUT_DIR, "fig1_pdp.png")
        fig.savefig(path, bbox_inches="tight")
        print(f"  Saved → {path}")
    return fig


# ── Figure 2 — 2D Heatmap: MaxTemp × Humidity3pm ─────────────────────────────
def plot_heatmap_2d(model, feature_names=None,
                    feat_x="MaxTemp", feat_y="Humidity3pm",
                    n_points=40, save=True):
    """
    Grid of P(rain) values as two features vary simultaneously.
    All other features stay at baseline.
    """
    if feature_names is None:
        feature_names = get_feature_names(model) or list(BASELINE.keys())
    col_map = {f.split("__")[-1]: f for f in feature_names}

    if feat_x not in col_map or feat_y not in col_map:
        print(f"  WARNING: {feat_x} or {feat_y} not in model features. Skipping fig2.")
        return None

    baseline_df = make_baseline_df(model, feature_names)
    col_x, col_y = col_map[feat_x], col_map[feat_y]

    info_x = FEATURE_CATALOG.get(feat_x, {"range": (0, 50), "unit": "°C", "label": feat_x})
    info_y = FEATURE_CATALOG.get(feat_y, {"range": (0, 100), "unit": "%", "label": feat_y})

    x_vals = np.linspace(*info_x["range"], n_points)
    y_vals = np.linspace(*info_y["range"], n_points)

    # Build all combinations efficiently
    rows = []
    for y in y_vals:
        for x in x_vals:
            row = baseline_df.copy()
            row.loc[0, col_x] = x
            row.loc[0, col_y] = y
            rows.append(row)
    big_df = pd.concat(rows, ignore_index=True)
    probs = predict_proba(model, big_df).reshape(n_points, n_points)

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.pcolormesh(x_vals, y_vals, probs,
                       cmap="RdYlBu_r", vmin=0, vmax=1, shading="auto")
    cbar = fig.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label("P(rain)", fontsize=11)
    cbar.ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))

    # 50 % decision boundary
    ax.contour(x_vals, y_vals, probs, levels=[0.5],
               colors="white", linewidths=1.5, linestyles="--")
    ax.text(x_vals[-1] * 0.98, info_y["range"][1] * 0.96,
            "50% boundary", color="white", fontsize=9,
            ha="right", va="top")

    # Baseline cross
    bx = baseline_df.loc[0, col_x]
    by = baseline_df.loc[0, col_y]
    ax.scatter([bx], [by], marker="+", s=200, color="white", lw=2,
               zorder=5, label=f"Baseline ({bx:.0f}, {by:.0f})")
    ax.legend(fontsize=9, frameon=False, loc="upper left",
              labelcolor="white")

    ax.set_xlabel(f"{info_x['label']} ({info_x['unit']})")
    ax.set_ylabel(f"{info_y['label']} ({info_y['unit']})")
    ax.set_title(f"P(rain) — {info_x['label']} × {info_y['label']}",
                 fontsize=13, pad=10)

    fig.tight_layout()
    if save:
        path = os.path.join(OUTPUT_DIR, "fig2_heatmap_2d.png")
        fig.savefig(path, bbox_inches="tight")
        print(f"  Saved → {path}")
    return fig


# ── Figure 3 — Tornado chart: feature sensitivity ────────────────────────────
def plot_tornado(model, feature_names=None, n_std=1.0, save=True):
    """
    For each feature: compute P(rain) at baseline-n_std and baseline+n_std.
    Show the swing as a horizontal bar, sorted by impact.
    Uses population std estimated from FEATURE_CATALOG ranges (or data if available).
    """
    if feature_names is None:
        feature_names = get_feature_names(model) or list(BASELINE.keys())
    col_map = {f.split("__")[-1]: f for f in feature_names}

    baseline_df = make_baseline_df(model, feature_names)
    base_prob = float(predict_proba(model, baseline_df)[0])

    results = []
    for short_name, col_name in col_map.items():
        if short_name not in FEATURE_CATALOG:
            continue
        info = FEATURE_CATALOG[short_name]
        lo, hi = info["range"]
        std = (hi - lo) / 6.0          # approximate ≈ 1 std if range ≈ 6σ
        base_val = baseline_df.loc[0, col_name]

        low_val  = max(lo, base_val - n_std * std)
        high_val = min(hi, base_val + n_std * std)

        row_low = baseline_df.copy(); row_low.loc[0, col_name] = low_val
        row_high = baseline_df.copy(); row_high.loc[0, col_name] = high_val
        p_low  = float(predict_proba(model, row_low)[0])
        p_high = float(predict_proba(model, row_high)[0])

        results.append({
            "feature": info["label"],
            "p_low":   p_low,
            "p_high":  p_high,
            "swing":   abs(p_high - p_low),
            "low_val":  low_val,
            "high_val": high_val,
            "unit":     info["unit"],
        })

    df = pd.DataFrame(results).sort_values("swing", ascending=True)

    fig, ax = plt.subplots(figsize=(9, max(5, len(df) * 0.45)))

    for i, row in enumerate(df.itertuples()):
        lo_p = min(row.p_low, row.p_high)
        hi_p = max(row.p_low, row.p_high)
        color = BLUE if row.p_high >= row.p_low else CORAL

        # Full bar (light background)
        ax.barh(i, 1, left=0, height=0.5, color="#f0f0f0", zorder=1)
        # Swing bar
        ax.barh(i, hi_p - lo_p, left=lo_p, height=0.5, color=color, alpha=0.8, zorder=2)

        # Baseline line
        ax.axvline(base_prob, color=GRAY, lw=1, ls="--", zorder=3)

        # Annotations at bar tips
        ax.text(lo_p - 0.005, i, f"{lo_p:.0%}", va="center", ha="right",
                fontsize=8.5, color=CORAL if row.p_high >= row.p_low else BLUE)
        ax.text(hi_p + 0.005, i, f"{hi_p:.0%}", va="center", ha="left",
                fontsize=8.5, color=BLUE if row.p_high >= row.p_low else CORAL)

    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(df["feature"].tolist(), fontsize=10)
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
    ax.set_xlabel("P(rain)")
    ax.set_title(f"Tornado chart — P(rain) sensitivity (±{n_std:.0f} std per feature)\n"
                 f"Baseline P(rain) = {base_prob:.1%}",
                 fontsize=12, pad=10)
    ax.grid(axis="x", zorder=0)
    ax.grid(axis="y", visible=False)

    fig.tight_layout()
    if save:
        path = os.path.join(OUTPUT_DIR, "fig3_tornado.png")
        fig.savefig(path, bbox_inches="tight")
        print(f"  Saved → {path}")
    return fig


# ── Figure 4 — ICE curves (Individual Conditional Expectation) ───────────────
def plot_ice(model, X_sample, feature_names=None,
             target_feature="MaxTemp", n_ice=50, save=True):
    """
    ICE curves: show per-sample variation around the PDP mean.
    Requires a sample of real data rows (X_sample, DataFrame).

    Unlike PDP (which shows the average), ICE reveals whether all samples
    respond the same way or whether subgroups behave differently.
    """
    if feature_names is None:
        feature_names = get_feature_names(model) or list(BASELINE.keys())
    col_map = {f.split("__")[-1]: f for f in feature_names}

    if target_feature not in col_map:
        print(f"  WARNING: '{target_feature}' not in model features. Skipping ICE plot.")
        return None

    col_name = col_map[target_feature]
    info = FEATURE_CATALOG.get(target_feature, {"range": (0, 50), "unit": "°C", "label": target_feature})

    sample = X_sample.sample(min(n_ice, len(X_sample)), random_state=42)
    grid   = np.linspace(*info["range"], 60)

    fig, ax = plt.subplots(figsize=(8, 5))

    # ICE lines
    for _, row_vals in sample.iterrows():
        row_df = pd.DataFrame([row_vals])
        probs = []
        for val in grid:
            r = row_df.copy()
            r.loc[r.index[0], col_name] = val
            probs.append(float(predict_proba(model, r)[0]))
        ax.plot(grid, probs, color=BLUE, alpha=0.12, lw=0.8)

    # PDP mean on top
    mean_probs = []
    baseline_df = make_baseline_df(model, feature_names)
    for val in grid:
        baseline_df.loc[0, col_name] = val
        mean_probs.append(float(predict_proba(model, baseline_df)[0]))
    ax.plot(grid, mean_probs, color=CORAL, lw=2.5, label="PDP mean", zorder=5)

    ax.set_xlabel(f"{info['label']} ({info['unit']})")
    ax.set_ylabel("P(rain)")
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
    ax.set_title(f"ICE curves — P(rain) vs {info['label']}\n"
                 f"{len(sample)} individual samples (blue) + PDP mean (orange)",
                 fontsize=12, pad=10)
    ax.legend(frameon=False)
    fig.tight_layout()

    if save:
        path = os.path.join(OUTPUT_DIR, "fig4_ice.png")
        fig.savefig(path, bbox_inches="tight")
        print(f"  Saved → {path}")
    return fig


# ── Figure 5 — Dashboard (all panels) ─────────────────────────────────────────
def plot_dashboard(model, feature_names=None,
                   feat_x="MaxTemp", feat_y="Humidity3pm",
                   n_points=30, n_std=1.0, save=True):
    """Compact 2×2 dashboard combining all what-if techniques."""
    if feature_names is None:
        feature_names = get_feature_names(model) or list(BASELINE.keys())
    col_map = {f.split("__")[-1]: f for f in feature_names}
    baseline_df = make_baseline_df(model, feature_names)
    base_prob   = float(predict_proba(model, baseline_df)[0])

    fig = plt.figure(figsize=(16, 13))
    fig.suptitle("What-if scenario analysis — rain prediction (XGBoost)",
                 fontsize=14, y=0.99)
    gs  = fig.add_gridspec(2, 2, hspace=0.38, wspace=0.30)

    # ── Panel A: PDPs for 4 key features ──────────────────────────────────────
    ax_pdp = [fig.add_subplot(gs[0, 0])]   # single axis, 4 overlaid curves
    ax = ax_pdp[0]
    pdp_targets = ["MaxTemp", "Humidity3pm", "Pressure3pm", "Cloud3pm"]
    colors_pdp  = [CORAL, BLUE, TEAL, AMBER]

    for feat, color in zip(pdp_targets, colors_pdp):
        if feat not in col_map:
            continue
        col_name = col_map[feat]
        info = FEATURE_CATALOG.get(feat, {"range": (0, 50), "unit": "", "label": feat})
        grid = np.linspace(*info["range"], 60)

        # normalise to 0-1 for overlay (x-axis = percentile rank)
        probs = []
        for val in grid:
            row = baseline_df.copy(); row.loc[0, col_name] = val
            probs.append(float(predict_proba(model, row)[0]))

        pct_grid = np.linspace(0, 100, len(grid))
        ax.plot(pct_grid, probs, color=color, lw=1.8, label=info["label"])

    ax.axhline(base_prob, color=GRAY, lw=0.8, ls=":", alpha=0.7,
               label=f"Baseline {base_prob:.0%}")
    ax.axhline(0.5, color=GRAY, lw=0.5, ls="--", alpha=0.4)
    ax.set_xlabel("Feature percentile rank")
    ax.set_ylabel("P(rain)")
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
    ax.set_title("A — PDP: key features", fontsize=11, pad=6)
    ax.legend(fontsize=8.5, frameon=False)

    # ── Panel B: 2D heatmap ────────────────────────────────────────────────────
    ax_hm = fig.add_subplot(gs[0, 1])
    if feat_x in col_map and feat_y in col_map:
        col_x, col_y = col_map[feat_x], col_map[feat_y]
        info_x = FEATURE_CATALOG.get(feat_x, {"range": (0, 50), "unit": "°C", "label": feat_x})
        info_y = FEATURE_CATALOG.get(feat_y, {"range": (0, 100), "unit": "%", "label": feat_y})
        x_vals = np.linspace(*info_x["range"], n_points)
        y_vals = np.linspace(*info_y["range"], n_points)
        rows = []
        for y in y_vals:
            for x in x_vals:
                row = baseline_df.copy()
                row.loc[0, col_x] = x; row.loc[0, col_y] = y
                rows.append(row)
        probs_2d = predict_proba(model, pd.concat(rows, ignore_index=True)).reshape(n_points, n_points)
        im = ax_hm.pcolormesh(x_vals, y_vals, probs_2d, cmap="RdYlBu_r",
                              vmin=0, vmax=1, shading="auto")
        ax_hm.contour(x_vals, y_vals, probs_2d, levels=[0.5],
                      colors="white", linewidths=1.2, linestyles="--")
        fig.colorbar(im, ax=ax_hm, pad=0.02).set_label("P(rain)")
        bx = baseline_df.loc[0, col_x]; by = baseline_df.loc[0, col_y]
        ax_hm.scatter([bx], [by], marker="+", s=150, color="white", lw=2, zorder=5)
        ax_hm.set_xlabel(f"{info_x['label']} ({info_x['unit']})")
        ax_hm.set_ylabel(f"{info_y['label']} ({info_y['unit']})")
    ax_hm.set_title("B — 2D heatmap: Temp × Humidity", fontsize=11, pad=6)

    # ── Panel C: Tornado ───────────────────────────────────────────────────────
    ax_tr = fig.add_subplot(gs[1, 0])
    results = []
    for short_name, col_name in col_map.items():
        if short_name not in FEATURE_CATALOG:
            continue
        info  = FEATURE_CATALOG[short_name]
        lo, hi = info["range"]
        std    = (hi - lo) / 6.0
        base_v = baseline_df.loc[0, col_name]
        low_v  = max(lo, base_v - n_std * std)
        high_v = min(hi, base_v + n_std * std)
        r_low  = baseline_df.copy(); r_low.loc[0, col_name] = low_v
        r_high = baseline_df.copy(); r_high.loc[0, col_name] = high_v
        p_low  = float(predict_proba(model, r_low)[0])
        p_high = float(predict_proba(model, r_high)[0])
        results.append({"feature": info["label"], "p_low": p_low,
                        "p_high": p_high, "swing": abs(p_high - p_low)})

    df_tr = pd.DataFrame(results).sort_values("swing", ascending=True).tail(10)
    for i, row in enumerate(df_tr.itertuples()):
        lo_p = min(row.p_low, row.p_high)
        hi_p = max(row.p_low, row.p_high)
        color = BLUE if row.p_high >= row.p_low else CORAL
        ax_tr.barh(i, hi_p - lo_p, left=lo_p, height=0.55, color=color, alpha=0.8)
    ax_tr.axvline(base_prob, color=GRAY, lw=1, ls="--")
    ax_tr.set_yticks(range(len(df_tr)))
    ax_tr.set_yticklabels(df_tr["feature"].tolist(), fontsize=9)
    ax_tr.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
    ax_tr.set_title("C — Tornado: ±1 std sensitivity (top 10)", fontsize=11, pad=6)
    ax_tr.grid(axis="x"); ax_tr.grid(axis="y", visible=False)

    # ── Panel D: Scenario comparison table ────────────────────────────────────
    ax_sc = fig.add_subplot(gs[1, 1])
    ax_sc.axis("off")

    scenarios = [
        ("Baseline",          {}),
        ("Hot dry day",       {"MaxTemp": 38, "Humidity3pm": 20, "Pressure3pm": 1022}),
        ("Hot humid day",     {"MaxTemp": 35, "Humidity3pm": 85, "Cloud3pm": 7}),
        ("Cool rainy",        {"MaxTemp": 15, "Humidity3pm": 90, "Rainfall": 10, "Cloud3pm": 8}),
        ("High pressure",     {"Pressure3pm": 1030, "Humidity3pm": 35}),
        ("Storm conditions",  {"Humidity3pm": 95, "WindGustSpeed": 85, "Cloud3pm": 8,
                               "Pressure3pm": 998}),
    ]

    sc_results = []
    for name, kwargs in scenarios:
        bdf = make_baseline_df(model, feature_names)
        for k, v in kwargs.items():
            matched = [f for f in feature_names if f == k or f.endswith("__" + k)]
            if matched:
                bdf.loc[0, matched[0]] = v
        p = float(predict_proba(model, bdf)[0])
        sc_results.append((name, p, "RAIN" if p >= 0.5 else "no rain"))

    headers = ["Scenario", "P(rain)", "Verdict"]
    cell_text = [[name, f"{p:.1%}", verdict] for name, p, verdict in sc_results]
    colors_sc = [
        ["#f5f5f5"] * 3,
        *[
            [["#f5f5f5", "#EAF3DE" if v == "no rain" else "#FCEBEB", "#f5f5f5"]]
            for _, _, v in sc_results[1:]
        ]
    ]

    tbl = ax_sc.table(cellText=cell_text, colLabels=headers,
                      loc="center", cellLoc="left")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9.5)
    tbl.scale(1.0, 1.6)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor("#dddddd")
        if r == 0:
            cell.set_facecolor("#E6F1FB")
        elif c == 1:
            _, p, v = sc_results[r - 1]
            cell.set_facecolor("#FCEBEB" if v == "RAIN" else "#EAF3DE")
        else:
            cell.set_facecolor("#f9f9f9")

    ax_sc.set_title("D — Scenario comparison", fontsize=11, pad=6)

    if save:
        path = os.path.join(OUTPUT_DIR, "fig5_whatif_dashboard.png")
        fig.savefig(path, bbox_inches="tight")
        print(f"  Saved → {path}")
    return fig


# ── Interactive CLI ────────────────────────────────────────────────────────────
def run_interactive_cli(model, feature_names=None):
    """
    Terminal-based what-if explorer. User types feature=value pairs
    and gets an instant P(rain) with a change vs baseline.
    """
    if feature_names is None:
        feature_names = get_feature_names(model) or list(BASELINE.keys())

    baseline_df = make_baseline_df(model, feature_names)
    base_prob   = float(predict_proba(model, baseline_df)[0])

    print("\n" + "─" * 60)
    print("  What-if rain predictor (XGBoost)")
    print(f"  Baseline P(rain) = {base_prob:.1%}")
    print("  Type  feature=value  (e.g.  MaxTemp=38  Humidity3pm=90)")
    print("  Multiple pairs separated by spaces.")
    print("  Type  'reset' to go back to baseline.  'quit' to exit.")
    print("─" * 60)

    while True:
        try:
            raw = input("\n  what-if> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n  Bye.")
            break

        if not raw:
            continue
        if raw.lower() in ("quit", "exit", "q"):
            print("  Bye.")
            break
        if raw.lower() == "reset":
            print(f"  Reset to baseline: P(rain) = {base_prob:.1%}")
            continue

        kwargs = {}
        for token in raw.split():
            if "=" not in token:
                print(f"  Skipping '{token}' — expected format: feature=value")
                continue
            k, v = token.split("=", 1)
            try:
                kwargs[k.strip()] = float(v.strip())
            except ValueError:
                print(f"  Could not parse value '{v}' for '{k}'. Skipped.")

        if not kwargs:
            continue

        p = predict_scenario(model, feature_names, verbose=False, **kwargs)
        delta = p - base_prob
        sign  = "+" if delta >= 0 else ""
        bar   = "█" * int(p * 30)
        print(f"\n  P(rain) = {p:.1%}  ({sign}{delta:.1%} vs baseline)")
        print(f"  [{bar:<30}]  {'◀ RAIN' if p >= 0.5 else ''}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="What-if analysis for rain prediction.")
    parser.add_argument("--interactive", action="store_true",
                        help="Launch the terminal what-if explorer.")
    parser.add_argument("--data",        type=str, default=DATA_PATH,
                        help="Path to original CSV data (enables ICE curves).")
    parser.add_argument("--model",       type=str, default=MODEL_PATH,
                        help="Path to the joblib model file.")
    args = parser.parse_args()

    print("\nLoading model...")
    model = load_model(args.model)
    feature_names = get_feature_names(model)
    if feature_names:
        print(f"  Features detected: {len(feature_names)}")

    if args.interactive:
        run_interactive_cli(model, feature_names)
        return

    print("\nGenerating what-if figures...")
    plot_pdp(model, feature_names)
    plot_heatmap_2d(model, feature_names)
    plot_tornado(model, feature_names)

    if args.data and os.path.exists(args.data):
        print(f"  Loading data for ICE: {args.data}")
        X = pd.read_csv(args.data)
        X_numeric = X.select_dtypes(include="number").dropna()
        if feature_names:
            shared = [c for c in X_numeric.columns if c in feature_names]
            if shared:
                X_numeric = X_numeric[shared]
        plot_ice(model, X_numeric, feature_names)
    else:
        print("  Skipping ICE (no data file). "
              "Pass --data your_data.csv to enable it.")

    plot_dashboard(model, feature_names)

    # Quick scenario demo
    print("\nScenario demo:")
    scenarios = [
        ("Hot dry day",      {"MaxTemp": 38, "Humidity3pm": 20}),
        ("Hot humid day",    {"MaxTemp": 35, "Humidity3pm": 90}),
        ("Storm conditions", {"Humidity3pm": 95, "WindGustSpeed": 85,
                               "Cloud3pm": 8, "Pressure3pm": 998}),
    ]
    for name, kwargs in scenarios:
        p = predict_scenario(model, feature_names, verbose=False, **kwargs)
        print(f"  {name:<20}  P(rain) = {p:.1%}")

    print(f"\nDone. All outputs saved to: {os.path.abspath(OUTPUT_DIR)}")


if __name__ == "__main__":
    main()
