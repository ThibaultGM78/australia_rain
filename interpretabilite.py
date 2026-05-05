
import joblib
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import re

# ── 1. Chargement ─────────────────────────────
model = joblib.load("/saved_models/xgboost.joblib")
df = pd.read_csv("/australia_rain/weatherAUS_clean_features.csv")

# ── 2. Colonnes manquantes ──
df["RainToday"]     = df["RainToday_bin"]
df["City_Encoded"]  = df["Location"].astype("category").cat.codes
df["Humidity_Avg"]  = df["HumidityMean"]
df["Pressure_Diff"] = df["PressureChange"]

# ── 3. Target & Features ────────────────────────
y = df["RainTomorrow_enc"]
X = df.drop(columns=["RainTomorrow_enc"], errors="ignore")

# ── 4. Extraire le XGBoost + préprocesseur ──────
xgb_model    = model[list(model.named_steps.keys())[-1]]
preprocessor = model[:-1]

X_transformed = preprocessor.transform(X)
try:
    feature_names = preprocessor.get_feature_names_out()
except:
    feature_names = [f"f{i}" for i in range(X_transformed.shape[1])]

X_transformed_df = pd.DataFrame(X_transformed, columns=feature_names)

# ── 5. PATCH base_score ─────────────────────────
booster = xgb_model.get_booster()
config  = booster.save_config()

# Extraire la valeur numérique entre crochets si présent
config = re.sub(
    r'"base_score":\s*"\[([^\]]+)\]"',
    lambda m: f'"base_score": "{float(m.group(1))}"',
    config
)
booster.load_config(config)

# ── 6. SHAP TreeExplainer ───────────────────────
explainer  = shap.TreeExplainer(booster)
shap_values = explainer(X_transformed_df)

# ── 7. Beeswarm ────────────────────────────────
plt.figure()
shap.plots.beeswarm(shap_values, max_display=20, show=False)
plt.title("SHAP Beeswarm - Top 20 features")
plt.tight_layout()
plt.savefig("shap_beeswarm.png", dpi=150, bbox_inches="tight")
plt.show()

# ── 8. Bar plot ────────────────────────────────
plt.figure()
shap.plots.bar(shap_values, max_display=20, show=False)
plt.title("SHAP Bar - Top 20 features")
plt.tight_layout()
plt.savefig("shap_bar.png", dpi=150, bbox_inches="tight")
plt.show()
