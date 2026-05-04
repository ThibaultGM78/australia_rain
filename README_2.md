# 🌧️ Australian Rain Prediction Dashboard

Application Streamlit pour visualiser et explorer les prédictions de pluie en Australie.

## 📁 Structure attendue

```
ton_projet/
├── app.py
├── requirements.txt
├── saved_models/
│   ├── xgboost.joblib
│   └── logistic_regression.joblib
└── data/               # (optionnel)
    └── predictions.csv
```

## 🚀 Lancement

```bash
# 1. Installer les dépendances
pip install -r requirements.txt

# 2. Lancer l'app
streamlit run app.py
```

L'app s'ouvre automatiquement sur http://localhost:8501

## 🎯 Fonctionnalités

| Feature | Description |
|---------|-------------|
| 🤖 Changement de modèle | Bascule XGBoost ↔ Logistic Regression en temps réel |
| 🎚️ Sliders météo | 16 paramètres groupés (temp, humidité, vent, pression…) |
| 🗺️ Carte interactive | Probabilité de pluie par ville, colorée par intensité |
| 🌪️ Tornado chart | Sensibilité ±1σ de chaque feature sur P(pluie) |
| 📋 Tableau filtrable | Tri, recherche et filtre pluie/sec |
| ↩️ Reset | Remet tous les sliders aux valeurs médianes australiennes |

## 💡 Notes

- Les modèles doivent être des pipelines sklearn avec `predict_proba()`.
- Les features attendues sont celles du dataset original (MinTemp, MaxTemp, etc.).
- La carte utilise les mêmes paramètres pour toutes les villes — elle montre
  "si toute l'Australie avait ces conditions, où pleurait-il ?".
