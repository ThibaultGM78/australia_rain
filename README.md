# Rain in Australia - Prediction Project

Ce projet vise à analyser les données météorologiques australiennes et à entraîner des modèles de Machine Learning classiques pour prédire les précipitations.

## Structure du Projet

- data  Contient les données brutes.
- weatherAUS_clean_features.csv  Dataset nettoyé prêt pour l'entraînement.
- saved_models  Stockage des modèles entraînés.
- scripts & classical_models.py  Définition des architectures et fonctions utilitaires.
- exploration_dl  Dossier dédié aux tests sur le Deep Learning.
- results & logs  Sorties des entraînements et suivis de performance via training_logger.py.
australia_rain/



## Utilisation

### 1. Analyse Exploratoire
Consultez le notebook rain_australia_analysis.ipynb pour comprendre les corrélations et les distributions des données météo.

### 2. Entraînement des Modèles
Vous pouvez lancer l'entraînement via les notebooks ou les scripts Python 
- classical_training.ipynb  Pipeline complet d'entraînement.
- classical_finetuning.py  Optimisation des modèles existants.
- hyperparameter_tuning.py  Recherche des meilleurs paramètres.

### 3. Application et Prédiction
- extract_prediction_classical.ipynb  Script d'extraction des résultats de prédiction.
- # Australian Rain Prediction Dashboard
Application Streamlit pour visualiser et explorer les prédictions de pluie en Australie.

## 🚀 Lancement

```bash
# 1. Installer les dépendances
pip install -r requirements.txt

# 2. Lancer l'app
streamlit run app.py
```

L'app s'ouvre automatiquement sur [http://localhost:8501](http://localhost:8502/)

## 🎯 Fonctionnalités

| Feature | Description |
|---------|-------------|
| 🤖 Changement de modèle | Bascule XGBoost ↔ Logistic Regression en temps réel |
| 🎚️ Sliders météo | 16 paramètres groupés (temp, humidité, vent, pression…) |
| 🗺️ Carte interactive | Probabilité de pluie par ville, colorée par intensité |
| 🌪️ Tornado chart | Sensibilité ±1σ de chaque feature sur P(pluie) |
| 📋 Tableau filtrable | Tri, recherche et filtre pluie/sec |
| ↩️ Reset | Remet tous les sliders aux valeurs médianes australiennes |


## Installation

# Installer les dépendances
pip install -r requirements.txt

## 📁 Structure du projet (à compléter)

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

