# Pluie en Australie — Projet de Prédiction

Prédiction de **s'il pleuvra demain** (`RainTomorrow`) et de la **température maximale** (`MaxTemp`) à partir d'observations météorologiques quotidiennes provenant de ~49 stations australiennes.

Ce projet implémente des pipelines de **deep learning** (PyTorch CNN/LSTM) et de **machine learning classique** (scikit-learn, XGBoost), avec une évaluation rigoureuse, un réglage des hyperparamètres, une calibration des probabilités et une journalisation structurée. Il inclut également un **dashboard Streamlit** interactif pour visualiser et explorer les prédictions en temps réel.

---

## Table des Matières

- [Dataset](#dataset)
- [Structure du Projet](#structure-du-projet)
- [Installation](#installation)
- [Vue d'Ensemble du Pipeline](#vue-densemble-du-pipeline)
- [Modèles](#modèles)
- [Résultats Clés](#résultats-clés)
- [Notebooks](#notebooks)
- [Modules](#modules)
- [Journalisation](#journalisation)
- [Dashboard](#dashboard)
- [Contributeurs](#contributeurs)

---

## Dataset

**Source :** [Kaggle — Weather Dataset (Rattle Package)](https://www.kaggle.com/datasets/jsphyg/weather-dataset-rattle-package)

| Propriété | Valeur |
|-----------|--------|
| Lignes | 145 460 |
| Colonnes | 23 (brutes) |
| Stations | 49 villes australiennes |
| Période | 2007–2017 |
| Cible (classification) | `RainTomorrow` (binaire) |
| Cible (régression) | `MaxTemp` (°C) |
| Déséquilibre des classes | ~78% Non / ~22% Oui |

---

## Structure du Projet

```
australia_rain/
├── data/
│   ├── weatherAUS.csv                  # Dataset brut Kaggle (~14 Mo)
│   └── clean_data.csv                  # Dataset nettoyé pour le pipeline DL
├── artifacts/                          # Visualisations EDA (PNG)
├── predict/                            # CSV de sorties de prédiction
├── saved_models/                       # Modèles ML sérialisés (.joblib)
│   ├── xgboost.joblib
│   └── logistic_regression.joblib
├── logs/                               # Logs d'entraînement et de fine-tuning (CSV + fichiers log)
├── exploration_dl/                     # Exploration et tests Deep Learning
│
├── app.py                              # Application Streamlit
├── requirements.txt                    # Dépendances Python
├── model.py                            # Définitions des modèles PyTorch (CNN, LSTM)
├── classical_models.py                 # Pipelines ML classiques & évaluation
├── hyperparameter_tuning.py            # GridSearch, RandomSearch, Optuna, calibration
├── training_logger.py                  # Journalisation structurée (fichier + console + CSV)
├── interpretabilite.py                 # Importance des variables & interprétabilité
│
├── data_explo_and_prep.ipynb           # Nettoyage des données & feature engineering de base
├── rain_australia_analysis.ipynb       # EDA complète & feature engineering avancé
├── training.ipynb                      # Entraînement des modèles DL (CNN/LSTM)
├── classical_training.ipynb            # Entraînement ML classique & évaluation
├── classical_finetuning.ipynb          # Réglage des hyperparamètres & calibration
├── exrtact_prediction.ipynb            # Inférence & évaluation DL
├── extract_prediction_classical.ipynb  # Inférence ML classique (supporte le filtrage par ville)
│
├── weatherAUS_clean_features.csv       # Dataset avec features engineerées et colonne Location
├── weather_model.pth                   # Checkpoint du modèle DL sauvegardé
├── ARCHITECTURE.md                     # Documentation détaillée de l'architecture
├── CHANGELOG.md                        # Historique des versions
├── PROGRESS.md                         # Suivi de l'avancement vs cahier des charges
└── README.md                           # Ce fichier
```

---

## Installation

### Prérequis
- Python 3.10+
- pip

### Installer les dépendances
```bash
pip install -r requirements.txt
```

Ou manuellement :
```bash
pip install pandas numpy scikit-learn xgboost matplotlib seaborn joblib optuna tqdm torch kagglehub streamlit
```

### Démarrage rapide
1. **Feature engineering :** Lancer `rain_australia_analysis.ipynb` pour générer `weatherAUS_clean_features.csv`
2. **Entraînement :** Lancer `classical_training.ipynb` pour entraîner les 4 modèles classiques + le régresseur de température
3. **Fine-tuning (optionnel) :** Lancer `classical_finetuning.ipynb` pour le réglage des hyperparamètres et la calibration
4. **Prédiction :** Lancer `extract_prediction_classical.ipynb` pour les prédictions (toutes les villes ou une ville spécifique)
5. **Dashboard :** Lancer l'application Streamlit (voir [Dashboard](#dashboard))

---

## Vue d'Ensemble du Pipeline

```
Données Brutes → EDA & Feature Engineering → Dataset Propre → Entraînement → Fine-Tuning → Inférence
                                                    ↓                              ↓
                                               Logs (CSV)               Modèles Sauvegardés (.joblib)
                                                                                   ↓
                                                                        Dashboard Streamlit
```

### 1. Exploration & Feature Engineering
- **`data_explo_and_prep.ipynb`** : Nettoyage de base (imputation médiane/mode), `City_Encoded`, encodage cyclique
- **`rain_australia_analysis.ipynb`** : EDA avancée avec 7 visualisations de qualité publication, KNNImputer, 70+ features engineerées (lag/rolling, interactions de température, target encoding `Location_rainrate`), et **conservation de la colonne `Location`** pour le filtrage en aval

### 2. Entraînement des Modèles
- **`classical_training.ipynb`** : Entraîne 4 classifieurs (Régression Logistique, Arbre de Décision, Random Forest, XGBoost) + 1 régresseur de température (GradientBoosting) avec validation croisée stratifiée en 5 folds
- **`training.ipynb`** : Entraînement DL avec WeatherCNN (fenêtres de séquences de 7 jours)

### 3. Réglage des Hyperparamètres & Calibration
- **`classical_finetuning.ipynb`** : GridSearchCV, RandomizedSearchCV, optimisation bayésienne Optuna, calibration des probabilités (isotonique), courbes d'apprentissage

### 4. Interprétabilité
- **`interpretabilite.py`** : Visualisation de l'importance des variables et interprétation des résultats des modèles

### 5. Inférence & Évaluation
- **`extract_prediction_classical.ipynb`** : Prédictions par localisation — prédit pour toutes les villes ou une ville spécifique (ex. `location="Sydney"`)
- **`exrtact_prediction.ipynb`** : Inférence DL avec prédiction par ville

---

## Modèles

### Modèles ML Classiques (dans `classical_models.py`)

| Modèle | Description | Hyperparamètres Clés |
|--------|-------------|----------------------|
| **Régression Logistique** | Régularisation L2, poids de classe équilibrés | `C=1.0`, `solver=lbfgs` |
| **Arbre de Décision** | Profondeur contrainte, équilibré | `max_depth=10`, `min_samples_leaf=50` |
| **Random Forest** | 200 estimateurs, équilibré | `max_depth=15`, `min_samples_leaf=20` |
| **XGBoost** | Gradient boosting avec `scale_pos_weight` pour le déséquilibre | `n_estimators=200`, `lr=0.05` |
| **Régresseur de Température** | GradientBoostingRegressor pour MaxTemp | `n_estimators=200`, `lr=0.05` |

### Modèles Deep Learning (dans `model.py`)

| Modèle | Architecture | Entrée |
|--------|-------------|--------|
| **WeatherCNN** | Conv1D → BatchNorm → ReLU → Dropout → FC | Séquences de 7 jours |
| **WeatherLSTM** | LSTM → Dropout → FC | Séquences de 7 jours |

### Encodage de la Localisation

La colonne `Location` utilise un **target encoding** (`Location_rainrate`) — le taux de pluie par ville — comme feature numérique pour les modèles. La chaîne brute `Location` est conservée dans le dataset à des fins de filtrage et d'affichage.

---

## Résultats Clés

### Validation Croisée (5 folds Stratifiés)

| Modèle | ROC-AUC | F1 | Accuracy |
|--------|---------|-----|----------|
| XGBoost | 0.8801 ± 0.0025 | 0.6404 ± 0.0042 | 0.8111 ± 0.0019 |
| Random Forest | 0.8732 ± 0.0020 | 0.6345 ± 0.0026 | 0.8158 ± 0.0008 |
| Régression Logistique | 0.8511 ± 0.0026 | 0.5968 ± 0.0034 | 0.7783 ± 0.0013 |
| Arbre de Décision | 0.8440 ± 0.0015 | 0.5906 ± 0.0008 | 0.7733 ± 0.0030 |

> **Note :** Ces résultats proviennent de l'ancienne source de données (`clean_data.csv`). Les résultats peuvent différer avec `weatherAUS_clean_features.csv` qui contient 70+ features.

---

## Notebooks

| Notebook | Objectif |
|----------|---------|
| `rain_australia_analysis.ipynb` | EDA, visualisation, feature engineering → `weatherAUS_clean_features.csv` |
| `data_explo_and_prep.ipynb` | Nettoyage de base → `data/clean_data.csv` |
| `classical_training.ipynb` | Entraînement des 4 classifieurs + régresseur de température, avec CV et journalisation |
| `classical_finetuning.ipynb` | GridSearch, RandomSearch, Optuna, calibration, courbes d'apprentissage |
| `extract_prediction_classical.ipynb` | Inférence ML classique — supporte les prédictions par localisation |
| `training.ipynb` | Entraînement des modèles DL (CNN/LSTM) |
| `exrtact_prediction.ipynb` | Inférence & évaluation DL |

---

## Modules

| Module | Objectif |
|--------|---------|
| `classical_models.py` | Pipelines de modèles, préprocesseur, évaluation, constantes (`FEATURE_COLUMNS`, `TARGET_RAIN`, `LOCATION_COLUMN`, `DATA_PATH`) |
| `hyperparameter_tuning.py` | GridSearchCV, RandomizedSearchCV, Optuna, calibration, courbes d'apprentissage |
| `training_logger.py` | Journalisation structurée avec persistance CSV |
| `model.py` | Architectures des modèles DL PyTorch |
| `interpretabilite.py` | Importance des variables & interprétabilité des modèles |

---

## Journalisation

L'entraînement et le fine-tuning produisent des logs structurés dans le répertoire `logs/` :

| Fichier | Contenu |
|---------|---------|
| `classical_training.log` | Logs lisibles de la session d'entraînement |
| `classical_finetuning.log` | Logs lisibles de la session de fine-tuning |
| `training_metrics.csv` | Métriques d'évaluation par modèle (accuracy, ROC-AUC, F1, précision, rappel) |
| `cv_results.csv` | Résultats de validation croisée (moyenne ± écart-type par métrique) |
| `tuning_results.csv` | Meilleurs hyperparamètres et scores par méthode de tuning |
| `calibration_results.csv` | Comparaison modèle brut vs modèle calibré |

---

## Dashboard

Un **dashboard Streamlit** interactif (`app.py`) permet de visualiser et d'explorer les prédictions de pluie à travers l'Australie.

### Lancement

```bash
# Installer les dépendances
pip install -r requirements.txt

# Lancer l'application
streamlit run app.py
```

L'application s'ouvre automatiquement sur **http://localhost:8504/**.

### Fonctionnalités

| Fonctionnalité | Description |
|----------------|-------------|
| 🤖 Changement de modèle | Bascule entre XGBoost et Régression Logistique en temps réel |
| 🎚️ Sliders météo | 16 paramètres ajustables groupés par catégorie (température, humidité, vent, pression…) |
| 🗺️ Carte interactive | Probabilité de pluie par ville, colorée par intensité |
| 🌪️ Tornado chart | Sensibilité de chaque feature (±1σ) sur P(pluie) |
| 📋 Tableau filtrable | Tri, recherche et filtre pluie/sec |
| ↩️ Réinitialisation | Remet tous les sliders aux valeurs médianes australiennes |

---

## Contributeurs

- **Thibault GM**
- **LIMAMMohamedlimam**
- **lindylyndi**
