# Rain in Australia — Prediction Project

Predicting **whether it will rain tomorrow** (`RainTomorrow`) and **maximum temperature** (`MaxTemp`) using daily meteorological observations from ~49 Australian weather stations.

This project implements both **deep learning** (PyTorch CNN/LSTM) and **classical machine learning** (scikit-learn, XGBoost) pipelines, with rigorous evaluation, hyperparameter tuning, probability calibration, and structured logging. It also includes an interactive **Streamlit dashboard** for visualising and exploring predictions in real time.

---

## Table of Contents

- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Setup & Installation](#setup--installation)
- [Pipeline Overview](#pipeline-overview)
- [Models](#models)
- [Key Results](#key-results)
- [Notebooks](#notebooks)
- [Modules](#modules)
- [Logging](#logging)
- [Dashboard](#dashboard)
- [Contributors](#contributors)

---

## Dataset

**Source:** [Kaggle — Weather Dataset (Rattle Package)](https://www.kaggle.com/datasets/jsphyg/weather-dataset-rattle-package)

| Property | Value |
|----------|-------|
| Rows | 145,460 |
| Columns | 23 (raw) |
| Stations | 49 Australian cities |
| Period | 2007–2017 |
| Target (classification) | `RainTomorrow` (binary) |
| Target (regression) | `MaxTemp` (°C) |
| Class imbalance | ~78% No / ~22% Yes |

---

## Project Structure

```
australia_rain/
├── data/
│   ├── weatherAUS.csv                  # Raw Kaggle dataset (~14 MB)
│   └── clean_data.csv                  # Cleaned dataset for DL pipeline
├── artifacts/                          # EDA visualisations (PNG)
├── predict/                            # Prediction output CSVs
├── saved_models/                       # Serialised ML models (.joblib)
│   ├── xgboost.joblib
│   └── logistic_regression.joblib
├── logs/                               # Training & fine-tuning logs (CSV + log files)
├── exploration_dl/                     # Deep Learning exploration & tests
│
├── app.py                              # Streamlit dashboard application
├── requirements.txt                    # Python dependencies
├── model.py                            # PyTorch model definitions (CNN, LSTM)
├── classical_models.py                 # Classical ML pipelines & evaluation
├── hyperparameter_tuning.py            # GridSearch, RandomSearch, Optuna, calibration
├── training_logger.py                  # Structured logging (file + console + CSV)
├── interpretabilite.py                 # Feature importance & model interpretability
│
├── data_explo_and_prep.ipynb           # Data cleaning & basic feature engineering
├── rain_australia_analysis.ipynb       # Comprehensive EDA & advanced feature engineering
├── training.ipynb                      # DL model training (CNN/LSTM)
├── classical_training.ipynb            # Classical ML training & evaluation
├── classical_finetuning.ipynb          # Hyperparameter tuning & calibration
├── exrtact_prediction.ipynb            # DL inference & evaluation
├── extract_prediction_classical.ipynb  # Classical ML inference (supports per-location)
│
├── weatherAUS_clean_features.csv       # Feature-engineered dataset with Location column
├── weather_model.pth                   # Saved DL model checkpoint
├── ARCHITECTURE.md                     # Detailed architecture documentation
├── CHANGELOG.md                        # Version history
├── PROGRESS.md                         # Progress tracking vs cahier de charge
└── README.md                           # This file
```

---

## Setup & Installation

### Prerequisites
- Python 3.10+
- pip

### Install dependencies
```bash
pip install -r requirements.txt
```

Or manually:
```bash
pip install pandas numpy scikit-learn xgboost matplotlib seaborn joblib optuna tqdm torch kagglehub streamlit
```

### Quick start
1. **Feature engineering:** Run `rain_australia_analysis.ipynb` to generate `weatherAUS_clean_features.csv`
2. **Train models:** Run `classical_training.ipynb` to train and save all 4 classical models + temperature regressor
3. **Fine-tune (optional):** Run `classical_finetuning.ipynb` for hyperparameter tuning and calibration
4. **Predict:** Run `extract_prediction_classical.ipynb` for predictions (all cities or specific location)
5. **Dashboard:** Launch the Streamlit app (see [Dashboard](#dashboard))

---

## Pipeline Overview

```
Raw Data → EDA & Feature Engineering → Clean Dataset → Training → Fine-Tuning → Inference
                                           ↓                          ↓
                                      Logs (CSV)              Saved Models (.joblib)
                                                                      ↓
                                                           Streamlit Dashboard
```

### 1. Data Exploration & Feature Engineering
- **`data_explo_and_prep.ipynb`**: Basic cleaning (median/mode imputation), `City_Encoded`, cyclical encoding
- **`rain_australia_analysis.ipynb`**: Advanced EDA with 7 publication-quality visualisations, KNNImputer, 70+ engineered features including lag/rolling, temperature interactions, target encoding (`Location_rainrate`), and **preserves the `Location` column** for downstream filtering

### 2. Model Training
- **`classical_training.ipynb`**: Trains 4 classifiers (Logistic Regression, Decision Tree, Random Forest, XGBoost) + 1 temperature regressor (GradientBoosting) with 5-fold stratified cross-validation
- **`training.ipynb`**: DL training with WeatherCNN (7-day sequence windows)

### 3. Hyperparameter Tuning & Calibration
- **`classical_finetuning.ipynb`**: GridSearchCV, RandomizedSearchCV, Optuna Bayesian optimisation, probability calibration (isotonic), learning curves

### 4. Interpretability
- **`interpretabilite.py`**: Feature importance visualisation and model result interpretation

### 5. Inference & Evaluation
- **`extract_prediction_classical.ipynb`**: Location-aware predictions — predict for all cities or a specific location (e.g., `location="Sydney"`)
- **`exrtact_prediction.ipynb`**: DL inference with per-city prediction

---

## Models

### Classical ML Models (in `classical_models.py`)

| Model | Description | Key Hyperparameters |
|-------|-------------|---------------------|
| **Logistic Regression** | L2-regularised, class-weight balanced | `C=1.0`, `solver=lbfgs` |
| **Decision Tree** | Depth-constrained, balanced | `max_depth=10`, `min_samples_leaf=50` |
| **Random Forest** | 200 estimators, balanced | `max_depth=15`, `min_samples_leaf=20` |
| **XGBoost** | Gradient boosting with `scale_pos_weight` for imbalance | `n_estimators=200`, `lr=0.05` |
| **Temperature Regressor** | GradientBoostingRegressor for MaxTemp | `n_estimators=200`, `lr=0.05` |

### Deep Learning Models (in `model.py`)

| Model | Architecture | Input |
|-------|-------------|-------|
| **WeatherCNN** | 1D Conv → BatchNorm → ReLU → Dropout → FC | 7-day sequences |
| **WeatherLSTM** | LSTM → Dropout → FC | 7-day sequences |

### Feature Encoding: Location

The `Location` column uses **target encoding** (`Location_rainrate`) — the rain rate per city — as the numeric feature for models. The raw `Location` string is preserved in the dataset for filtering and display purposes.

---

## Key Results

### Cross-Validation (5-fold Stratified)

| Model | ROC-AUC | F1 | Accuracy |
|-------|---------|-----|----------|
| XGBoost | 0.8801 ± 0.0025 | 0.6404 ± 0.0042 | 0.8111 ± 0.0019 |
| Random Forest | 0.8732 ± 0.0020 | 0.6345 ± 0.0026 | 0.8158 ± 0.0008 |
| Logistic Regression | 0.8511 ± 0.0026 | 0.5968 ± 0.0034 | 0.7783 ± 0.0013 |
| Decision Tree | 0.8440 ± 0.0015 | 0.5906 ± 0.0008 | 0.7733 ± 0.0030 |

> **Note:** These results are from the previous data source (`clean_data.csv`). Results may differ with the updated `weatherAUS_clean_features.csv` which has 70+ features.

---

## Notebooks

| Notebook | Purpose |
|----------|---------|
| `rain_australia_analysis.ipynb` | EDA, visualisation, feature engineering → `weatherAUS_clean_features.csv` |
| `data_explo_and_prep.ipynb` | Basic data cleaning → `data/clean_data.csv` |
| `classical_training.ipynb` | Train 4 classifiers + temp regressor, with CV and logging |
| `classical_finetuning.ipynb` | GridSearch, RandomSearch, Optuna, calibration, learning curves |
| `extract_prediction_classical.ipynb` | Classical ML inference — supports per-location predictions |
| `training.ipynb` | DL model training (CNN/LSTM) |
| `exrtact_prediction.ipynb` | DL inference & evaluation |

---

## Modules

| Module | Purpose |
|--------|---------|
| `classical_models.py` | Model pipelines, preprocessor, evaluation, constants (`FEATURE_COLUMNS`, `TARGET_RAIN`, `LOCATION_COLUMN`, `DATA_PATH`) |
| `hyperparameter_tuning.py` | GridSearchCV, RandomizedSearchCV, Optuna, calibration, learning curves |
| `training_logger.py` | Structured logging with CSV persistence |
| `model.py` | PyTorch DL model architectures |
| `interpretabilite.py` | Feature importance & model interpretability |

---

## Logging

Training and fine-tuning produce structured logs in the `logs/` directory:

| File | Content |
|------|---------|
| `classical_training.log` | Human-readable training session logs |
| `classical_finetuning.log` | Human-readable fine-tuning session logs |
| `training_metrics.csv` | Per-model evaluation metrics (accuracy, ROC-AUC, F1, precision, recall) |
| `cv_results.csv` | Cross-validation results (mean ± std for each metric) |
| `tuning_results.csv` | Best hyperparameters and scores per tuning method |
| `calibration_results.csv` | Raw vs calibrated model comparison |

---

## Dashboard

An interactive **Streamlit dashboard** (`app.py`) lets you visualise and explore rain predictions across Australia.

### Launch

```bash
# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

The app opens automatically at **http://localhost:8504/**.

### Features

| Feature | Description |
|---------|-------------|
| 🤖 Model switcher | Toggle between XGBoost and Logistic Regression in real time |
| 🎚️ Weather sliders | 16 adjustable parameters grouped by category (temperature, humidity, wind, pressure…) |
| 🗺️ Interactive map | Rain probability per city, colour-coded by intensity |
| 🌪️ Tornado chart | Sensitivity of each feature (±1σ) on P(rain) |
| 📋 Filterable table | Sort, search, and filter by rain/dry forecast |
| ↩️ Reset | Restore all sliders to Australian median values |

---

## Contributors

- **Thibault GM**
- **LIMAMMohamedlimam**
- **lindylyndi**
