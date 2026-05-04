# Rain in Australia - Prediction Project

Ce projet vise à analyser les données météorologiques australiennes et à entraîner des modèles de Machine Learning classiques pour prédire les précipitations.

## Structure du Projet

- data  Contient les données brutes.
- weatherAUS_clean_features.csv  Dataset nettoyé prêt pour l'entraînement.
- saved_models  Stockage des modèles entraînés.
- scripts & classical_models.py  Définition des architectures et fonctions utilitaires.
- exploration_dl  Dossier dédié aux tests sur le Deep Learning.
- results & logs  Sorties des entraînements et suivis de performance via training_logger.py.

## Utilisation

### 1. Analyse Exploratoire
Consultez le notebook rain_australia_analysis.ipynb pour comprendre les corrélations et les distributions des données météo.

### 2. Entraînement des Modèles
Vous pouvez lancer l'entraînement via les notebooks ou les scripts Python 
- classical_training.ipynb  Pipeline complet d'entraînement.
- classical_finetuning.py  Optimisation des modèles existants.
- hyperparameter_tuning.py  Recherche des meilleurs paramètres.

### 3. Application et Prédiction
- Application.ipynb  Interface ou script pour tester le modèle sur de nouvelles données.
- extract_prediction_classical.ipynb  Script d'extraction des résultats de prédiction.

## Installation

# Installer les dépendances
pip install -r requirements.txt