# Saved Models

After running `classical_training.ipynb` with training enabled, this directory contains:

| File | Model | Format |
|------|-------|--------|
| `logistic_regression.joblib` | LogisticRegression (sklearn) | joblib |
| `decision_tree.joblib` | DecisionTreeClassifier (sklearn) | joblib |
| `random_forest.joblib` | RandomForestClassifier (sklearn) | joblib |
| `xgboost.joblib` | XGBClassifier (xgboost) | joblib |

## Loading a saved model

```python
import joblib

model = joblib.load("saved_models/random_forest.joblib")
y_proba = model.predict_proba(X_test)[:, 1]
```
