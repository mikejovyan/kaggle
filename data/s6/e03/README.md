# Predict Customer Churn

**Competition:** [Predict Customer Churn](https://www.kaggle.com/competitions/playground-series-s6e3)

A binary classification project predicting whether a telecom customer will churn. Submissions are evaluated on area under the ROC curve between the predicted probability and the observed target `Churn`.

## Dataset

```bash
kaggle competitions download -c playground-series-s6e3
```

- Training samples: 594,194
- Target classes: No (77%), Yes (23%)
- Features: 19 total
  - Numerical: 3 (`MonthlyCharges`, `TotalCharges`, `tenure`)
  - Nominal: 1 (`PaymentMethod`)
  - Ordinal: 15 (`Contract`, `Dependents`, `DeviceProtection`, `InternetService`, `MultipleLines`, `OnlineBackup`, `OnlineSecurity`, `PaperlessBilling`, `Partner`, `PhoneService`, `SeniorCitizen`, `StreamingMovies`, `StreamingTV`, `TechSupport`, `gender`)
- Missing values: none

## Results

The Kaggle submission achieved a private score of `0.91484`, compared to the [leaderboard](https://www.kaggle.com/competitions/playground-series-s6e3/leaderboard) top score of `0.91856`.

Models ranked by test ROC AUC (80/20 train-test split):

| Model | ROC AUC | Train time (s) |
|-------|---------|-----------------|
| XGBoost (Tuned) | 0.917 | 1793.5 |
| CatBoost | 0.917 | 137.6 |
| XGBoost | 0.916 | 15.0 |
| LightGBM | 0.916 | 13.2 |
| Gradient Boosting | 0.914 | 244.7 |
| AdaBoost | 0.912 | 57.5 |
| Logistic Regression | 0.907 | 22.4 |
| Random Forest | 0.896 | 204.6 |
| Decision Tree | 0.725 | 15.6 |

**Best model configuration (XGBoost):**
- `n_estimators`: 586
- `max_depth`: 3
- `learning_rate`: 0.19397
- `subsample`: 0.5993
- `colsample_bytree`: 0.8696
- `min_child_weight`: 9
- `gamma`: 2.27e-06

## Key findings

- 30 minutes tuning XGBoost only moved ROC AUC from 0.916 to 0.917 — not worth repeating.
- Balanced accuracy (0.785) is well below plain accuracy (0.862). With 77% of customers not churning, the model can score well just by leaning toward "no churn".
