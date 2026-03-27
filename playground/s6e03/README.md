# Predict Customer Churn

**Competition:** [Predict Customer Churn](https://www.kaggle.com/competitions/playground-series-s6e3)

A binary classification project predicting whether a customer will churn. Submissions are evaluated on area under the ROC curve between the predicted probability and the observed target `Churn`.

## Dataset

```bash
kaggle competitions download -c playground-series-s6e3
```

- Training samples: 594,194
- Target classes: 0/Not churned (78%), 1/Churned (22%)
- Features: 19 total
  - Numerical: 3 (`MonthlyCharges`, `tenure`, `TotalCharges`)
  - Nominal: 2 (`InternetService`, `PaymentMethod`)
  - Ordinal: 14 (`Contract`, `Dependents`, `DeviceProtection`, `gender`, `MultipleLines`, `OnlineBackup`, `OnlineSecurity`, `PaperlessBilling`, `Partner`, `PhoneService`, `SeniorCitizen`, `StreamingMovies`, `StreamingTV`, `TechSupport`)
- Missing values: none

## Results

The Kaggle submission achieved a private score of `TBD`, compared to the [leaderboard](https://www.kaggle.com/competitions/playground-series-s6e3/leaderboard) top score of `TBD`.

Models ranked by test ROC AUC (80/20 train-test split):

| Model | ROC AUC | Train time (s) |
|-------|---------|----------------|
| LightGBM (Tuned) | 0.917 | 1421.4 |
| CatBoost | 0.917 | 131.2 |
| XGBoost | 0.916 | 12.8 |
| LightGBM | 0.916 | 11.9 |
| Gradient Boosting | 0.914 | 234.5 |
| AdaBoost | 0.912 | 55.0 |
| Logistic Regression | 0.909 | 8.2 |
| Random Forest | 0.895 | 184.7 |
| Decision Tree | 0.725 | 15.4 |
| SVC | Too slow | - |

**Best model configuration (LightGBM):**
- `n_estimators`: 743
- `learning_rate`: 0.04511
- `num_leaves`: 46
- `subsample`: 0.543
- `colsample_bytree`: 0.505
- `min_child_samples`: 157
- `reg_alpha`: 1.55e-06
- `reg_lambda`: 9.918

## Key findings

- Gradient boosting algorithms (LightGBM, CatBoost, XGBoost, GB) dominated, all achieving ROC AUC above 0.914
- Default CatBoost matched tuned LightGBM (0.917 ROC AUC) at a fraction of the tuning time (131s vs 1421s)
- Random Forest underperformed relative to other ensemble methods (0.895 vs 0.914+)
- Dataset is heavily imbalanced (77.5% not churned vs 22.5% churned), though boosting methods handled this well without resampling

