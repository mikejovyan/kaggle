# Binary Classification with a Bank Churn Dataset

**Competition:** [Binary Classification with a Bank Churn Dataset](https://www.kaggle.com/competitions/playground-series-s4e1)

A binary classification project predicting whether a bank customer will churn. Submissions are evaluated on area under the ROC curve between the predicted probability and the observed target `Exited`.

## Dataset

```bash
kaggle competitions download -c playground-series-s4e1
```

- Training samples: 165,034
- Target classes: 0/No churn (79%), 1/Churn (21%)
- Features: 12 total
  - Numerical: 4 (`Age`, `Balance`, `CreditScore`, `EstimatedSalary`)
  - Nominal: 1 (`Geography`)
  - Ordinal: 5 (`Gender`, `HasCrCard`, `IsActiveMember`, `NumOfProducts`, `Tenure`)
  - High-cardinality: 2 (`CustomerId`, `Surname`)
- Missing values: none

## Results

The Kaggle submission achieved a private score of `0.89272`, compared to the [leaderboard](https://www.kaggle.com/competitions/playground-series-s4e1/leaderboard) top score of `0.90585`.

Models ranked by test ROC AUC (80/20 train-test split):

| Model | ROC AUC | Train time (s) |
|-------|---------|-----------------|
| LightGBM (Tuned) | 0.896 | 444.9 |
| CatBoost | 0.895 | 58.3 |
| LightGBM | 0.895 | 2.9 |
| Gradient Boosting | 0.894 | 69.6 |
| XGBoost | 0.892 | 2.7 |
| Random Forest | 0.884 | 8.0 |
| AdaBoost | 0.876 | 14.3 |
| Logistic Regression | 0.825 | 1.9 |
| Decision Tree | 0.706 | 4.4 |

**Best model configuration (LightGBM):**
- `n_estimators`: 342
- `num_leaves`: 31
- `learning_rate`: 0.03066
- `subsample`: 0.6064
- `colsample_bytree`: 0.5857
- `min_child_samples`: 166
- `reg_alpha`: 3.29e-07
- `reg_lambda`: 3.27e-05

## Key findings

- `CustomerId` and `Surname` — normally dropped as pure identifiers — are kept and target-encoded as features here, and the dataset has only 23,221 unique `CustomerId`s across 165,034 rows, so each customer appears roughly 7 times. That repetition is presumably what makes the IDs carry real signal rather than just noise.
