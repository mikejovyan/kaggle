# Predicting Loan Payback

**Competition:** [Predicting Loan Payback](https://www.kaggle.com/competitions/playground-series-s5e11)

A binary classification project predicting whether a loan will be paid back. Submissions are evaluated on area under the ROC curve between the predicted probability and the observed target `loan_paid_back`.

## Dataset

```bash
kaggle competitions download -c playground-series-s5e11
```

- Training samples: 593,994
- Target classes: 0/Unpaid (20%), 1/Paid (80%)
- Features: 12 total
  - Numerical: 5 (`annual_income`, `credit_score`, `debt_to_income_ratio`, `interest_rate`, `loan_amount`)
  - Nominal: 5 (`education_other`, `employment_status`, `gender`, `loan_purpose`, `marital_status`)
  - Ordinal: 2 (`education_level`, `grade_subgrade`)
- Missing values: none in the raw data, but `education_level` values of `"Other"` are deliberately recoded to NaN (with a separate `education_other` flag column) rather than kept as their own category

## Results

The Kaggle submission achieved a private score of `0.92465`, compared to the [leaderboard](https://www.kaggle.com/competitions/playground-series-s5e11/leaderboard) top score of `0.92939`.

Models ranked by test ROC AUC (80/20 train-test split):

| Model | ROC AUC | Train time (s) |
|-------|---------|-----------------|
| XGBoost (Tuned) | 0.924 | 1742.0 |
| CatBoost | 0.923 | 134.5 |
| XGBoost | 0.923 | 19.5 |
| LightGBM | 0.920 | 11.7 |
| Gradient Boosting | 0.916 | 336.6 |
| Random Forest | 0.908 | 290.9 |

**Best model configuration (XGBoost):**
- `n_estimators`: 881
- `max_depth`: 8
- `learning_rate`: 0.03015
- `subsample`: 0.6461
- `colsample_bytree`: 0.6123
- `min_child_weight`: 10
- `gamma`: 7.78e-07

## Key findings

- 29 minutes tuning XGBoost moved the Kaggle private score from 0.92338 to 0.92465 — a small but real gain, worth the time here.
- Recoding `education_level`'s rare `"Other"` value as NaN instead of its own category knocked 4 of the usual 9 candidate models (Logistic Regression, Decision Tree, AdaBoost, SVC) out of the pool, since none of them handle missing values — only tree-based models that support NaNs natively were trained.
