# Predicting Loan Payback

**Competition:** [Predicting Loan Payback](https://www.kaggle.com/competitions/playground-series-s5e11)

A binary classification project predicting whether a loan will be paid back based on borrower financial and demographic data. Submissions are evaluated on area under the ROC curve between the predicted probability and the observed target `loan_paid_back`.

## Dataset

```bash
kaggle competitions download -c playground-series-s5e11
```

- Training samples: 475,195
- Target classes: 0/Unpaid (20%), 1/Paid (80%)
- Features: 11 total
  - Numerical: 5 (`annual_income`, `credit_score`, `debt_to_income_ratio`, `interest_rate`, `loan_amount`)
  - Nominal: 4 (`employment_status`, `gender`, `loan_purpose`, `marital_status`)
  - Ordinal: 2 (`education_level`, `grade_subgrade`)
- Missing values: none

## Results

The Kaggle submission achieved a private score of `0.92392`, compared to the [leaderboard](https://www.kaggle.com/competitions/playground-series-s5e11/leaderboard) top score of `0.92939`.

Models ranked by test ROC AUC (80/20 train-test split):

| Model | ROC AUC | Train time (s) |
|-------|---------|----------------|
| LightGBM (Tuned) | 0.924 | 1568.2 |
| CatBoost | 0.923 | 131.1 |
| XGBoost | 0.921 | 10.5 |
| LightGBM | 0.920 | 9.1 |
| Gradient Boosting | 0.916 | 310.4 |
| Logistic Regression | 0.911 | 20.9 |
| AdaBoost | 0.908 | 63.1 |
| Random Forest | 0.908 | 292.4 |
| Decision Tree | 0.769 | 24.6 |

**Best model configuration (LightGBM):**
- `n_estimators`: 783
- `num_leaves`: 54
- `learning_rate`: 0.06318
- `subsample`: 0.54352
- `colsample_bytree`: 0.50787
- `min_child_samples`: 153
- `reg_alpha`: 1.146e-07
- `reg_lambda`: 6.53894

## Key findings

- The dataset is moderately imbalanced (20% unpaid), making ROC AUC and balanced accuracy more informative than raw accuracy
- Tuning LightGBM with Optuna improved ROC AUC from 0.920 to 0.924, the largest gain among the top models
- `interest_rate` and `grade_subgrade` are strongly correlated with the target, as both encode lender-assessed credit risk
- The top three boosting models (LightGBM, CatBoost, XGBoost) are tightly clustered within 0.004 ROC AUC of each other with default parameters
