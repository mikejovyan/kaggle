# Predicting Loan Payback

**Competition:** [Predicting Loan Payback](https://www.kaggle.com/competitions/playground-series-s5e11)

A binary classification project predicting whether a loan will be paid back based on borrower financial and demographic data. Submissions are evaluated on ROC AUC.

## Dataset

```bash
kaggle competitions download -c playground-series-s5e11
```

- Training samples: 475,195 (80% of 593,994)
- Features: 11 total (excluding target)
  - Numerical: 5 (`annual_income`, `credit_score`, `debt_to_income_ratio`, `interest_rate`, `loan_amount`)
  - Nominal: 4 (`employment_status`, `gender`, `loan_purpose`, `marital_status`)
  - Ordinal: 2 (`education_level`, `grade_subgrade`)
- Target classes: 0/Unpaid (20.12%), 1/Paid (79.88%)

## Results

The Kaggle submission achieved a private score of `0.92392`, compared to the [leaderboard](https://www.kaggle.com/competitions/playground-series-s5e11/leaderboard) top score of `0.92939`.

Models ranked by test ROC AUC (80/20 stratified split):

| Model | Accuracy | Balanced accuracy | MCC | ROC AUC | Train time (s) |
|-------|----------|-------------------|-----|---------|----------------|
| LightGBM (Tuned) | 0.906 | 0.799 | 0.688 | 0.924 | 1568.2 |
| CatBoost | 0.906 | 0.800 | 0.688 | 0.923 | 131.1 |
| XGBoost | 0.905 | 0.793 | 0.682 | 0.921 | 10.5 |
| LightGBM | 0.904 | 0.791 | 0.681 | 0.920 | 9.1 |
| Gradient Boosting | 0.903 | 0.787 | 0.677 | 0.916 | 310.4 |
| Logistic Regression | 0.900 | 0.791 | 0.666 | 0.911 | 20.9 |
| AdaBoost | 0.900 | 0.786 | 0.665 | 0.908 | 63.1 |
| Random Forest | 0.901 | 0.787 | 0.671 | 0.908 | 292.4 |
| Decision Tree | 0.847 | 0.769 | 0.532 | 0.769 | 24.6 |

**Best model configuration (LightGBM, tuned with Optuna, 40 trials):**
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
