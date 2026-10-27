# Loan Approval Prediction

**Competition:** [Loan Approval Prediction](https://www.kaggle.com/competitions/playground-series-s4e10)

A binary classification project predicting whether an applicant is approved for a loan. Submissions are evaluated on area under the ROC curve between the predicted probability and the observed target `loan_status`.

## Dataset

```bash
kaggle competitions download -c playground-series-s4e10
```

- Training samples: 58,645
- Target classes: 0/Not Approved (86%), 1/Approved (14%)
- Features: 11 total
  - Numerical: 7 (`cb_person_cred_hist_length`, `loan_amnt`, `loan_int_rate`, `loan_percent_income`, `person_age`, `person_emp_length`, `person_income`)
  - Nominal: 2 (`loan_intent`, `person_home_ownership`)
  - Ordinal: 2 (`cb_person_default_on_file`, `loan_grade`)
- Missing values: none

## Results

The Kaggle submission achieved a private score of `0.96038`, compared to the [leaderboard](https://www.kaggle.com/competitions/playground-series-s4e10/leaderboard) top score of `0.96938`.

Models ranked by test ROC AUC (80/20 train-test split):

| Model | ROC AUC | Train time (s) |
|-------|---------|-----------------|
| LightGBM (Tuned) | 0.961 | 110.6 |
| LightGBM | 0.959 | 1.0 |
| CatBoost | 0.956 | 20.9 |
| XGBoost | 0.954 | 1.1 |
| Gradient Boosting | 0.944 | 16.8 |
| Random Forest | 0.940 | 14.3 |
| AdaBoost | 0.919 | 5.5 |
| Logistic Regression | 0.896 | 4.5 |
| Decision Tree | 0.837 | 1.0 |

**Best model configuration (LightGBM):**
- `n_estimators`: 418
- `num_leaves`: 38
- `learning_rate`: 0.02981
- `subsample`: 0.5693
- `colsample_bytree`: 0.6951
- `min_child_samples`: 44
- `reg_alpha`: 1.46e-06
- `reg_lambda`: 1.01e-07

## Key findings

- The 86/14 class imbalance means accuracy (0.95) looks great but balanced accuracy is much lower (0.86) for every model. ROC AUC is the metric to trust here — raw accuracy would be misleading.
- Tuning LightGBM (30 trials, under 2 minutes) moved ROC AUC from 0.959 to 0.961 — a real but modest gain, worth doing again if this one comes back.
- Logistic Regression and AdaBoost both fell well behind the tree ensembles (0.896 and 0.919 vs 0.94+), suggesting the approval decision depends on feature interactions a linear model can't capture.
