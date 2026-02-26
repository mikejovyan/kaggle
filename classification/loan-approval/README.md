# Loan Approval Classification

**Competition:** [Binary Classification with a Loan Approval Dataset](https://www.kaggle.com/competitions/playground-series-s4e10)

A binary classification project predicting whether an applicant is approved for a loan. Submissions are evaluated on area under the ROC curve between the predicted probability and the observed target `loan_status`.

## Dataset

```bash
kaggle competitions download -c playground-series-s4e10
```

- Training samples: 58,645
- Features: 11 total (excluding target)
  - Numerical: 7 features
  - Nominal: 2 features
  - Ordinal: 2 features
- Target classes: 0/Not Approved (85.8%), 1/Approved (14.2%)

## Results

The Kaggle submission achieved a private score of `0.96038`, compared to the [leaderboard](https://www.kaggle.com/competitions/playground-series-s4e10/leaderboard) top score of `0.96938`.

Models ranked by test ROC-AUC (80/20 train-test split):

| Model | ROC-AUC | Train Time (s) |
|-------|---------|----------------|
| LightGBM (Tuned) | 0.961 | 268.2 |
| LightGBM | 0.959 | 0.9 |
| XGBoost | 0.956 | 0.9 |
| CatBoost | 0.956 | 16.2 |
| Gradient Boosting | 0.944 | 15.6 |
| Random Forest | 0.940 | 13.5 |
| AdaBoost | 0.919 | 4.0 |
| Logistic Regression | 0.896 | 0.8 |
| SVC | 0.895 | 342.9 |
| Decision Tree | 0.837 | 1.0 |

**Best Model Configuration (LightGBM):**
- `n_estimators`: 448
- `learning_rate`: 0.02137
- `num_leaves`: 79
- `subsample`: 0.877
- `colsample_bytree`: 0.575
- `min_child_samples`: 32
- `reg_alpha`: 3.62e-08
- `reg_lambda`: 0.346

## Key Findings

- Gradient boosting algorithms (LightGBM, XGBoost, CatBoost, GB) dominated, all achieving ROC-AUC above 0.944
- Default LightGBM achieved 0.959 ROC-AUC with near-instant training (0.9s vs 268s for tuned)
- SVC completed but was much slower than most models (342.9s) for only 0.895 ROC-AUC
- Dataset is heavily imbalanced (85.8% not approved vs 14.2% approved), though boosting methods handled this well without resampling
