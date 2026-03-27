# Binary Classification with a Bank Churn Dataset

**Competition:** [Binary Classification with a Bank Churn Dataset](https://www.kaggle.com/competitions/playground-series-s4e1)

A binary classification project predicting whether a bank customer will churn (exit). Submissions are evaluated on area under the ROC curve between the predicted probability and the observed target `Exited`.

## Dataset

```bash
kaggle competitions download -c playground-series-s4e1
```

- Training samples: 165,034
- Target classes: 0/Not Exited (79%), 1/Exited (21%)
- Features: 12 total
  - Numerical: 4 (`Age`, `Balance`, `CreditScore`, `EstimatedSalary`)
  - Nominal: 1 (`Geography`)
  - Ordinal: 5 (`Gender`, `HasCrCard`, `IsActiveMember`, `NumOfProducts`, `Tenure`)
  - High-cardinality: 2 (`CustomerId`, `Surname`)
- Missing values: none

## Results

The Kaggle submission achieved a private score of `0.89295`, compared to the [leaderboard](https://www.kaggle.com/competitions/playground-series-s4e1/leaderboard) top score of `0.90585`.

Models ranked by test ROC AUC (80/20 train-test split):

| Model | ROC AUC | Train time (s) |
|-------|---------|----------------|
| LightGBM | 0.896 | 3.8 |
| LightGBM (Tuned) | 0.896 | 1689.3 |
| CatBoost | 0.895 | 59.1 |
| Gradient Boosting | 0.894 | 86.7 |
| XGBoost | 0.893 | 5.6 |
| Random Forest | 0.886 | 87.4 |
| AdaBoost | 0.877 | 18.2 |
| Logistic Regression | 0.825 | 4.7 |
| Decision Tree | 0.709 | 5.2 |
| SVC | Too slow | - |

**Best model configuration (LightGBM):**
- `n_estimators`: 865
- `learning_rate`: 0.01502
- `num_leaves`: 40
- `subsample`: 0.658
- `colsample_bytree`: 0.658
- `min_child_samples`: 166
- `reg_alpha`: 0.0384
- `reg_lambda`: 4.784

## Key findings

- Gradient boosting algorithms (LightGBM, CatBoost, GB, XGBoost) dominated, all achieving ROC AUC above 0.892
- Default LightGBM matched the tuned model (0.896 ROC AUC) at a fraction of the training time (3.8s vs 1689.3s)

- Submission uses predicted churn probabilities (`predict_proba`) rather than class labels, as required by the competition
