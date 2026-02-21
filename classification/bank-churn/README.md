# Bank Churn Classification

**Competition:** [Binary Classification with a Bank Churn Dataset](https://www.kaggle.com/competitions/playground-series-s4e1)

A binary classification project predicting whether a bank customer will churn (exit). Submissions are evaluated on area under the ROC curve between the predicted probability and the observed target `Exited`.

## Dataset

```bash
kaggle competitions download -c playground-series-s4e1
```

- Training samples: 165,034
- Features: 12 total (excluding target)
  - Numerical: 4 features
  - Nominal: 1 feature
  - Ordinal: 5 features
  - High cardinality: 2 features
- Target classes: Not Exited (78.8%), Exited (21.2%)

## Results

Models ranked by test ROC-AUC (80/20 train-test split):

| Model | ROC-AUC | Train Time (s) |
|-------|---------|----------------|
| LightGBM | 0.896 | 26.1 |
| LightGBM (Tuned) | 0.896 | 411.7 |
| Gradient Boosting | 0.894 | 73.7 |
| XGBoost | 0.892 | 33.2 |
| Random Forest | 0.885 | 121.1 |
| AdaBoost | 0.876 | 14.6 |
| Logistic Regression | 0.825 | 10.8 |
| Decision Tree | 0.709 | 7.5 |
| SVC | Too slow | - |

**Best Model Configuration (LightGBM):**
- `n_estimators`: 100
- `learning_rate`: 0.1
- `num_leaves`: 31
- `subsample`: 1.0
- `colsample_bytree`: 0.8

## Key Findings

- Gradient boosting algorithms (LightGBM, GB, XGBoost) dominated, all achieving ROC-AUC above 0.892
- Default LightGBM matched the tuned model (0.896 ROC-AUC) at a fraction of the training time (26s vs 412s)
- SVC was excluded from full evaluation due to excessive training time on this dataset size
- Submission uses predicted churn probabilities (`predict_proba`) rather than class labels, as required by the competition
