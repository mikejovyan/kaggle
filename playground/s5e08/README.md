# Binary Classification with a Bank Dataset

**Competition:** [Binary Classification with a Bank Dataset](https://www.kaggle.com/competitions/playground-series-s5e8)

A binary classification project predicting whether a bank customer subscribes to a term deposit based on demographic and campaign data. Submissions are evaluated on area under the ROC curve between the predicted probability and the observed target `y`.

## Dataset

```bash
kaggle competitions download -c playground-series-s5e8
```

- Training samples: 750,000
- Target classes: 0/No subscribe (88%), 1/Subscribe (12%)
- Features: 16 total
  - Numerical: 7 (`age`, `balance`, `campaign`, `day`, `duration`, `pdays`, `previous`)
  - Nominal: 4 (`contact`, `job`, `marital`, `poutcome`)
  - Ordinal: 5 (`default`, `education`, `housing`, `loan`, `month`)
- Missing values: none

## Results

The Kaggle submission achieved a private score of `0.96787`, compared to the [leaderboard](https://www.kaggle.com/competitions/playground-series-s5e8/leaderboard) top score of `0.97801`.

Models ranked by test ROC AUC (80/20 train-test split):

| Model | ROC AUC | Train time (s) |
|-------|---------|----------------|
| LightGBM (Tuned) | 0.967 | 862.8 |
| CatBoost | 0.964 | 138.1 |
| XGBoost | 0.964 | 14.1 |
| LightGBM | 0.962 | 14.0 |
| Random Forest | 0.959 | 269.1 |
| Gradient Boosting | 0.952 | 286.4 |
| AdaBoost | 0.939 | 69.3 |
| Logistic Regression | 0.929 | 111.7 |
| Decision Tree | 0.774 | 23.1 |

**Best model configuration (LightGBM):**
- `n_estimators`: 771
- `num_leaves`: 58
- `learning_rate`: 0.09471
- `subsample`: 0.77658
- `colsample_bytree`: 0.59626
- `min_child_samples`: 83
- `reg_alpha`: 9.86875
- `reg_lambda`: 0.02781

## Key findings

- The dataset is heavily imbalanced (12% positive class), making balanced accuracy and ROC AUC more informative than raw accuracy
- Tuning LightGBM with Optuna improved ROC AUC from 0.962 to 0.967, the largest single-model gain in the comparison
- `duration` (last contact duration) is a strong predictor but is unknown before a call is made, which limits real-world applicability
