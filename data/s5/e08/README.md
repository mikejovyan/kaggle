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

The Kaggle submission achieved a private score of `0.96965`, compared to the [leaderboard](https://www.kaggle.com/competitions/playground-series-s5e8/leaderboard) top score of `0.97801`.

Models ranked by test ROC AUC (80/20 train-test split):

| Model | ROC AUC | Train time (s) |
|-------|---------|-----------------|
| LightGBM (Tuned) | 0.969 | 830.1 |
| CatBoost | 0.966 | 199.3 |
| XGBoost | 0.966 | 21.4 |
| LightGBM | 0.965 | 14.1 |
| Random Forest | 0.961 | 102.8 |
| Gradient Boosting | 0.955 | 360.5 |
| AdaBoost | 0.943 | 84.0 |
| Logistic Regression | 0.934 | 62.3 |
| Decision Tree | 0.784 | 25.2 |

**Best model configuration (LightGBM):**
- `n_estimators`: 695
- `num_leaves`: 109
- `learning_rate`: 0.07458
- `subsample`: 0.7713
- `colsample_bytree`: 0.5220
- `min_child_samples`: 90
- `reg_alpha`: 2.23e-06
- `reg_lambda`: 4.9296

## Key findings

- `duration` (last contact length) correlates with the target at 0.52 — by far the strongest single feature here. It's also only known after the call ends, so a real-world version of this model couldn't use it to decide whether to make the call in the first place.
- Tuning LightGBM (20 trials, ~14 minutes) gave a real lift: 0.965 → 0.969 ROC AUC, one of the bigger tuning wins across these episodes.
- The 88/12 imbalance means balanced accuracy (0.83) is notably lower than raw accuracy (0.94) for every model — ROC AUC is the metric that actually reflects performance here.
