# Cirrhosis Outcomes Classification

**Competition:** [Multi-Class Prediction of Cirrhosis Outcomes](https://www.kaggle.com/competitions/playground-series-s3e26)

A multi-class classification project predicting cirrhosis patient outcomes (C, CL, or D). Submissions are evaluated using the multi-class logarithmic loss against the observed target `Status`.

## Dataset

```bash
kaggle competitions download -c playground-series-s3e26
```

- Training samples: 7,905
- Features: 18 total
  - Numerical: 11 features
  - Nominal: 2 features
  - Ordinal: 5 features
- Target classes: Status_C (62.8%), Status_D (33.7%), Status_CL (3.5%)

## Results

The Kaggle submission achieved a private score of `0.42704`, compared to the [leaderboard](https://www.kaggle.com/competitions/playground-series-s3e26/leaderboard) top score of `0.39104`.

Models ranked by test log loss (80/20 train-test split):

| Model | Log Loss | Train Time (s) |
|-------|----------|----------------|
| Gradient Boosting (Tuned) | 0.436 | 1428.9 |
| Gradient Boosting | 0.439 | 13.1 |
| LightGBM | 0.462 | 1.4 |
| CatBoost | 0.465 | 16.2 |
| Random Forest | 0.480 | 3.8 |
| SVC | 0.506 | 16.5 |
| XGBoost | 0.508 | 4.6 |
| Logistic Regression | 0.526 | 3.2 |
| AdaBoost | 1.003 | 1.4 |
| Decision Tree | 9.644 | 0.4 |

**Best Model Configuration (Gradient Boosting):**
- `n_estimators`: 220
- `learning_rate`: 0.02224
- `max_depth`: 5
- `subsample`: 0.530
- `min_samples_split`: 11
- `min_samples_leaf`: 2

## Key Findings

- Gradient boosting algorithms dominated, with tuned GB achieving the lowest log loss (0.433)
- Default LightGBM was competitive despite near-instant training time (1.2s vs 1811s for tuned GB)
- AdaBoost and Decision Tree performed poorly on log loss, suggesting poor probability calibration
- SVC completed evaluation on this smaller dataset (16.5s) and placed between Random Forest and XGBoost
