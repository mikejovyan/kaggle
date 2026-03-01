# Academic success classification

**Competition:** [Classification with an academic success dataset](https://www.kaggle.com/competitions/playground-series-s4e6)

A multi-class classification project predicting student academic outcomes (Dropout, Enrolled, or Graduate). Submissions are evaluated using the accuracy score against the observed target `Target`.

## Dataset

```bash
kaggle competitions download -c playground-series-s4e6
```

- Training samples: 76,518
- Features: 36 total
  - Numerical: 18 features
  - Nominal: 2 features
  - Ordinal: 16 features
- Target classes: Dropout (33.1%), Graduate (47.4%), Enrolled (19.5%)

## Results

The Kaggle submission achieved a private score of `0.83734`, compared to the [leaderboard](https://www.kaggle.com/competitions/playground-series-s4e6/leaderboard) top score of `0.84035`.

Models ranked by test accuracy (80/20 train-test split):

| Model | Accuracy | Train time (s) |
|-------|----------|----------------|
| LightGBM (Tuned) | 0.835 | 1190.3 |
| LightGBM | 0.834 | 4.9 |
| CatBoost | 0.833 | 56.6 |
| XGBoost | 0.832 | 5.8 |
| Gradient Boosting | 0.831 | 204.2 |
| Random Forest | 0.827 | 45.0 |
| Logistic Regression | 0.822 | 11.4 |
| AdaBoost | 0.811 | 14.6 |
| Decision Tree | 0.745 | 5.6 |
| SVC | Too slow | - |

**Best model configuration (LightGBM):**
- `n_estimators`: 976
- `learning_rate`: 0.02816
- `num_leaves`: 31
- `subsample`: 0.502
- `colsample_bytree`: 0.523
- `min_child_samples`: 105
- `reg_alpha`: 2.05e-08
- `reg_lambda`: 0.0555

## Key findings

- Gradient boosting algorithms (CatBoost, LightGBM, XGBoost, GB) significantly outperformed other methods
- Default LightGBM achieved 83.4% accuracy with minimal training time (4.9s)
- Optuna-tuned LightGBM outperformed CatBoost (83.5% vs 83.3% accuracy) at a much higher time cost (1190s vs 57s)
- SVC was excluded from full evaluation due to excessive training time on this dataset size
