# Academic Success Classification

**Competition:** [Classification with an Academic Success Dataset](https://www.kaggle.com/competitions/playground-series-s4e6)

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

Kaggle submission: private score `0.83620`, [leaderboard](https://www.kaggle.com/competitions/playground-series-s4e6/leaderboard) `0.84035`.

Models ranked by test accuracy (80/20 train-test split):

| Model | Accuracy | Train Time (s) |
|-------|----------|----------------|
| CatBoost | 0.835 | 84.5 |
| LightGBM (Tuned) | 0.835 | 2346.1 |
| LightGBM | 0.833 | 7.2 |
| XGBoost | 0.832 | 17.3 |
| Gradient Boosting | 0.830 | 170.3 |
| Random Forest | 0.828 | 31.2 |
| Logistic Regression | 0.818 | 182.5 |
| AdaBoost | 0.809 | 13.1 |
| Decision Tree | 0.741 | 4.3 |
| SVC | Too slow | - |

**Best Model Configuration (LightGBM):**
- `n_estimators`: 821
- `learning_rate`: 0.0145
- `num_leaves`: 45
- `subsample`: 0.530
- `colsample_bytree`: 0.647
- `min_child_samples`: 137
- `reg_alpha`: 3.04e-07
- `reg_lambda`: 3.63e-08

## Key Findings

- Gradient boosting algorithms (CatBoost, LightGBM, XGBoost, GB) significantly outperformed other methods
- Default LightGBM achieved 83.3% accuracy with minimal training time (7.2s)
- Optuna-tuned LightGBM matched CatBoost (83.5% accuracy) at a much higher time cost (2346s vs 85s)
- SVC was excluded from full evaluation due to excessive training time on this dataset size
