# Regression with an Abalone Dataset

**Competition:** [Regression with an Abalone Dataset](https://www.kaggle.com/competitions/playground-series-s4e4)

A regression project predicting the age (number of rings) of abalones from physical measurements. Submissions are evaluated using root mean squared logarithmic error between the predicted value and the observed target `Rings`.

## Dataset

```bash
kaggle competitions download -c playground-series-s4e4
```

- Training samples: 90,615
- Target: mean ± std = 9.70 ± 3.18, range [1, 29]
- Features: 8 total
  - Numerical: 7 (`Diameter`, `Height`, `Length`, `Shell weight`, `Whole weight`, `Whole weight.1`, `Whole weight.2`)
  - Nominal: 1 (`Sex`)
- Missing values: none

## Results

The Kaggle submission achieved a private score of `0.14686`, compared to the [leaderboard](https://www.kaggle.com/competitions/playground-series-s4e4/leaderboard) top score of `0.14374`.

Models ranked by test RMSLE (80/20 train-test split):

| Model | RMSLE | Train time (s) |
|-------|-------|----------------|
| LightGBM (Tuned) | 0.151 | 636.3 |
| CatBoost | 0.152 | 11.1 |
| LightGBM | 0.153 | 0.9 |
| XGBoost | 0.153 | 0.9 |
| Gradient Boosting | 0.156 | 24.1 |
| Random Forest | 0.156 | 101.6 |
| SVR | 0.156 | 704.4 |
| Ridge | 0.167 | 0.2 |
| AdaBoost | 0.211 | 8.5 |
| Decision Tree | 0.217 | 1.7 |

**Best model configuration (LightGBM):**
- `n_estimators`: 786
- `num_leaves`: 50
- `learning_rate`: 0.03294
- `subsample`: 0.5672
- `colsample_bytree`: 0.5993
- `min_child_samples`: 72
- `reg_alpha`: 3.43e-05
- `reg_lambda`: 8.886

## Key findings

- Gradient boosting models (LightGBM, CatBoost, XGBoost) dominated, all within 0.004 RMSLE of each other with default parameters
- Tuning LightGBM improved RMSLE from 0.153 to 0.151, a modest but consistent gain
- Train and test distributions were indistinguishable (adversarial AUC = 0.50), indicating no distribution shift
- SVR matched boosting models in accuracy but at 704 seconds was by far the slowest baseline model
