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

The Kaggle submission achieved a private score of `0.14701`, compared to the [leaderboard](https://www.kaggle.com/competitions/playground-series-s4e4/leaderboard) top score of `0.14374`.

Models ranked by test RMSLE (80/20 train-test split):

| Model | RMSLE | Train time (s) |
|-------|-------|-----------------|
| LightGBM (Tuned) | 0.151 | 1038.4 |
| CatBoost | 0.152 | 17.3 |
| LightGBM | 0.153 | 1.3 |
| XGBoost | 0.153 | 1.2 |
| Gradient Boosting | 0.156 | 26.9 |
| Random Forest | 0.156 | 116.1 |
| SVR | 0.156 | 643.9 |
| Ridge | 0.167 | 0.2 |
| AdaBoost | 0.211 | 9.3 |
| Decision Tree | 0.217 | 1.8 |

**Best model configuration (LightGBM):**
- `n_estimators`: 767
- `num_leaves`: 124
- `learning_rate`: 0.01274
- `subsample`: 0.8933
- `colsample_bytree`: 0.5540
- `min_child_samples`: 76
- `reg_alpha`: 3.45e-05
- `reg_lambda`: 1.36e-08

## Key findings

- LightGBM, CatBoost, and XGBoost all landed within 0.004 RMSLE of each other by default — not much room to separate models on this dataset.
- 100 rounds of tuning only moved LightGBM from 0.153 to 0.151 RMSLE, for 17 minutes of extra training time. Probably not worth repeating.
- SVR matched the boosting models' accuracy but took over 10 minutes to train, versus LightGBM's 1.3 seconds.
