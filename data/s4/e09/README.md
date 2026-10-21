# Regression of Used Car Prices

**Competition:** [Regression of Used Car Prices](https://www.kaggle.com/competitions/playground-series-s4e9)

A regression project predicting the price of used cars from vehicle attributes and specifications. Submissions are evaluated using root mean squared error between the predicted value and the observed target `price`.

## Dataset

```bash
kaggle competitions download -c playground-series-s4e9
```

- Training samples: 188,533
- Target: mean ± std = 43,880 ± 78,820, range [2,000, 2,954,000]
- Features: 15 total
  - Numerical: 6 (`cylinders`, `displacement`, `horsepower`, `milage`, `model_year`, `transmission_speed`)
  - Nominal: 4 (`ext_col`, `fuel_type`, `int_col`, `transmission`)
  - Ordinal: 3 (`accident`, `clean_title`, `turbo`)
  - High-cardinality: 2 (`brand`, `model`)
- Missing values: present in 3 raw features (up to 21,419 missing per feature); `cylinders`, `displacement`, `horsepower`, and `transmission_speed` are also derived from `engine`/`transmission` text and can be null when unparseable

## Results

The Kaggle submission achieved a private score of `63310.90210`, compared to the [leaderboard](https://www.kaggle.com/competitions/playground-series-s4e9/leaderboard) top score of `62917.05988`.

Models ranked by test RMSE (80/20 train-test split):

| Model | RMSE | Train time (s) |
|-------|------|-----------------|
| LightGBM (Tuned) | 67823.5 | 792.1 |
| Gradient Boosting | 68033.1 | 78.2 |
| LightGBM | 68175.1 | 3.6 |
| CatBoost | 68560.8 | 27.7 |
| Ridge | 69118.3 | 2.0 |
| AdaBoost | 69141.0 | 9.8 |
| XGBoost | 70294.8 | 4.7 |
| Decision Tree | 102364.6 | 8.8 |

**Best model configuration (LightGBM):**
- `n_estimators`: 656
- `num_leaves`: 100
- `learning_rate`: 0.00562
- `subsample`: 0.8982
- `colsample_bytree`: 0.5864
- `min_child_samples`: 161
- `reg_alpha`: 0.2799
- `reg_lambda`: 1.77e-06

## Key findings

- This one needed far more manual feature engineering than most episodes: the free-text `engine` field had to be regex-parsed into cylinders/displacement/horsepower/turbo, and the hundreds of raw color strings needed a hand-built lookup table to map down to a dozen main colors.
- Despite all that cleanup, R² stayed low (0.11–0.17) for every model — most of the price variance still isn't explained by these features, likely because a handful of extremely expensive listings skew the long-tailed target.
- 50 rounds of tuning only moved LightGBM from 68,175 to 67,824 RMSE, for 13 extra minutes of training — the manual feature engineering mattered more than tuning here.
