# Regression of Used Car Prices

**Competition:** [Regression of Used Car Prices](https://www.kaggle.com/competitions/playground-series-s4e9)

A regression project predicting the price of used cars from vehicle attributes and specifications. Submissions are evaluated using root mean squared error between the predicted value and the observed target `price`.

## Dataset

```bash
kaggle competitions download -c playground-series-s4e9
```

- Training samples: 188,533
- Target: mean ± std = 42,390 ± 48,695, range [2,000, 500,000]
- Features: 15 total
  - Numerical: 6 (`cylinders`, `displacement`, `horsepower`, `milage`, `model_year`, `transmission_speed`)
  - Nominal: 4 (`ext_col`, `fuel_type`, `int_col`, `transmission`)
  - Ordinal: 3 (`accident`, `clean_title`, `turbo`)
  - High-cardinality: 2 (`brand`, `model`)
- Missing values: present in 3 features (up to 21,419 missing per feature)

## Results

The Kaggle submission achieved a private score of `63289.40664`, compared to the [leaderboard](https://www.kaggle.com/competitions/playground-series-s4e9/leaderboard) top score of `62917.05988`.

Models ranked by test RMSE (80/20 train-test split):

| Model | RMSE | Train time (s) |
|-------|------|----------------|
| LightGBM (Tuned) | 39583.810 | 5848.2 |
| LightGBM | 39611.211 | 367.9 |
| CatBoost | 39839.997 | 1542.5 |
| XGBoost | 40216.352 | 480.1 |
| Ridge | 41003.773 | 246.0 |
| SVR | Too slow | - |
| Random Forest | Too slow | - |
| Decision Tree | Too slow | - |
| Gradient Boosting | Too slow | - |
| AdaBoost | Too slow | - |

**Best model configuration (LightGBM):**
- `n_estimators`: 374
- `num_leaves`: 82
- `learning_rate`: 0.01175
- `subsample`: 0.6456
- `colsample_bytree`: 0.8059
- `min_child_samples`: 45
- `reg_alpha`: 4.26e-06
- `reg_lambda`: 1.98e-05

## Key findings

- Gradient boosting models (LightGBM, CatBoost, XGBoost) all ranked within 633 RMSE of each other; Ridge lagged by ~1,400 RMSE, reflecting the non-linear price relationships
- The raw `engine` field was parsed to extract five structured features: cylinders, displacement, horsepower, turbo count, and transmission speed
- High-cardinality `brand` (57 values) and `model` (1,897 values) features were encoded with target encoding
- Tuning LightGBM over 5 Optuna trials reduced RMSE from 39,611 to 39,584 — a marginal gain given the limited search budget
- Train and test distributions were indistinguishable (adversarial AUC = 0.50), indicating no distribution shift
