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

- LightGBM (tuned) led the comparison; tuning with Optuna (50 trials) improved RMSE from 68,175 to 67,824, a modest gain at a large time cost (3.6s vs 792.1s)
- Gradient Boosting was competitive with default parameters (68,033 RMSE) despite no tuning
- The raw `engine` field was parsed with regex to extract five structured features: `cylinders`, `displacement`, `horsepower`, `turbo` count, and `transmission_speed`; `transmission` text was also normalized into coarse categories (`AT`, `MT`, `CVT`, `DCT`, `Unknown`)
- Free-text `ext_col`/`int_col` values (hundreds of distinct strings) were mapped down to a small set of main colors via a manually curated lookup table
- High-cardinality `brand` (57 values) and `model` (1,897 values) features were encoded with target encoding
- R² remained low across all models (0.11–0.17), indicating substantial unexplained variance in price, likely driven by rare, extremely expensive listings in the long-tailed target distribution
- Train and test distributions were indistinguishable (adversarial balanced accuracy = 0.50), indicating no distribution shift
