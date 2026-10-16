# Regression with an Insurance Dataset

**Competition:** [Regression with an Insurance Dataset](https://www.kaggle.com/competitions/playground-series-s4e12)

A regression project predicting the insurance premium amount for policyholders based on demographic and policy attributes. Submissions are evaluated using root mean squared logarithmic error between the predicted value and the observed target `Premium Amount`.

## Dataset

```bash
kaggle competitions download -c playground-series-s4e12
```

- Training samples: 1,200,000
- Target: mean ± std = 1,102.54 ± 865.00, range [20, 4,999]
- Features: 19 total
  - Numerical: 8 (`Age`, `Annual Income`, `Credit Score`, `Health Score`, `Insurance Duration`, `Number of Dependents`, `Previous Claims`, `Vehicle Age`)
  - Nominal: 3 (`Marital Status`, `Occupation`, `Property Type`)
  - Ordinal: 6 (`Customer Feedback`, `Education Level`, `Exercise Frequency`, `Gender`, `Location`, `Policy Type`, `Smoking Status`)
  - High-cardinality: 1 (`Policy Start Date`)
- Missing values: present in 9 features (up to 364,029 missing per feature)

## Results

The Kaggle submission achieved a private score of `1.13226`, compared to the [leaderboard](https://www.kaggle.com/competitions/playground-series-s4e12/leaderboard) top score of `1.01706`.

Models ranked by test RMSLE (80/20 train-test split):

| Model | RMSLE | Train time (s) |
|-------|-------|-----------------|
| LightGBM (Tuned) | 1.133 | 1263.5 |
| LightGBM | 1.134 | 35.9 |
| CatBoost | 1.134 | 202.7 |
| XGBoost | 1.136 | 32.2 |
| Ridge | 1.165 | 21.4 |
| AdaBoost | 1.279 | 162.8 |
| Decision Tree | 1.505 | 113.3 |

**Best model configuration (LightGBM):**
- `n_estimators`: 260
- `num_leaves`: 121
- `learning_rate`: 0.02391
- `subsample`: 0.8682
- `colsample_bytree`: 0.9885
- `min_child_samples`: 49
- `reg_alpha`: 4.76e-07
- `reg_lambda`: 2.86e-07

## Key findings

- LightGBM led all models; tuning with Optuna (20 trials) improved RMSLE only marginally (1.134 → 1.133) at a large time cost (35.9s vs 1263.5s)
- CatBoost and XGBoost scored within 0.003 RMSLE of LightGBM, while Ridge lagged by 0.031, reflecting strong non-linear relationships in the data
- AdaBoost and Decision Tree performed poorly, suggesting these methods struggle with the noisy, weakly-correlated features in this dataset
- Missing values were pervasive — `Previous Claims` and `Occupation` each had over 30% missing, likely structural absences rather than random gaps; missingness itself was encoded as a feature for `Annual Income`, `Customer Feedback`, and `Health Score`
- `Policy Start Date` was decomposed into seven features (year, day-of-week, Unix timestamp, and cyclical month/day sin-cos encodings) to capture temporal patterns
- Train and test distributions are indistinguishable (adversarial balanced accuracy = 0.50), indicating no distribution shift
- R² remained low across all models (0.06–0.07), indicating most of the variance in premium amount is not explained by the available features
