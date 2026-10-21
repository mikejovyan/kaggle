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

- R² never got above 0.07 for any model, even with 1.2 million rows to train on. More data didn't help — the features themselves barely predict premium amount, so the ceiling here is set by signal, not sample size.
- `Previous Claims` and `Occupation` were missing in over 30% of rows. That's too large to be random — more likely people who genuinely have no claims or no listed job — so missingness itself was kept as a feature rather than just imputed away.
- 20 rounds of tuning LightGBM (21 minutes) barely moved RMSLE (1.134 → 1.133). Not worth repeating on this dataset.
