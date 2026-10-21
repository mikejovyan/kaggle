# Backpack Prediction Challenge

**Competition:** [Backpack Prediction Challenge](https://www.kaggle.com/competitions/playground-series-s5e2)

A regression project predicting the price of backpacks based on brand, material, and physical attributes. Submissions are evaluated using root mean squared error between the predicted value and the observed target `Price`.

## Dataset

```bash
kaggle competitions download -c playground-series-s5e2
```

- Training samples: 300,000
- Target: mean ± std = 81.41 ± 39.04, range [15, 150]
- Features: 9 total
  - Numerical: 2 (`Compartments`, `Weight Capacity (kg)`)
  - Nominal: 4 (`Brand`, `Color`, `Material`, `Style`)
  - Ordinal: 3 (`Laptop Compartment`, `Size`, `Waterproof`)
- Missing values: present in 7 features (up to 9,950 missing per feature)

## Results

The Kaggle submission achieved a private score of `38.92943`, compared to the [leaderboard](https://www.kaggle.com/competitions/playground-series-s5e2/leaderboard) top score of `38.61628`.

Models ranked by test RMSE (80/20 train-test split):

| Model | RMSE | Train time (s) |
|-------|------|-----------------|
| Gradient Boosting (Tuned) | 38.914 | 3477.0 |
| Gradient Boosting | 38.918 | 90.0 |
| Ridge | 38.922 | 2.7 |
| AdaBoost | 38.927 | 12.7 |
| LightGBM | 38.932 | 4.1 |
| CatBoost | 39.008 | 37.6 |
| XGBoost | 39.115 | 6.3 |
| Decision Tree | 55.825 | 9.8 |

**Best model configuration (Gradient Boosting):**
- `n_estimators`: 378
- `learning_rate`: 0.01525
- `max_depth`: 4
- `subsample`: 0.5665
- `min_samples_split`: 13
- `min_samples_leaf`: 8

## Key findings

- Every model scored almost the same (RMSE 38.9–39.1, R² near zero) regardless of approach, from Ridge to boosted trees. Brand, material, size, and the other backpack attributes barely relate to price at all — this looks like one of the "no real signal" Playground datasets.
- Tuning Gradient Boosting for an hour (3,477s) gained almost nothing over the 90-second default run.
- `training_extra.csv` sits in the data folder but isn't used by this notebook — only `train.csv` was loaded.
