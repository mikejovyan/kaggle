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

- All models except Decision Tree clustered within 0.2 RMSE of each other (38.9–39.1), suggesting the features carry very little signal for predicting price
- R² was near zero across all models (~0.0–0.002), indicating that brand, material, size, and physical attributes barely explain price variance — the target appears largely random given these inputs
- Tuning Gradient Boosting with Optuna (15 trials) improved RMSE from 38.918 to 38.914, a negligible gain at a substantial time cost (90s vs 3477s)
- Ridge matched the boosting models closely (38.922), consistent with the absence of useful non-linear structure
- Missing values were widespread across 7 of 9 features, with categorical columns losing up to 9,950 entries (~3.3% of training rows)
- Train and test distributions are indistinguishable (adversarial balanced accuracy = 0.50), indicating no distribution shift
- `training_extra.csv` is present in the data directory but unused by this notebook — only `train.csv` was used for training
