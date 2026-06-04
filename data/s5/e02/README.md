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
- Missing values: present in 8 features (up to 9,950 missing per feature)

## Results

The Kaggle submission achieved a private score of `38.92790`, compared to the [leaderboard](https://www.kaggle.com/competitions/playground-series-s5e2/leaderboard) top score of `38.61628`.

Models ranked by test RMSE (80/20 train-test split):

| Model | RMSE | Train time (s) |
|-------|------|----------------|
| Gradient Boosting (Tuned) | 38.915 | 6646.0 |
| Gradient Boosting | 38.918 | 79.1 |
| Ridge | 38.922 | 2.8 |
| AdaBoost | 38.927 | 11.4 |
| LightGBM | 38.932 | 3.8 |
| CatBoost | 39.008 | 32.6 |
| XGBoost | 39.093 | 5.4 |
| Decision Tree | 55.900 | 10.7 |
| SVR | Too slow | - |
| Random Forest | Too slow | - |

**Best model configuration (Gradient Boosting):**
- `n_estimators`: 435
- `learning_rate`: 0.009379
- `max_depth`: 6
- `subsample`: 0.6383
- `min_samples_split`: 9
- `min_samples_leaf`: 7

## Key findings

- All models except Decision Tree clustered within 0.2 RMSE of each other (38.9–39.1), suggesting the features carry very little signal for predicting price
- R² was near zero across all models (~0.001), indicating that brand, material, size, and physical attributes barely explain price variance — the target appears largely random given these inputs
- Tuning Gradient Boosting with Optuna (20 trials) improved RMSE from 38.918 to 38.915, a negligible gain at substantial time cost (79s vs 6646s)
- Ridge matched the boosting models closely (38.922), consistent with the absence of useful non-linear structure
- Missing values were widespread across 8 of 9 features, with categorical columns losing up to 9,950 entries (~3.3% of training rows)
