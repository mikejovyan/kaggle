# Predicting Optimal Fertilizers

**Competition:** [Predicting Optimal Fertilizers](https://www.kaggle.com/competitions/playground-series-s5e6)

A multi-class classification project predicting the best fertilizer for a given set of soil and crop conditions. Submissions are evaluated using Mean Average Precision @ 3 (MAP@3) between the top 3 predicted fertilizers and the observed target `Fertilizer Name`.

## Dataset

```bash
kaggle competitions download -c playground-series-s5e6
```

- Training samples: 750,000
- Target classes: 14-35-14 (15%), 10-26-26 (15%), 17-17-17 (15%), 28-28 (15%), 20-20 (15%), DAP (13%), Urea (12%)
- Features: 8 total
  - Numerical: 6 (`Humidity`, `Moisture`, `Nitrogen`, `Phosphorous`, `Potassium`, `Temparature`)
  - Nominal: 2 (`Crop Type`, `Soil Type`)
- Missing values: none

## Results

The Kaggle submission achieved a private score of `0.34638`, compared to the [leaderboard](https://www.kaggle.com/competitions/playground-series-s5e6/leaderboard) top score of `0.38652`.

Models ranked by test MAP@3 (80/20 train-test split):

| Model | MAP@3 | Train time (s) |
|-------|-------|-----------------|
| XGBoost (Tuned) | 0.342 | 2797.3 |
| XGBoost | 0.332 | 89.4 |
| CatBoost | 0.329 | 498.3 |
| LightGBM | 0.323 | 43.4 |
| Random Forest | 0.291 | 349.9 |
| Logistic Regression | 0.287 | 10.9 |
| AdaBoost | 0.279 | 56.1 |
| Decision Tree | 0.257 | 18.9 |

**Best model configuration (XGBoost):**
- `n_estimators`: 437
- `max_depth`: 10
- `learning_rate`: 0.06505
- `subsample`: 0.7993
- `colsample_bytree`: 0.5780
- `min_child_weight`: 2
- `gamma`: 2.92e-08

## Key findings

- ROC AUC never climbs above 0.582, even for the tuned XGBoost model — the soil/crop measurements barely relate to which fertilizer was actually used, so the ceiling here is set by weak signal, not model choice or tuning.
- Tuning XGBoost for just 10 trials took 46 minutes (2,797s) and only moved MAP@3 from 0.332 to 0.342 — a small gain for a large time cost.
