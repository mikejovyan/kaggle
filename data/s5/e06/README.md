# Predicting Optimal Fertilizers

**Competition:** [Predicting Optimal Fertilizers](https://www.kaggle.com/competitions/playground-series-s5e6)

A multi-class classification project predicting the optimal fertilizer recommendation based on soil nutrient levels and environmental conditions. Submissions are evaluated using mean average precision at 3 (MAP@3) between the top-3 predicted fertilizers and the observed target `Fertilizer Name`.

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

The Kaggle submission achieved a private score of `0.34589`, compared to the [leaderboard](https://www.kaggle.com/competitions/playground-series-s5e6/leaderboard) top score of `0.38652`.

Models ranked by test MAP@3 (80/20 train-test split):

| Model | MAP@3 | Train time (s) |
|-------|-------|----------------|
| XGBoost (Tuned) | 0.342 | 1810.4 |
| XGBoost | 0.331 | 92.6 |
| CatBoost | 0.329 | 526.2 |
| LightGBM | 0.323 | 37.9 |
| Gradient Boosting | 0.309 | 1799.6 |
| Random Forest | 0.291 | 344.7 |
| Logistic Regression | 0.287 | 9.8 |
| AdaBoost | 0.279 | 58.2 |
| Decision Tree | 0.257 | 21.0 |

**Best model configuration (XGBoost):**
- `n_estimators`: 437
- `max_depth`: 10
- `learning_rate`: 0.06505
- `subsample`: 0.79933
- `colsample_bytree`: 0.57801
- `min_child_weight`: 2
- `gamma`: 2.915e-08

## Key findings

- The 7-class target is near-perfectly balanced (~13–15% each), making MAP@3 a reliable metric across all classes
- Tuning XGBoost with Optuna improved MAP@3 from 0.331 to 0.342, the largest gain among all models tested
- XGBoost outperforms other boosting algorithms (CatBoost 0.329, LightGBM 0.323) with default parameters
- Local test scores (~0.342) are lower than the Kaggle private score (0.365), suggesting the official test set is slightly easier than the local hold-out
- Train and test distributions are indistinguishable (validation score 0.50), indicating good dataset quality
