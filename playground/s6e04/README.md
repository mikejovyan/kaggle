# Predicting Irrigation Need

**Competition:** [Predicting Irrigation Need](https://www.kaggle.com/competitions/playground-series-s6e4)

A multi-class classification project predicting the irrigation need level for agricultural fields. Submissions are evaluated using balanced accuracy score between the predicted value and the observed target `Irrigation_Need`.

## Dataset

```bash
kaggle competitions download -c playground-series-s6e4
```

- Training samples: 630,000
- Target classes: Low (59%), Medium (38%), High (3%)
- Features: 19 total
  - Numerical: 11 (`Electrical_Conductivity`, `Field_Area_hectare`, `Humidity`, `Organic_Carbon`, `Previous_Irrigation_mm`, `Rainfall_mm`, `Soil_Moisture`, `Soil_pH`, `Sunlight_Hours`, `Temperature_C`, `Wind_Speed_kmh`)
  - Nominal: 6 (`Crop_Type`, `Irrigation_Type`, `Region`, `Season`, `Soil_Type`, `Water_Source`)
  - Ordinal: 2 (`Crop_Growth_Stage`, `Mulching_Used`)
- Missing values: none

## Results

The Kaggle submission achieved a private score of `TBD`, compared to the [leaderboard](https://www.kaggle.com/competitions/playground-series-s6e4/leaderboard) top score of `TBD`.

Models ranked by test balanced accuracy (80/20 train-test split):

| Model | Balanced Acc | Train time (s) |
|-------|-------------|----------------|
| LightGBM (Tuned) | 0.971 | 347.7 |
| LightGBM | 0.971 | 393.2 |
| XGBoost | 0.960 | 79.4 |
| CatBoost | 0.960 | 314.2 |
| Decision Tree | 0.946 | 46.3 |
| AdaBoost | 0.657 | 132.1 |
| Logistic Regression | 0.622 | 20.3 |
| SVC | Too slow | - |
| Random Forest | Too slow | - |
| Gradient Boosting | Too slow | - |

**Best model configuration (LightGBM):**
- `class_weight`: balanced
- `colsample_bytree`: 0.3068
- `learning_rate`: 0.04804
- `min_child_samples`: 124
- `n_estimators`: 2611
- `num_leaves`: 6
- `reg_alpha`: 2.4e-04
- `subsample`: 0.4107
- `subsample_freq`: 1

## Key findings

- LightGBM led all models at 0.971 balanced accuracy; XGBoost and CatBoost followed closely at 0.960
- AdaBoost (0.657) and Logistic Regression (0.622) struggled significantly with the three-class imbalanced target
- Decision Tree achieved a competitive 0.946 balanced accuracy despite being a single tree
- Severe class imbalance (High: 3.3%) was addressed via `class_weight="balanced"` in LightGBM