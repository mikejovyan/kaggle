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

The Kaggle submission achieved a private score of `0.96237`, compared to the [leaderboard](https://www.kaggle.com/competitions/playground-series-s6e4/leaderboard) top score of `0.98158`.

Models ranked by test balanced accuracy (80/20 train-test split):

| Model | Balanced Acc | Train time (s) |
|-------|--------------|-----------------|
| XGBoost | 0.961 | 46.7 |
| LightGBM (Tuned) | 0.960 | 2666.2 |
| CatBoost | 0.960 | 294.7 |
| LightGBM | 0.959 | 27.2 |
| Decision Tree | 0.946 | 34.6 |
| AdaBoost | 0.657 | 117.2 |
| Logistic Regression | 0.622 | 25.9 |

**Best model configuration (LightGBM):**
- `n_estimators`: 658
- `num_leaves`: 40
- `learning_rate`: 0.07996
- `subsample`: 0.9039
- `colsample_bytree`: 0.9611
- `min_child_samples`: 184
- `reg_alpha`: 0.8322
- `reg_lambda`: 0.8931

## Key findings

- XGBoost, tuned LightGBM, and CatBoost all landed within 0.001 of each other (0.960–0.961 balanced accuracy) — no clear winner among the top models.
- A single Decision Tree got surprisingly close to the boosted models (0.946). That suggests the classes split along a few simple rules rather than needing complex feature interactions.
- AdaBoost (0.657) and Logistic Regression (0.622) both collapsed on the rare "High" class (3.3% of rows), dragging their balanced accuracy well below the tree-based models.
- A side experiment with just 9 engineered threshold flags (see `notebook-explore.ipynb`) matched or beat the full 19-feature model (0.962–0.964 vs 0.959–0.961). Most of the signal here comes from a handful of coarse cutoffs, not the raw sensor readings.
