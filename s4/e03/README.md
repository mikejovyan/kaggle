# Steel Plate Defect Prediction

**Competition:** [Steel Plate Defect Prediction](https://www.kaggle.com/competitions/playground-series-s4e3)

A multi-class classification project predicting the type of fault on a steel plate. Submissions are evaluated using the mean ROC AUC across 7 fault categories (`Pastry`, `Z_Scratch`, `K_Scatch`, `Stains`, `Dirtiness`, `Bumps`, `Other_Faults`), with one AUC computed per category and then averaged.

## Dataset

```bash
kaggle competitions download -c playground-series-s4e3
```

- Training samples: 19,219
- Target classes: Other_Faults (34%), Bumps (25%), K_Scatch (18%), Pastry (8%), Z_Scratch (6%), No_Defect (4%), Stains (3%), Dirtiness (3%)
- Features: 27 total
  - Numerical: 24 (`Edges_Index`, `Edges_X_Index`, `Edges_Y_Index`, `Empty_Index`, `Length_of_Conveyer`, `Log_X_Index`, `Log_Y_Index`, `LogOfAreas`, `Luminosity_Index`, `Maximum_of_Luminosity`, `Minimum_of_Luminosity`, `Orientation_Index`, `Outside_X_Index`, `Pixels_Areas`, `SigmoidOfAreas`, `Square_Index`, `Steel_Plate_Thickness`, `Sum_of_Luminosity`, `X_Maximum`, `X_Minimum`, `X_Perimeter`, `Y_Maximum`, `Y_Minimum`, `Y_Perimeter`)
  - Ordinal: 3 (`Outside_Global_Index`, `TypeOfSteel_A300`, `TypeOfSteel_A400`)
- Missing values: none

## Results

The Kaggle submission achieved a private score of `0.88684`, compared to the [leaderboard](https://www.kaggle.com/competitions/playground-series-s4e3/leaderboard) top score of `0.88977`.

Models ranked by test ROC AUC (80/20 train-test split):

| Model | ROC AUC | Train time (s) |
|-------|---------|----------------|
| XGBoost (Tuned) | 0.822 | 670.8 |
| Gradient Boosting | 0.816 | 157.4 |
| LightGBM | 0.811 | 4.1 |
| CatBoost | 0.809 | 72.9 |
| XGBoost | 0.805 | 5.3 |
| Random Forest | 0.799 | 17.3 |
| Logistic Regression | 0.795 | 11.6 |
| AdaBoost | 0.737 | 4.4 |
| Decision Tree | 0.637 | 1.5 |
| SVC | Too slow | - |

**Best model configuration (XGBoost):**
- `n_estimators`: 612
- `max_depth`: 5
- `learning_rate`: 0.01585
- `subsample`: 0.6966
- `colsample_bytree`: 0.5083
- `min_child_weight`: 7
- `gamma`: 6.56e-06

## Key findings

- Gradient boosting algorithms dominated, with XGBoost (tuned), Gradient Boosting, LightGBM, and CatBoost all achieving ROC AUC above 0.809
- Tuning XGBoost improved ROC AUC from 0.805 to 0.822, a meaningful gain on this smaller dataset
- 21 rows had multiple defects; these were resolved by assigning K_Scatch as the primary class
- Target is heavily imbalanced (Other_Faults 34% vs Dirtiness 2.5%), though boosting methods handled this well without resampling

