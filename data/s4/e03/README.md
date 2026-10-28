# Steel Plate Defect Prediction

**Competition:** [Steel Plate Defect Prediction](https://www.kaggle.com/competitions/playground-series-s4e3)

A multi-class classification project predicting the type of fault on a steel plate. Submissions are evaluated using the mean ROC AUC across 7 fault categories (`Pastry`, `Z_Scratch`, `K_Scatch`, `Stains`, `Dirtiness`, `Bumps`, `Other_Faults`), with one AUC computed per category and then averaged.

## Dataset

```bash
kaggle competitions download -c playground-series-s4e3
```

- Training samples: 19,219
- Target classes: No_Defect (4%), Pastry (8%), Z_Scratch (6%), K_Scatch (18%), Stains (3%), Dirtiness (3%), Bumps (25%), Other_Faults (34%)
- Features: 27 total
  - Numerical: 27 (`Edges_Index`, `Edges_X_Index`, `Edges_Y_Index`, `Empty_Index`, `Length_of_Conveyer`, `Log_X_Index`, `Log_Y_Index`, `LogOfAreas`, `Luminosity_Index`, `Maximum_of_Luminosity`, `Minimum_of_Luminosity`, `Orientation_Index`, `Outside_Global_Index`, `Outside_X_Index`, `Pixels_Areas`, `SigmoidOfAreas`, `Square_Index`, `Steel_Plate_Thickness`, `Sum_of_Luminosity`, `TypeOfSteel_A300`, `TypeOfSteel_A400`, `X_Maximum`, `X_Minimum`, `X_Perimeter`, `Y_Maximum`, `Y_Minimum`, `Y_Perimeter`)
- Missing values: none

## Results

The Kaggle submission achieved a private score of `0.88638`, compared to the [leaderboard](https://www.kaggle.com/competitions/playground-series-s4e3/leaderboard) top score of `0.88977`.

Models ranked by test ROC AUC (80/20 train-test split):

| Model | ROC AUC | Train time (s) |
|-------|---------|-----------------|
| XGBoost (Tuned) | 0.821 | 960.0 |
| Gradient Boosting | 0.816 | 159.3 |
| CatBoost | 0.811 | 76.8 |
| LightGBM | 0.808 | 5.0 |
| XGBoost | 0.803 | 10.3 |
| Random Forest | 0.803 | 16.0 |
| Logistic Regression | 0.795 | 58.4 |
| AdaBoost | 0.737 | 4.2 |
| Decision Tree | 0.638 | 1.4 |

**Best model configuration (XGBoost):**
- `n_estimators`: 736
- `max_depth`: 4
- `learning_rate`: 0.01590
- `subsample`: 0.6555
- `colsample_bytree`: 0.8867
- `min_child_weight`: 8
- `gamma`: 0.08375

## Key findings

- This is actually a multi-label problem (a plate can have more than one defect), collapsed into a single multiclass target by picking `K_Scatch` for the 21 double-defect rows. The submission's 7 probability columns are then filled directly from that one model's softmax output (classes 1–7, skipping `No_Defect`) rather than training 7 separate binary classifiers — a neat shortcut since multi-defect rows are rare (0.1% of the data).
