# Multi-Class Prediction of Obesity Risk

**Competition:** [Multi-Class Prediction of Obesity Risk](https://www.kaggle.com/competitions/playground-series-s4e2)

A multi-class classification project predicting a person's obesity risk category based on eating habits and physical condition. Submissions are evaluated using accuracy score between the predicted value and the observed target `NObeyesdad`.

## Dataset

```bash
kaggle competitions download -c playground-series-s4e2
```

- Training samples: 20,758
- Target classes: Obesity_Type_III (19%), Obesity_Type_II (16%), Normal_Weight (15%), Obesity_Type_I (14%), Insufficient_Weight (12%), Overweight_Level_II (12%), Overweight_Level_I (12%)
- Features: 16 total
  - Numerical: 8 (`Age`, `CH2O`, `FAF`, `FCVC`, `Height`, `NCP`, `TUE`, `Weight`)
  - Nominal: 1 (`MTRANS`)
  - Ordinal: 7 (`CAEC`, `CALC`, `family_history`, `FAVC`, `Gender`, `SCC`, `SMOKE`)
- Missing values: none

## Results

The Kaggle submission achieved a private score of `0.90570`, compared to the [leaderboard](https://www.kaggle.com/competitions/playground-series-s4e2/leaderboard) top score of `0.91157`.

Models ranked by test accuracy (80/20 train-test split):

| Model | Accuracy | Train time (s) |
|-------|----------|-----------------|
| LightGBM (Tuned) | 0.908 | 932.9 |
| Gradient Boosting | 0.905 | 52.2 |
| LightGBM | 0.905 | 3.9 |
| XGBoost | 0.904 | 5.1 |
| CatBoost | 0.902 | 33.2 |
| Random Forest | 0.894 | 5.7 |
| Logistic Regression | 0.856 | 26.8 |
| Decision Tree | 0.845 | 0.5 |
| AdaBoost | 0.693 | 2.1 |

**Best model configuration (LightGBM):**
- `n_estimators`: 381
- `num_leaves`: 105
- `learning_rate`: 0.01184
- `subsample`: 0.5534
- `colsample_bytree`: 0.6906
- `min_child_samples`: 84
- `reg_alpha`: 1.71e-06
- `reg_lambda`: 2.86e-07

## Key findings

- The 7 classes are fairly even (12–19% each), with no rare class to drag things down — balanced accuracy (0.898) tracks plain accuracy (0.908) closely for the tuned model.
