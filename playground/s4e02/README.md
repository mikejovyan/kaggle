# Multi-Class Prediction of Obesity Risk

**Competition:** [Multi-Class Prediction of Obesity Risk](https://www.kaggle.com/competitions/playground-series-s4e2)

A multi-class classification project predicting obesity risk levels based on eating habits and physical condition. Submissions are evaluated using accuracy score between the predicted value and the observed target `NObeyesdad`.

## Dataset

```bash
kaggle competitions download -c playground-series-s4e2
```

- Training samples: 20,758
- Target classes: Obesity_Type_III (19%), Obesity_Type_II (16%), Normal_Weight (15%), Obesity_Type_I (14%), Insufficient_Weight (12%), Overweight_Level_II (12%), Overweight_Level_I (12%)
- Features: 16 total
  - Numerical: 8 (`Age`, `CH2O`, `FAF`, `FCVC`, `Height`, `NCP`, `TUE`, `Weight`)
  - Nominal: 1 (`MTRANS`)
  - Ordinal: 7 (`CAEC`, `CALC`, `family_history_with_overweight`, `FAVC`, `Gender`, `SCC`, `SMOKE`)
- Missing values: none

## Results

The Kaggle submission achieved a private score of `0.90796`, compared to the [leaderboard](https://www.kaggle.com/competitions/playground-series-s4e2/leaderboard) top score of `0.91157`.

Models ranked by test accuracy (80/20 train-test split):

| Model | Accuracy | Train time (s) |
|-------|----------|----------------|
| LightGBM (Tuned) | 0.908 | 1469.4 |
| Gradient Boosting | 0.905 | 52.8 |
| LightGBM | 0.904 | 3.4 |
| CatBoost | 0.902 | 31.7 |
| XGBoost | 0.900 | 3.3 |
| Random Forest | 0.894 | 6.4 |
| Logistic Regression | 0.856 | 7.5 |
| Decision Tree | 0.843 | 0.5 |
| AdaBoost | 0.693 | 2.2 |

**Best model configuration (LightGBM):**
- `n_estimators`: 696
- `num_leaves`: 54
- `learning_rate`: 0.00750
- `subsample`: 0.894
- `colsample_bytree`: 0.522
- `min_child_samples`: 45
- `reg_alpha`: 0.129
- `reg_lambda`: 0.00317

## Key findings

- LightGBM achieved the highest default accuracy (0.906) and improved marginally with tuning (0.908) at a significant time cost (3.4s vs 1469.4s)
- Gradient Boosting matched LightGBM (0.905) with far less tuning effort
- Tree-based ensemble methods (LightGBM, GB, LGBMClassifier, CatBoost, XGBoost) all performed similarly, clustered between 0.900–0.908
- Train and test distributions are statistically indistinguishable (validation score 0.50)
- The 7-class target is roughly balanced, making accuracy a reliable evaluation metric
