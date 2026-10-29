# Diabetes Prediction Challenge

**Competition:** [Diabetes Prediction Challenge](https://www.kaggle.com/competitions/playground-series-s5e12)

A binary classification project predicting whether a patient has diabetes based on health and lifestyle data. Submissions are evaluated on area under the ROC curve between the predicted probability and the observed target `diagnosed_diabetes`.

## Dataset

```bash
kaggle competitions download -c playground-series-s5e12
```

- Training samples: 700,000
- Target classes: 0/No diabetes (38%), 1/Diabetes (62%)
- Features: 24 total
  - Numerical: 18 (`age`, `alcohol_consumption_per_week`, `bmi`, `cardiovascular_history`, `cholesterol_total`, `diastolic_bp`, `diet_score`, `family_history_diabetes`, `hdl_cholesterol`, `heart_rate`, `hypertension_history`, `ldl_cholesterol`, `physical_activity_minutes_per_week`, `screen_time_hours_per_day`, `sleep_hours_per_day`, `systolic_bp`, `triglycerides`, `waist_to_hip_ratio`)
  - Nominal: 4 (`employment_status`, `ethnicity`, `gender`, `smoking_status`)
  - Ordinal: 2 (`education_level`, `income_level`)
- Missing values: none

## Results

The Kaggle submission achieved a private score of `0.69444`, compared to the [leaderboard](https://www.kaggle.com/competitions/playground-series-s5e12/leaderboard) top score of `0.70504`.

Models ranked by test ROC AUC (80/20 train-test split):

| Model | ROC AUC | Train time (s) |
|-------|---------|-----------------|
| LightGBM (Tuned) | 0.727 | 872.6 |
| CatBoost | 0.726 | 154.7 |
| XGBoost | 0.724 | 21.3 |
| LightGBM | 0.722 | 15.0 |
| Gradient Boosting | 0.705 | 520.9 |
| Logistic Regression | 0.694 | 17.6 |
| AdaBoost | 0.694 | 108.4 |
| Random Forest | 0.691 | 461.4 |
| Decision Tree | 0.561 | 42.1 |

**Best model configuration (LightGBM):**
- `n_estimators`: 383
- `num_leaves`: 79
- `learning_rate`: 0.04491
- `subsample`: 0.8331
- `colsample_bytree`: 0.5277
- `min_child_samples`: 34
- `reg_alpha`: 2.48e-08
- `reg_lambda`: 2.01e-08

## Key findings

- Despite a full set of clinically relevant risk factors (BMI, cholesterol, blood pressure, family/cardiovascular/hypertension history), ROC AUC tops out around 0.73 — modest for a diagnosis task with this much medical signal available, and 700k rows didn't push it any higher.
- The local test score (0.727) is noticeably higher than the Kaggle private score (0.694), a bigger gap than most episodes. The adversarial validation score (0.55) also came closer to the "distinguishable" threshold (0.6) than usual, consistent with some real train/test difference even though it didn't cross the line.
- The target is majority-diabetes (62%) rather than the rare-disease imbalance you'd typically expect for this kind of prediction task.
