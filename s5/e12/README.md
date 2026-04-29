# Diabetes Prediction Challenge

**Competition:** [Diabetes Prediction Challenge](https://www.kaggle.com/competitions/playground-series-s5e12)

A binary classification project predicting whether a patient has diabetes based on health and lifestyle data. Submissions are evaluated on area under the ROC curve between the predicted probability and the observed target `diagnosed_diabetes`.

## Dataset

```bash
kaggle competitions download -c playground-series-s5e12
```

- Training samples: 560,000
- Target classes: 0/No diabetes (37.67%), 1/Diabetes (62.33%)
- Features: 24 total
  - Numerical: 15 (`age`, `alcohol_consumption_per_week`, `bmi`, `cholesterol_total`, `diastolic_bp`, `diet_score`, `hdl_cholesterol`, `heart_rate`, `ldl_cholesterol`, `physical_activity_minutes_per_week`, `screen_time_hours_per_day`, `sleep_hours_per_day`, `systolic_bp`, `triglycerides`, `waist_to_hip_ratio`)
  - Nominal: 4 (`ethnicity`, `employment_status`, `gender`, `smoking_status`)
  - Ordinal: 5 (`cardiovascular_history`, `education_level`, `family_history_diabetes`, `hypertension_history`, `income_level`)
- Missing values: none

## Results

The Kaggle submission achieved a private score of `0.69501`, compared to the [leaderboard](https://www.kaggle.com/competitions/playground-series-s5e12/leaderboard) top score of `0.70504`.

Models ranked by test ROC AUC (80/20 train-test split):

| Model | ROC AUC | Train time (s) |
|-------|---------|----------------|
| LightGBM (Tuned) | 0.728 | 1062.8 |
| CatBoost | 0.726 | 134.9 |
| XGBoost | 0.723 | 11.6 |
| LightGBM | 0.722 | 13.7 |
| Gradient Boosting | 0.705 | 466.7 |
| Logistic Regression | 0.694 | 10.9 |
| AdaBoost | 0.694 | 98.6 |
| Random Forest | 0.692 | 463.0 |
| Decision Tree | 0.560 | 38.5 |

**Best model configuration (LightGBM):**
- `n_estimators`: 953
- `num_leaves`: 31
- `learning_rate`: 0.04971
- `subsample`: 0.50859
- `colsample_bytree`: 0.51235
- `min_child_samples`: 98
- `reg_alpha`: 1.423e-08
- `reg_lambda`: 5.19864

## Key findings

- The dataset is imbalanced toward the positive class (62% diabetes), which inflates raw accuracy and makes ROC AUC a more reliable evaluation metric
- Tuning LightGBM with Optuna improved ROC AUC from 0.722 to 0.728, the largest gain among all models tested
- The boosting models (LightGBM, CatBoost, XGBoost) all outperform the ensemble methods (Random Forest, Gradient Boosting) by a significant margin on this dataset
- Local test scores (~0.728) are noticeably higher than the Kaggle private score (0.695), suggesting the synthetic dataset has distributional differences between train and test splits
- Adversarial validation (discriminator AUC = 0.61) detected train/test distribution shift which can cause generalisation issues but did not appear to impact results
