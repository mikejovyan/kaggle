# Binary Prediction of Smoker Status using Bio-Signals

**Competition:** [Binary Prediction of Smoker Status using Bio-Signals](https://www.kaggle.com/competitions/playground-series-s3e24)

A binary classification project predicting whether a patient is a smoker based on bio-signal data. Submissions are evaluated on area under the ROC curve between the predicted probability and the observed target `smoking`.

## Dataset

```bash
kaggle competitions download -c playground-series-s3e24
```

- Training samples: 159,256
- Target classes: 0/Not smoker (56%), 1/Smoker (44%)
- Features: 22 total
  - Numerical: 18 (`age`, `ALT`, `AST`, `Cholesterol`, `eyesight(left)`, `eyesight(right)`, `fasting blood sugar`, `Gtp`, `HDL`, `height(cm)`, `hemoglobin`, `LDL`, `relaxation`, `serum creatinine`, `systolic`, `triglyceride`, `waist(cm)`, `weight(kg)`)
  - Ordinal: 4 (`dental caries`, `hearing(left)`, `hearing(right)`, `Urine protein`)
- Missing values: none

## Results

The Kaggle submission achieved a private score of `0.87111`, compared to the [leaderboard](https://www.kaggle.com/competitions/playground-series-s3e24/leaderboard) top score of `0.87946`.

Models ranked by test ROC AUC (80/20 train-test split):

| Model | ROC AUC | Train time (s) |
|-------|---------|-----------------|
| LightGBM (Tuned) | 0.867 | 621.5 |
| CatBoost | 0.865 | 42.5 |
| XGBoost | 0.863 | 3.9 |
| LightGBM | 0.862 | 3.7 |
| Gradient Boosting | 0.856 | 84.2 |
| Random Forest | 0.853 | 79.1 |
| AdaBoost | 0.838 | 17.4 |
| Logistic Regression | 0.832 | 6.0 |
| Decision Tree | 0.689 | 6.4 |

**Best model configuration (LightGBM):**
- `n_estimators`: 495
- `num_leaves`: 93
- `learning_rate`: 0.04557
- `subsample`: 0.8165
- `colsample_bytree`: 0.5202
- `min_child_samples`: 61
- `reg_alpha`: 1.03e-05
- `reg_lambda`: 1.24e-07

## Key findings

- Tuning actually paid off here: 50 Optuna trials (about 10 minutes) took LightGBM from 0.862 to 0.870 ROC AUC. That's a bigger gain than most episodes see from tuning — worth doing again if this one comes back.
- A single Decision Tree fell off a cliff (0.689), a much wider gap from the ensemble models (0.85+) than usual. This dataset needs the averaging/boosting, not just one tree.
- Clean dataset — no missing values across 159k rows, so the time went into modeling rather than cleanup.
