# Regression with an Insurance Dataset

**Competition:** [Regression with an Insurance Dataset](https://www.kaggle.com/competitions/playground-series-s4e12)

A regression project predicting the insurance premium amount for policyholders based on demographic and policy attributes. Submissions are evaluated using root mean squared logarithmic error between the predicted value and the observed target `Premium Amount`.

## Dataset

```bash
kaggle competitions download -c playground-series-s4e12
```

- Training samples: 1,200,000
- Target: mean ± std = 1,102.54 ± 865.00, range [20, 4,999]
- Features: 19 total
  - Numerical: 8 (`Age`, `Annual Income`, `Credit Score`, `Health Score`, `Insurance Duration`, `Number of Dependents`, `Previous Claims`, `Vehicle Age`)
  - Nominal: 4 (`Gender`, `Marital Status`, `Occupation`, `Property Type`)
  - Ordinal: 6 (`Customer Feedback`, `Education Level`, `Exercise Frequency`, `Location`, `Policy Type`, `Smoking Status`)
  - High-cardinality: 1 (`Policy Start Date`)
- Missing values: present in 11 features (up to 364,029 missing per feature)

## Results

The Kaggle submission achieved a private score of `1.13953`, compared to the [leaderboard](https://www.kaggle.com/competitions/playground-series-s4e12/leaderboard) top score of `1.01706`.

Models ranked by test RMSLE (80/20 train-test split):

| Model | RMSLE | Train time (s) |
|-------|-------|----------------|
| LightGBM (Tuned) | 1.140 | 1413.9 |
| LightGBM | 1.140 | 28.0 |
| CatBoost | 1.141 | 168.5 |
| XGBoost | 1.143 | 35.5 |
| Ridge | 1.169 | 17.9 |
| AdaBoost | 1.275 | 174.3 |
| Decision Tree | 1.513 | 114.1 |
| SVR | Too slow | - |
| Random Forest | Too slow | - |
| Gradient Boosting | Too slow | - |

**Best model configuration (LightGBM):**
- `n_estimators`: 892
- `num_leaves`: 64
- `learning_rate`: 0.008604
- `subsample`: 0.5515
- `colsample_bytree`: 0.9995
- `min_child_samples`: 117
- `reg_alpha`: 9.864
- `reg_lambda`: 2.39e-06

## Key findings

- LightGBM led all models; tuning with Optuna improved CV RMSLE from 1.139 to 1.138 but made no measurable difference on the hold-out test set
- CatBoost and XGBoost scored within 0.003 RMSLE of LightGBM, while Ridge lagged by 0.029, reflecting strong non-linear relationships in the data
- SVR, Random Forest, and Gradient Boosting were skipped due to excessive runtime on the 1.2M-sample training set
- Missing values were pervasive — Previous Claims and Occupation each had over 30% missing, likely structural absences rather than random gaps
- `Policy Start Date` was decomposed into seven features (year, day-of-week, Unix timestamp, and cyclical month/day encodings) to capture temporal patterns
