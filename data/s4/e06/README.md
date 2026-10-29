# Classification with an Academic Success Dataset

**Competition:** [Classification with an Academic Success Dataset](https://www.kaggle.com/competitions/playground-series-s4e6)

A multi-class classification project predicting student academic outcomes (Dropout, Enrolled, or Graduate). Submissions are evaluated using accuracy score between the predicted value and the observed target `Target`.

## Dataset

```bash
kaggle competitions download -c playground-series-s4e6
```

- Training samples: 76,518
- Target classes: Dropout (33%), Enrolled (20%), Graduate (47%)
- Features: 36 total
  - Numerical: 26 (`Admission grade`, `Age at enrollment`, `Curricular units 1st sem (approved)`, `Curricular units 1st sem (credited)`, `Curricular units 1st sem (enrolled)`, `Curricular units 1st sem (evaluations)`, `Curricular units 1st sem (grade)`, `Curricular units 1st sem (without evaluations)`, `Curricular units 2nd sem (approved)`, `Curricular units 2nd sem (credited)`, `Curricular units 2nd sem (enrolled)`, `Curricular units 2nd sem (evaluations)`, `Curricular units 2nd sem (grade)`, `Curricular units 2nd sem (without evaluations)`, `Daytime/evening attendance`, `Debtor`, `Displaced`, `Educational special needs`, `GDP`, `Gender`, `Inflation rate`, `International`, `Previous qualification (grade)`, `Scholarship holder`, `Tuition fees up to date`, `Unemployment rate`)
  - Nominal: 2 (`Application order`, `Marital status`)
  - High-cardinality: 8 (`Application mode`, `Course`, `Father's occupation`, `Father's qualification`, `Mother's occupation`, `Mother's qualification`, `Nacionality`, `Previous qualification`)
- Missing values: none

## Results

The Kaggle submission achieved a private score of `0.83766`, compared to the [leaderboard](https://www.kaggle.com/competitions/playground-series-s4e6/leaderboard) top score of `0.84035`.

Models ranked by test accuracy (80/20 train-test split):

| Model | Accuracy | Train time (s) |
|-------|----------|-----------------|
| CatBoost | 0.835 | 58.7 |
| XGBoost (Tuned) | 0.835 | 1329.8 |
| LightGBM | 0.833 | 5.7 |
| XGBoost | 0.833 | 6.1 |
| Gradient Boosting | 0.831 | 208.2 |
| Random Forest | 0.828 | 43.1 |
| Logistic Regression | 0.822 | 53.7 |
| AdaBoost | 0.813 | 15.6 |
| Decision Tree | 0.754 | 5.3 |

**Best model configuration (XGBoost):**
- `n_estimators`: 450
- `max_depth`: 7
- `learning_rate`: 0.05139
- `subsample`: 0.9412
- `colsample_bytree`: 0.6572
- `min_child_weight`: 1
- `gamma`: 0.00035

## Key findings

- Classes here are fairly balanced (33/20/47%), with no class rare, allowing Logistic Regression (0.822) and AdaBoost (0.813) to keep up with the ensembles (0.835) instead of falling far behind.
