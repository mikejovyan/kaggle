# Predicting Heart Disease

**Competition:** [Predicting Heart Disease](https://www.kaggle.com/competitions/playground-series-s6e2)

A binary classification project predicting whether a patient has heart disease. Submissions are evaluated on area under the ROC curve between the predicted probability and the observed target `Heart Disease`.

## Dataset

```bash
kaggle competitions download -c playground-series-s6e2
```

- Training samples: 630,000
- Target classes: Absence (55%), Presence (45%)
- Features: 13 total
  - Numerical: 5 (`Age`, `BP`, `Cholesterol`, `Max HR`, `ST depression`)
  - Nominal: 3 (`Chest pain type`, `EKG results`, `Thallium`)
  - Ordinal: 5 (`Exercise angina`, `FBS over 120`, `Number of vessels fluro`, `Sex`, `Slope of ST`)
- Missing values: none

## Results

The Kaggle submission achieved a private score of `0.95514`, compared to the [leaderboard](https://www.kaggle.com/competitions/playground-series-s6e2/leaderboard) top score of `0.95535`.

Models ranked by test ROC AUC (80/20 train-test split):

| Model | ROC AUC | Train time (s) |
|-------|---------|-----------------|
| CatBoost (Tuned) | 0.955 | 1728.0 |
| CatBoost | 0.955 | 167.8 |
| XGBoost | 0.955 | 12.8 |
| LightGBM | 0.955 | 14.2 |
| Gradient Boosting | 0.954 | 217.2 |
| AdaBoost | 0.953 | 46.4 |
| Logistic Regression | 0.952 | 11.7 |
| Random Forest | 0.946 | 200.3 |
| Decision Tree | 0.823 | 12.1 |

**Best model configuration (CatBoost):**
- `iterations`: 539
- `depth`: 3
- `learning_rate`: 0.26140
- `l2_leaf_reg`: 6.1993
- `subsample`: 0.6205

## Key findings

- These are the same clinical fields as the classic UCI/Statlog Heart Disease dataset (`Chest pain type`, `ST depression`, `Thallium`, `Number of vessels fluro`, etc.) — a well-studied medical feature set rather than a freshly engineered one.
- 30 trials tuning CatBoost took 29 minutes and didn't move ROC AUC at all (0.955 before and after, to 3 decimal places) — not worth repeating.
