# Binary Prediction of Poisonous Mushrooms

**Competition:** [Binary Prediction of Poisonous Mushrooms](https://www.kaggle.com/competitions/playground-series-s4e8)

A binary classification project predicting whether a mushroom is edible or poisonous. Submissions are evaluated using the Matthews correlation coefficient between the predicted value and the observed target `class`.

## Dataset

```bash
kaggle competitions download -c playground-series-s4e8
```

- Training samples: 3,116,945
- Target classes: p/Poisonous (55%), e/Edible (45%)
- Features: 20 total
  - Numerical: 3 (`cap-diameter`, `stem-height`, `stem-width`)
  - Nominal: 15 (`cap-color`, `cap-shape`, `cap-surface`, `gill-attachment`, `gill-color`, `gill-spacing`, `habitat`, `ring-type`, `season`, `spore-print-color`, `stem-color`, `stem-root`, `stem-surface`, `veil-color`, `veil-type`)
  - Ordinal: 2 (`does-bruise-or-bleed`, `has-ring`)
- Missing values: present in 17 features, from 4 up to 2,957,493 missing (`veil-type` is missing in 95% of rows)

## Results

The Kaggle submission achieved a private score of `0.98466`, compared to the [leaderboard](https://www.kaggle.com/competitions/playground-series-s4e8/leaderboard) top score of `0.98514`.

Models ranked by test Matthews correlation coefficient (80/20 train-test split):

| Model | MCC | Train time (s) |
|-------|-----|-----------------|
| XGBoost (Tuned) | 0.984 | 3742.7 |
| Random Forest | 0.984 | 2159.4 |
| CatBoost | 0.983 | 716.8 |
| XGBoost | 0.982 | 139.4 |
| LightGBM | 0.978 | 105.5 |
| Decision Tree | 0.967 | 404.3 |
| Gradient Boosting | 0.860 | 3475.3 |
| Logistic Regression | 0.698 | 345.7 |
| AdaBoost | 0.501 | 803.7 |

**Best model configuration (XGBoost):**
- `n_estimators`: 437
- `max_depth`: 10
- `learning_rate`: 0.06505
- `subsample`: 0.7993
- `colsample_bytree`: 0.5780
- `min_child_weight`: 2
- `gamma`: 2.92e-08

## Key findings

- `veil-type` is missing in 95% of rows, and several other features (`spore-print-color`, `stem-root`, `veil-color`, `stem-surface`) are missing in 60–95%. Rather than imputing, these are kept as NaN through a dedicated "missing" category, letting tree models split on absence itself.
- Tuning XGBoost took over an hour for just 10 Optuna trials (~6 minutes each) on this 3.1M-row dataset, and only nudged MCC from 0.982 to 0.984 — barely worth the wait.
