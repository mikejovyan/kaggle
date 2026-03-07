# Heart disease classification

**Competition:** [Binary classification with a tabular heart disease dataset](https://www.kaggle.com/competitions/playground-series-s6e2)

A binary classification project predicting whether a patient has heart disease. Submissions are evaluated on area under the ROC curve between the predicted probability and the observed target `Heart Disease`.

## Dataset

```bash
kaggle competitions download -c playground-series-s6e2
```

- Training samples: 630,000
- Features: 13 total (excluding target)
  - Numerical: 5 features
  - Nominal: 3 features
  - Ordinal: 5 features
- Target classes: Absence (55.2%), Presence (44.8%)

## Results

The Kaggle submission achieved a private score of `0.95515`, compared to the [leaderboard](https://www.kaggle.com/competitions/playground-series-s6e2/leaderboard) top score of `0.95535`.

Models ranked by test ROC-AUC (80/20 train-test split):

| Model | ROC-AUC | Train time (s) |
|-------|---------|----------------|
| CatBoost (Tuned) | 0.955 | 1969.6 |
| CatBoost | 0.955 | 162.7 |
| XGBoost | 0.955 | 7.0 |
| LightGBM | 0.955 | 8.4 |
| Gradient Boosting | 0.954 | 193.1 |
| AdaBoost | 0.953 | 44.5 |
| Logistic Regression | 0.952 | 6.7 |
| Random Forest | 0.946 | 174.9 |
| Decision Tree | 0.823 | 9.2 |
| SVC | Too slow | - |

**Best model configuration (CatBoost):**
- `iterations`: 877
- `depth`: 4
- `learning_rate`: 0.08962
- `l2_leaf_reg`: 0.5613
- `subsample`: 0.8554

## Key findings

- Gradient boosting algorithms (CatBoost, XGBoost, LightGBM, GB) dominated, all achieving ROC-AUC above 0.954
- Default CatBoost, XGBoost, and LightGBM all tied at 0.955 ROC-AUC with tuning providing only marginal improvement
- Random Forest underperformed relative to other ensemble methods (0.946 vs 0.954+)
- Dataset is relatively balanced (55.2% Absence vs 44.8% Presence), requiring no resampling
- SVC was excluded from full evaluation due to excessive training time on this dataset size
