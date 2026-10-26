# Binary Prediction with a Rainfall Dataset

**Competition:** [Binary Prediction with a Rainfall Dataset](https://www.kaggle.com/competitions/playground-series-s5e3)

A binary classification project predicting whether rainfall will occur on a given day. Submissions are evaluated on area under the ROC curve between the predicted probability and the observed target `rainfall`.

## Dataset

```bash
kaggle competitions download -c playground-series-s5e3
```

- Training samples: 2,190
- Target classes: 0/No rainfall (25%), 1/Rainfall (75%)
- Features: 11 total
  - Numerical: 11 (`cloud`, `day`, `dewpoint`, `humidity`, `maxtemp`, `mintemp`, `pressure`, `sunshine`, `temparature`, `winddirection`, `windspeed`)
- Missing values: none

## Results

The Kaggle submission achieved a private score of `0.90235`, compared to the [leaderboard](https://www.kaggle.com/competitions/playground-series-s5e3/leaderboard) top score of `0.90654`.

Models ranked by test ROC AUC (80/20 train-test split):

| Model | ROC AUC | Train time (s) |
|-------|---------|-----------------|
| Logistic Regression | 0.872 | 0.3 |
| AdaBoost (Tuned) | 0.871 | 61.2 |
| Gradient Boosting | 0.863 | 1.3 |
| CatBoost | 0.863 | 5.0 |
| Random Forest | 0.858 | 1.1 |
| AdaBoost | 0.857 | 0.5 |
| LightGBM | 0.853 | 0.3 |
| XGBoost | 0.853 | 0.5 |
| SVC | 0.832 | 1.1 |
| Decision Tree | 0.717 | 0.1 |

**Best model configuration (AdaBoost):**
- `n_estimators`: 307
- `learning_rate`: 0.3934

## Key findings

- Same model, three different scores: 0.900 CV, 0.871 holdout, 0.90235 private. Only ~438 test rows — don't trust the local holdout here.
- Logistic Regression beat every boosted tree by default. Worth trying first if this one comes back.
