# Predicting Airline Satisfaction

**Competition:** [Predicting Airline Satisfaction](https://www.kaggle.com/competitions/playground-series-s6e10)

A binary classification project predicting whether an airline passenger is satisfied with their flight experience. Submissions are evaluated on area under the ROC curve between the predicted probability and the observed target `satisfaction`.

## Dataset

```bash
kaggle competitions download -c playground-series-s6e10
```

- Training samples: 699,635
- Target classes: 0/Not satisfied (56%), 1/Satisfied (44%)
- Features: 21 total
  - Numerical: 17 (`Age`, `Arrival Delay in Minutes`, `Baggage handling`, `Checkin service`, `Cleanliness`, `Departure Delay in Minutes`, `Departure/Arrival time convenient`, `Ease of Online booking`, `Flight Distance`, `Food and drink`, `Gate location`, `Inflight entertainment`, `Inflight wifi service`, `Leg room service`, `On-board service`, `Online boarding`, `Seat comfort`)
  - Ordinal: 4 (`Class`, `Customer Type`, `Gender`, `Type of Travel`)
- Missing values: present in 1 feature (292 missing in `Arrival Delay in Minutes`)

## Results

The Kaggle submission achieved a private score of `TBD`, compared to the [leaderboard](https://www.kaggle.com/competitions/playground-series-s6e10/leaderboard) top score of `TBD`.

Models ranked by test ROC AUC (80/20 train-test split):

| Model | ROC AUC | Train time (s) |
|-------|---------|-----------------|
| XGBoost (Tuned) | 0.959 | 1807.1 |
| CatBoost | 0.958 | 169.0 |
| LightGBM | 0.958 | 11.0 |
| XGBoost | 0.958 | 9.4 |
| Random Forest | 0.955 | 291.1 |
| Gradient Boosting | 0.953 | 288.2 |
| AdaBoost | 0.941 | 61.6 |
| Logistic Regression | 0.921 | 20.9 |
| Decision Tree | 0.864 | 20.1 |

**Best model configuration (XGBoost):**
- `n_estimators`: 925
- `max_depth`: 8
- `learning_rate`: 0.02101
- `subsample`: 0.8303
- `colsample_bytree`: 0.8791
- `min_child_weight`: 9
- `gamma`: 2.27e-08

## Key findings

- Gradient boosting algorithms (XGBoost, CatBoost, LightGBM) dominated, all clustering tightly at 0.958–0.959 ROC AUC
- Tuning XGBoost with Optuna (30 trials) improved ROC AUC only marginally (0.958 → 0.959) at a very large time cost (9.4s vs 1807.1s)
- Random Forest was competitive with the boosting models (0.955) despite no tuning
- Train and test distributions are indistinguishable (adversarial balanced accuracy = 0.50), indicating no distribution shift
- Missing `Arrival Delay in Minutes` values were imputed with the column median before training
