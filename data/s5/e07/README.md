# Predict the Introverts from the Extroverts

**Competition:** [Predict the Introverts from the Extroverts](https://www.kaggle.com/competitions/playground-series-s5e7)

A binary classification project predicting personality type (Introvert or Extrovert) from behavioural survey data. Submissions are evaluated using accuracy score between the predicted value and the observed target `Personality`.

## Dataset

```bash
kaggle competitions download -c playground-series-s5e7
```

- Training samples: 18,524
- Target classes: 0/Extrovert (74%), 1/Introvert (26%)
- Features: 7 total
  - Numerical: 5 (`Friends_circle_size`, `Going_outside`, `Post_frequency`, `Social_event_attendance`, `Time_spent_Alone`)
  - Ordinal: 2 (`Drained_after_socializing`, `Stage_fear`)
- Missing values: present in all 7 features (1,054–1,893 missing per feature)

## Results

The Kaggle submission achieved a private score of `0.968825`, compared to the [leaderboard](https://www.kaggle.com/competitions/playground-series-s5e7/leaderboard) top score of `0.970647`.

Models ranked by test accuracy (80/20 train-test split):

| Model | Accuracy | Train time (s) |
|-------|----------|-----------------|
| SVC | 0.969 | 19.5 |
| Random Forest (Tuned) | 0.969 | 91.4 |
| AdaBoost | 0.969 | 1.0 |
| Logistic Regression | 0.968 | 0.4 |
| Gradient Boosting | 0.968 | 2.6 |
| LightGBM | 0.968 | 0.5 |
| CatBoost | 0.968 | 10.4 |
| XGBoost | 0.966 | 2.3 |
| Random Forest | 0.964 | 3.0 |
| Decision Tree | 0.936 | 0.2 |

**Best model configuration (Random Forest):**
- `n_estimators`: 184
- `max_depth`: 3
- `min_samples_split`: 17
- `min_samples_leaf`: 3
- `max_features`: log2

## Key findings

- 8 of 10 models land within 0.005 accuracy of each other (0.964–0.969), and plain Logistic Regression with zero tuning (0.968) keeps up with all of them. This problem is close to linearly separable.
- On the private leaderboard, Random Forest (tuned), Gradient Boosting, and CatBoost — three very different models — all landed on the exact same score (0.968825). At this precision, local differences between models are basically noise.
- Every one of the 7 features has missing values (6–10% per column), but it barely matters — every model still clears 0.96 accuracy.
