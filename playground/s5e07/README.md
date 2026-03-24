# Predict the Introverts from the Extroverts

**Competition:** [Predict the Introverts from the Extroverts](https://www.kaggle.com/competitions/playground-series-s5e7)

A binary classification project predicting personality type (Introvert or Extrovert) from behavioural survey data. Submissions are evaluated on classification accuracy.

## Dataset

```bash
kaggle competitions download -c playground-series-s5e7
```

- Training samples: 18,524
- Features: 7 total (excluding target)
  - Numerical: 5 (`Friends_circle_size`, `Going_outside`, `Post_frequency`, `Social_event_attendance`, `Time_spent_Alone`)
  - Categorical: 2 (`Drained_after_socializing`, `Stage_fear`)
- Missing values present in all feature columns (1,054–1,893 missing per column)
- Target classes: 0/Introvert (26.0%), 1/Extrovert (74.0%)

## Results

The Kaggle submission achieved a private score of `0.968825`, compared to the [leaderboard](https://www.kaggle.com/competitions/playground-series-s5e7/leaderboard) top score of `0.970647`.

Local test results show limited resolution for distinguishing models — many are tied at 3–4 decimal places due to the discrete nature of accuracy on a fixed holdout; the Kaggle private score is the more meaningful measure of model performance.

Models ranked by test accuracy (80/20 stratified split):

| Model | Accuracy | Balanced accuracy | MCC | ROC-AUC | Train time (s) |
|-------|----------|-------------------|-----|---------|----------------|
| SVC | 0.969 | 0.956 | 0.918 | 0.967 | 19.5 |
| AdaBoost | 0.969 | 0.956 | 0.918 | 0.965 | 0.8 |
| Logistic Regression | 0.968 | 0.956 | 0.917 | 0.958 | 0.5 |
| Gradient Boosting | 0.968 | 0.956 | 0.917 | 0.966 | 2.4 |
| Gradient Boosting (Tuned) | 0.968 | 0.956 | 0.917 | 0.966 | 274.5 |
| LightGBM | 0.968 | 0.955 | 0.916 | 0.965 | 0.4 |
| CatBoost | 0.968 | 0.956 | 0.916 | 0.966 | 9.3 |
| XGBoost | 0.966 | 0.952 | 0.911 | 0.965 | 0.4 |
| Random Forest | 0.964 | 0.951 | 0.906 | 0.961 | 2.4 |
| Decision Tree | 0.936 | 0.915 | 0.831 | 0.916 | 0.2 |

**Best model configuration (Gradient Boosting, tuned with Optuna):**
- `n_estimators`: 431
- `learning_rate`: 0.03515
- `max_depth`: 3
- `subsample`: 0.650
- `min_samples_split`: 9
- `min_samples_leaf`: 3

## Key findings

- The problem is nearly linearly separable — Logistic Regression matches the best boosting models at 0.969 accuracy with default parameters
- All models cluster tightly between 0.967–0.969 accuracy, making local evaluation unreliable for ranking; the Kaggle private leaderboard reveals meaningful differences of up to 0.002
- `gb_tuned` and `cat` (untuned) tied for the best Kaggle private score (0.968825), while `ada_tuned` and `svc_tuned` ranked highest locally but dropped to 13th/14th on Kaggle — a sign of overfitting to the holdout split during tuning
- Train and test distributions are indistinguishable (validation score 0.50), confirming the synthetic data generation is well-calibrated
