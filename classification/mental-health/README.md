# Mental health classification

**Competition:** [Binary classification with a mental health dataset](https://www.kaggle.com/competitions/playground-series-s4e11)

A binary classification project predicting whether an individual has depression. Submissions are evaluated using the accuracy score against the observed target `Depression`.

## Dataset

```bash
kaggle competitions download -c playground-series-s4e11
```

- Training samples: 140,700
- Features: 18 total (excluding target)
  - Numerical: 3 features
  - Ordinal: 11 features
  - High cardinality: 4 features
- Target classes: 0/Not Depressed (81.8%), 1/Depressed (18.2%)

## Results

The Kaggle submission achieved a private score of `0.94023`, compared to the [leaderboard](https://www.kaggle.com/competitions/playground-series-s4e11/leaderboard) top score of `0.94184`.

Models ranked by test accuracy (80/20 train-test split):

| Model | Accuracy | Train time (s) |
|-------|----------|----------------|
| CatBoost | 0.940 | 39.6 |
| CatBoost (Tuned) | 0.940 | 1214.8 |
| XGBoost | 0.938 | 5.2 |
| LightGBM | 0.938 | 4.8 |
| Gradient Boosting | 0.938 | 55.0 |
| Random Forest | 0.937 | 43.3 |
| Logistic Regression | 0.936 | 6.4 |
| AdaBoost | 0.935 | 14.5 |
| Decision Tree | 0.903 | 4.9 |
| SVC | Too slow | - |

**Best model configuration (CatBoost):**
- `iterations`: 716
- `learning_rate`: 0.07747
- `depth`: 5
- `l2_leaf_reg`: 0.0710
- `subsample`: 0.928

## Key findings

- CatBoost dominated, with both default and tuned versions achieving the highest accuracy (0.940)
- Tuning provided no accuracy gain over default CatBoost at a significant time cost (39.6s vs 1214.8s)
- XGBoost, LightGBM, and Gradient Boosting all tied at 0.938 accuracy
- The dataset has substantial missing values across several features, particularly for student/professional-specific columns
- SVC was excluded from full evaluation due to excessive training time on this dataset size
