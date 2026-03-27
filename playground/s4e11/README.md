# Exploring Mental Health Data

**Competition:** [Exploring Mental Health Data](https://www.kaggle.com/competitions/playground-series-s4e11)

A binary classification project predicting whether an individual has depression. Submissions are evaluated using accuracy score between the predicted value and the observed target `Depression`.

## Dataset

```bash
kaggle competitions download -c playground-series-s4e11
```

- Training samples: 140,700
- Target classes: 0/Not Depressed (82%), 1/Depressed (18%)
- Features: 18 total
  - Numerical: 3 (`Age`, `CGPA`, `Work/Study Hours`)
  - Ordinal: 11 (`Academic Pressure`, `Dietary Habits`, `Family History of Mental Illness`, `Financial Stress`, `Gender`, `Have you ever had suicidal thoughts ?`, `Job Satisfaction`, `Sleep Duration`, `Study Satisfaction`, `Work Pressure`, `Working Professional or Student`)
  - High-cardinality: 4 (`City`, `Degree`, `Name`, `Profession`)
- Missing values: present in 9 features (up to 112,803 missing per feature)

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

