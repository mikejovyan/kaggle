# Exploring Mental Health Data

**Competition:** [Exploring Mental Health Data](https://www.kaggle.com/competitions/playground-series-s4e11)

A binary classification project predicting whether a person is experiencing depression. Submissions are evaluated using accuracy score between the predicted value and the observed target `Depression`.

## Dataset

```bash
kaggle competitions download -c playground-series-s4e11
```

- Training samples: 140,700
- Target classes: 0/Not depressed (82%), 1/Depressed (18%)
- Features: 18 total
  - Numerical: 8 (`Academic Pressure`, `Age`, `CGPA`, `Financial Stress`, `Job Satisfaction`, `Study Satisfaction`, `Work Pressure`, `Work/Study Hours`)
  - Ordinal: 6 (`Dietary Habits`, `Family History of Mental Illness`, `Gender`, `Have you ever had suicidal thoughts ?`, `Sleep Duration`, `Working Professional or Student`)
  - High-cardinality: 4 (`City`, `Degree`, `Name`, `Profession`)
- Missing values: present in 8 features, dominated by `Academic Pressure`/`CGPA`/`Study Satisfaction` (~112,800 missing each — these only apply to students)

## Results

The Kaggle submission achieved a private score of `0.94052`, compared to the [leaderboard](https://www.kaggle.com/competitions/playground-series-s4e11/leaderboard) top score of `0.94184`.

Models ranked by test accuracy (80/20 train-test split):

| Model | Accuracy | Train time (s) |
|-------|----------|-----------------|
| CatBoost (Tuned) | 0.939 | 528.9 |
| CatBoost | 0.939 | 36.0 |
| LightGBM | 0.939 | 3.1 |
| XGBoost | 0.938 | 3.2 |
| Gradient Boosting | 0.938 | 51.5 |
| AdaBoost | 0.937 | 12.7 |
| Logistic Regression | 0.937 | 12.4 |
| Random Forest | 0.937 | 40.4 |
| Decision Tree | 0.904 | 3.3 |

**Best model configuration (CatBoost):**
- `iterations`: 333
- `depth`: 3
- `learning_rate`: 0.14751
- `l2_leaf_reg`: 2.0834
- `subsample`: 0.8466

## Key findings

- `Academic Pressure`, `CGPA`, and `Study Satisfaction` are missing in ~80% of rows because they only apply to students, while `Work Pressure`/`Job Satisfaction` are the professional equivalents — these are two disjoint feature sets stacked into one table, filled with 0 rather than imputed, rather than ordinary missingness.
- 9 hours of CatBoost tuning moved accuracy from 0.939 to 0.939 (barely visible at 3 decimal places) — not worth repeating; the untuned model would have scored the same on Kaggle.
- All 8 non-tree-based-tuned models land within 0.937–0.939 accuracy, the tightest spread seen across these episodes — even the Decision Tree only falls to 0.904, suggesting a few features carry most of the signal and the rest is marginal.
