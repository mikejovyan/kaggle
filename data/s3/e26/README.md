# Multi-Class Prediction of Cirrhosis Outcomes

**Competition:** [Multi-Class Prediction of Cirrhosis Outcomes](https://www.kaggle.com/competitions/playground-series-s3e26)

A multi-class classification project predicting a cirrhosis patient's outcome (censored, censored due to liver transplant, or deceased). Submissions are evaluated using multi-class logarithmic loss between the predicted probabilities and the observed target `Status`.

## Dataset

```bash
kaggle competitions download -c playground-series-s3e26
```

- Training samples: 7,905
- Target classes: C/Censored (63%), D/Deceased (34%), CL/Censored due to liver transplant (3%)
- Features: 18 total
  - Numerical: 11 (`Age`, `Albumin`, `Alk_Phos`, `Bilirubin`, `Cholesterol`, `Copper`, `N_Days`, `Platelets`, `Prothrombin`, `SGOT`, `Tryglicerides`)
  - Ordinal: 7 (`Ascites`, `Drug`, `Edema`, `Hepatomegaly`, `Sex`, `Spiders`, `Stage`)
- Missing values: none

## Results

The Kaggle submission achieved a private score of `0.41623`, compared to the [leaderboard](https://www.kaggle.com/competitions/playground-series-s3e26/leaderboard) top score of `0.39104`.

Models ranked by test log loss (80/20 train-test split):

| Model | Log loss ↓ | Train time (s) |
|-------|------------|-----------------|
| XGBoost (Tuned) | 0.434 | 150.4 |
| Gradient Boosting | 0.442 | 11.5 |
| LightGBM | 0.458 | 1.2 |
| CatBoost | 0.465 | 12.6 |
| Random Forest | 0.503 | 3.3 |
| XGBoost | 0.514 | 1.4 |
| Logistic Regression | 0.529 | 1.0 |
| SVC | 0.549 | 12.0 |
| AdaBoost | 1.003 | 1.1 |
| Decision Tree | 9.689 | 0.3 |

**Best model configuration (XGBoost):**
- `n_estimators`: 484
- `max_depth`: 8
- `learning_rate`: 0.01908
- `subsample`: 0.5169
- `colsample_bytree`: 0.5047
- `min_child_weight`: 10
- `gamma`: 0.41509

## Key findings

- Tuning XGBoost cut the Kaggle private score from 0.46964 to 0.41623 — a real, substantial gain, unlike most other episodes where tuning barely moved the number.
- Balanced accuracy (0.604) is far below plain accuracy (0.840) for the tuned model. The CL class is only 3% of the data, and the model is clearly struggling on it much more than the usual accuracy/balanced-accuracy gap would suggest.
