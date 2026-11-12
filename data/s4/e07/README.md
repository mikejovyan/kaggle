# Binary Classification of Insurance Cross Selling

**Competition:** [Binary Classification of Insurance Cross Selling](https://www.kaggle.com/competitions/playground-series-s4e7)

A binary classification project predicting whether a health insurance customer will respond positively to a vehicle insurance cross-sell offer. Submissions are evaluated on area under the ROC curve between the predicted probability and the observed target `Response`.

## Dataset

```bash
kaggle competitions download -c playground-series-s4e7
```

- Training samples: 11,504,798
- Target classes: 0/Not interested (88%), 1/Interested (12%)
- Features: 10 total
  - Numerical: 3 (`Age`, `Annual_Premium`, `Vintage`)
  - Ordinal: 5 (`Driving_License`, `Gender`, `Previously_Insured`, `Vehicle_Age`, `Vehicle_Damage`)
  - High-cardinality: 2 (`Policy_Sales_Channel`, `Region_Code`)
- Missing values: none, but `Region_Code` arrives as a float with corrupted non-integer values and is rounded before use

## Results

The Kaggle submission achieved a private score of `0.88075`, compared to the [leaderboard](https://www.kaggle.com/competitions/playground-series-s4e7/leaderboard) top score of `0.89754`.

Models ranked by test ROC AUC (80/20 train-test split):

| Model | ROC AUC | Train time (s) |
|-------|---------|-----------------|
| XGBoost (Tuned) | 0.881 | 2485.8 |
| CatBoost | 0.880 | 2200.1 |
| XGBoost | 0.879 | 139.7 |
| LightGBM | 0.876 | 175.8 |
| Logistic Regression | 0.849 | 98.2 |
| Decision Tree | 0.622 | 260.6 |

**Best model configuration (XGBoost):**
- `n_estimators`: 437
- `max_depth`: 10
- `learning_rate`: 0.06505
- `subsample`: 0.7993
- `colsample_bytree`: 0.5780
- `min_child_weight`: 2
- `gamma`: 2.92e-08

## Key findings

- At 11.5M rows, Random Forest, SVC, Gradient Boosting, and AdaBoost were all dropped from the candidate pool as "too slow" — only Logistic Regression, Decision Tree, XGBoost, LightGBM, and CatBoost were trained at all.
- Despite a strong ROC AUC (0.881), balanced accuracy for the tuned model is only 0.554 — barely above random (0.5) once forced into a single 0/1 decision. With the target at 88/12, ROC AUC is the metric that actually reflects this model's ranking ability; accuracy and a default threshold both favor the majority class too heavily to be useful here.
