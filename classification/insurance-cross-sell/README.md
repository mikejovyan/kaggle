# Insurance cross-sell classification

**Competition:** [Binary classification of insurance cross-sell](https://www.kaggle.com/competitions/playground-series-s4e7/)

A binary classification project predicting whether a health insurance customer would respond negatively or positively to  in vehicle insurance, based on demographic and vehicle data. Submissions are evaluated on area under the ROC curve between the predicted probability and the observed target.

## Dataset

```bash
kaggle competitions download -c playground-series-s4e7
```

- Training samples: 11,504,798
- Features: 10 total (excluding target)
  - Numerical: 3 features (Age, Annual_Premium, Vintage)
  - Ordinal: 5 features (Driving_License, Gender, Previously_Insured, Vehicle_Age, Vehicle_Damage)
  - High-cardinality categorical: 2 features (Region_Code, Policy_Sales_Channel)
- Target classes: 0/Negative response (87.7%), 1/Positive response (12.3%)

## Results

The Kaggle submission achieved a private score of `0.88036`, compared to the [leaderboard](https://www.kaggle.com/competitions/playground-series-s4e7/leaderboard) top score of `0.89754`.

Models ranked by test ROC-AUC (80/20 train-test split):

| Model | ROC-AUC | Train time (s) |
|-------|---------|----------------|
| LightGBM (Tuned) | 0.880 | 2859.2 |
| XGBoost | 0.878 | 107.0 |
| LightGBM | 0.876 | 137.2 |

**Best model configuration (LightGBM tuned):**
- `n_estimators`: 437
- `num_leaves`: 124
- `learning_rate`: 0.06505
- `subsample`: 0.79933
- `colsample_bytree`: 0.57801
- `min_child_samples`: 48
- `reg_alpha`: 3.332e-08
- `reg_lambda`: 0.62458

## Key findings

- At over 11.5 million training rows, this is by far the largest dataset in the series. The sheer size means most models (Logistic Regression, SVC, Random Forest, Decision Tree, Gradient Boosting, AdaBoost, CatBoost) were too slow to run and were excluded. Only XGBoost and LightGBM completed in a reasonable time, with even the Optuna tuning step taking nearly 48 minutes per trial. The submission file is not tracked in this repo due to its size (218 MB) — see [.gitignore](../../.gitignore#L43).
- The dataset is heavily imbalanced (87.7% class 0), which is reflected in the low balanced accuracy scores (0.53–0.55) despite high overall accuracy (~0.880).
- Tuning LightGBM with just 5 Optuna trials improved ROC-AUC from 0.876 to 0.880 at the cost of 47 minutes of training time (vs ~2 minutes untuned).
- XGBoost matched LightGBM untuned performance (0.878) and was faster to train (107s vs 137s).
- The gap between local test score (0.880) and Kaggle private score (0.880) is negligible, likely because the massive dataset size makes the 80/20 split a reliable proxy for generalisation.
