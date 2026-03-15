# Smoker status classification

**Competition:** [Binary prediction of smoker status using bio-signals](https://www.kaggle.com/competitions/playground-series-s3e24/)

A binary classification project predicting whether a patient is a smoker based on bio-signal data. Submissions are evaluated on area under the ROC curve between the predicted probability and the observed target.

## Dataset

```bash
kaggle competitions download -c playground-series-s3e24
```

- Training samples: 159,256
- Features: 22 total (excluding target)
  - Numerical: 18 features
  - Ordinal: 4 features
- Target classes: 0/Not smoker (56.3%), 1/Smoker (43.7%)

## Results

The Kaggle submission achieved a private score of `0.87222`, compared to the [leaderboard](https://www.kaggle.com/competitions/playground-series-s3e24/leaderboard) top score of `0.87946`.

Models ranked by test ROC-AUC (80/20 train-test split):

| Model | ROC-AUC | Train time (s) |
|-------|---------|----------------|
| LightGBM (Tuned) | 0.868 | 721.5 |
| CatBoost | 0.865 | 47.6 |
| XGBoost | 0.863 | 2.4 |
| LightGBM | 0.862 | 2.8 |
| Gradient Boosting | 0.856 | 81.8 |
| Random Forest | 0.853 | 81.7 |
| AdaBoost | 0.838 | 17.7 |
| Logistic Regression | 0.832 | 3.7 |
| Decision Tree | 0.689 | 6.2 |

**Best model configuration (LightGBM tuned):**
- `n_estimators`: 783
- `num_leaves`: 54
- `learning_rate`: 0.06318
- `subsample`: 0.5435
- `colsample_bytree`: 0.5079
- `min_child_samples`: 153
- `reg_alpha`: 1.146e-07
- `reg_lambda`: 6.539

## Key findings

- Gradient boosting algorithms (LightGBM, CatBoost, XGBoost, GB) dominated, all achieving ROC-AUC above 0.856
- Default XGBoost and LightGBM achieved near-equivalent performance (0.863 and 0.862) with tuning adding only marginal improvement (0.868) at a large time cost (2.4s vs 721.5s)
- CatBoost performed strongly without tuning (0.865), close to the tuned LightGBM result
- The dataset is moderately imbalanced (56.3% non-smoker vs 43.7% smoker), but this did not require resampling
- A gap remains between the local test score (0.868) and the Kaggle private score (0.872), suggesting the full training data helps generalisation
