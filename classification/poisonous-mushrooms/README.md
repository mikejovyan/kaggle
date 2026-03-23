# Binary prediction of poisonous mushrooms

**Competition:** [Binary prediction of poisonous mushrooms](https://www.kaggle.com/competitions/playground-series-s4e8)

A binary classification project predicting whether a mushroom is edible or poisonous based on physical characteristics. Submissions are evaluated on area under the ROC curve between the predicted probability and the observed target.

## Dataset

```bash
kaggle competitions download -c playground-series-s4e8
```

- Training samples: 3,116,945
- Features: 20 total (excluding target)
  - Numerical: 3 features (cap-diameter, stem-height, stem-width)
  - Nominal: 1 feature (season)
  - Ordinal: 16 features (cap-color, cap-shape, cap-surface, does-bruise-or-bleed, gill-attachment, gill-color, gill-spacing, habitat, has-ring, ring-type, spore-print-color, stem-color, stem-root, stem-surface, veil-color, veil-type)
- Target classes: e/Edible (45.3%), p/Poisonous (54.7%)

## Results

The Kaggle submission achieved a private score of `0.98454`, compared to the [leaderboard](https://www.kaggle.com/competitions/playground-series-s4e8/leaderboard) top score of `0.98514`.

Models ranked by test ROC-AUC (80/20 train-test split):

| Model | ROC-AUC | Train time (s) |
|-------|---------|----------------|
| XGBoost (Tuned) | 0.997 | 1564.9 |
| CatBoost | 0.997 | 681.2 |
| XGBoost | 0.997 | 48.7 |
| Random Forest | 0.996 | 1469.8 |
| LightGBM | 0.996 | 54.9 |
| Decision Tree | 0.983 | 100.5 |
| AdaBoost | 0.778 | 300.3 |
| Logistic Regression | 0.707 | 98.2 |

**Best model configuration (XGBoost tuned):**
- `n_estimators`: 437
- `max_depth`: 10
- `learning_rate`: 0.06505
- `subsample`: 0.79933
- `colsample_bytree`: 0.57801
- `min_child_weight`: 2
- `gamma`: 2.915e-08

## Key findings

- Tree-based ensemble methods dominated, with XGBoost, CatBoost, Random Forest, and LightGBM all achieving ROC-AUC above 0.996, indicating the mushroom features are highly discriminative.
- Default XGBoost matched the tuned model (0.997 ROC-AUC) at a fraction of the training time (49s vs 1565s), suggesting the default hyperparameters were already near-optimal for this dataset.
- Tuning with Optuna used the same hyperparameters as the insurance cross-sell competition (n_estimators, max_depth, learning_rate, subsample, colsample_bytree, min_child_weight, gamma), which happened to converge on nearly identical values.
- AdaBoost and Logistic Regression performed poorly (0.778 and 0.707 ROC-AUC respectively), likely because they struggle with the high-cardinality ordinal features and non-linear decision boundaries.
- The dataset is large (3.1M rows), but XGBoost and LightGBM remained fast enough for full evaluation, unlike the insurance cross-sell competition where most models had to be excluded.
