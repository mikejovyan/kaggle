# Horse health classification

**Competition:** [Multi-class prediction of horse health outcomes](https://www.kaggle.com/competitions/playground-series-s3e22)

A multi-class classification project predicting horse health outcomes (lived, died, or euthanized). Submissions are evaluated using the micro-averaged F1 score against the observed target `outcome`.

## Dataset

```bash
kaggle competitions download -c playground-series-s3e22
```

- Training samples: 1,235
- Features: 26 total (excluding target)
  - Numerical: 9 features
  - Nominal: 3 features
  - Ordinal: 13 features (excluding lesion_2 and lesion_3 which were dropped)
- Target classes: Lived (46.5%), Died (33.2%), Euthanized (20.3%)

## Results

The Kaggle submission achieved a private score of `0.74242`, compared to the [leaderboard](https://www.kaggle.com/competitions/playground-series-s3e22/leaderboard) top score of `0.78181`.

Models ranked by test F1-micro (80/20 train-test split):

| Model | F1-micro | Train time (s) |
|-------|----------|----------------|
| Gradient Boosting | 0.733 | 3.3 |
| Gradient Boosting (Tuned) | 0.729 | 632.6 |
| Random Forest | 0.725 | 1.0 |
| XGBoost | 0.717 | 1.6 |
| LightGBM | 0.704 | 0.9 |
| CatBoost | 0.696 | 12.3 |
| AdaBoost | 0.672 | 0.7 |
| Logistic Regression | 0.660 | 14.5 |
| Decision Tree | 0.534 | 0.2 |
| SVC | 0.490 | 0.8 |

**Best model configuration (Gradient Boosting):**
- `n_estimators`: 221
- `learning_rate`: 0.01976
- `max_depth`: 5
- `subsample`: 0.540
- `min_samples_split`: 16
- `min_samples_leaf`: 9

## Key findings

- Default Gradient Boosting outperformed its tuned counterpart (0.733 vs 0.729), likely due to the very small dataset size making hyperparameter tuning unreliable
- Random Forest was competitive with minimal training time (0.960s)
- SVC completed evaluation on this small dataset but performed poorly (0.490), struggling with the multi-class imbalance
- The dataset is very small (1,235 samples), which limits model generalisation and makes tuning less effective
