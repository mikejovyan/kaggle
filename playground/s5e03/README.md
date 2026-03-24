# Binary Prediction with a Rainfall Dataset

**Competition:** [Binary Prediction with a Rainfall Dataset](https://www.kaggle.com/competitions/playground-series-s5e3)

A binary classification project predicting whether rainfall will occur on a given day. Submissions are evaluated on area under the ROC curve between the predicted probability and the observed target `rainfall`.

## Dataset

```bash
kaggle competitions download -c playground-series-s5e3
```

- Training samples: 1,825 (temporal split; final year used as test set)
- Features: 11 total (excluding target)
  - Numerical: 11 features
- Target classes: 0/No rainfall (24.7%), 1/Rainfall (75.3%)

## Results

The Kaggle submission achieved a private score of `0.90335`, compared to the [leaderboard](https://www.kaggle.com/competitions/playground-series-s5e3/leaderboard) top score of `0.90654`.

Models ranked by test ROC-AUC (temporal split — last year held out):

| Model | ROC-AUC | Train time (s) |
|-------|---------|----------------|
| Gradient Boosting | 0.886 | 1.4 |
| AdaBoost | 0.885 | 0.5 |
| Logistic Regression | 0.885 | 0.2 |
| CatBoost (Tuned) | 0.877 | 119.1 |
| CatBoost | 0.874 | 5.1 |
| XGBoost | 0.869 | 0.4 |
| LightGBM | 0.868 | 0.3 |
| Random Forest | 0.860 | 1.1 |
| SVC | 0.839 | 1.0 |
| Decision Tree | 0.760 | 0.1 |

**Best model configuration (CatBoost, tuned with Optuna):**
- `iterations`: 946
- `depth`: 3
- `learning_rate`: 0.01147
- `l2_leaf_reg`: 9.901
- `subsample`: 0.522

## Key findings

- Local test results are unreliable due to the small dataset size (365 test samples); the Kaggle private score is the more meaningful measure of model performance
- CatBoost (Tuned) ranked 4th locally (0.877 ROC-AUC) but achieved the best Kaggle private score (0.90335), confirming the local test set is too small to distinguish models reliably
- CatBoost tuning with Optuna (30 trials) improved CV score from 0.890 to 0.896 and paid off on the actual leaderboard despite appearing to underperform locally
- Target is imbalanced (75.3% rainfall vs 24.7% no rainfall), with all boosting methods handling this without resampling
