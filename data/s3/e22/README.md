# Predict Health Outcomes of Horses

**Competition:** [Predict Health Outcomes of Horses](https://www.kaggle.com/competitions/playground-series-s3e22)

A multi-class classification project predicting horse health outcomes (lived, died, or euthanized). Submissions are evaluated using micro-averaged F1 score between the predicted value and the observed target `outcome`.

## Dataset

```bash
kaggle competitions download -c playground-series-s3e22
```

- Training samples: 1,235
- Target classes: Lived (46%), Died (33%), Euthanized (20%)
- Features: 25 total
  - Numerical: 9 (`abdomo_protein`, `hospital_number`, `lesion_1`, `nasogastric_reflux_ph`, `packed_cell_volume`, `pulse`, `rectal_temp`, `respiratory_rate`, `total_protein`)
  - Nominal: 3 (`abdomen`, `abdomo_appearance`, `mucous_membrane`)
  - Ordinal: 13 (`abdominal_distention`, `age`, `capillary_refill_time`, `cp_data`, `nasogastric_reflux`, `nasogastric_tube`, `pain`, `peripheral_pulse`, `peristalsis`, `rectal_exam_feces`, `surgery`, `surgical_lesion`, `temp_of_extremities`; excludes `lesion_2` and `lesion_3`, which were dropped)
- Missing values: present in 12 features (up to 213 missing per feature)

## Results

The Kaggle submission achieved a private score of `0.73787`, compared to the [leaderboard](https://www.kaggle.com/competitions/playground-series-s3e22/leaderboard) top score of `0.78181`.

Models ranked by test accuracy (equivalent to micro-averaged F1 for this single-label problem) (80/20 train-test split):

| Model | Accuracy | Train time (s) |
|-------|----------|-----------------|
| Gradient Boosting | 0.733 | 2.6 |
| Random Forest | 0.725 | 0.9 |
| LightGBM (Tuned) | 0.725 | 21.2 |
| XGBoost | 0.704 | 1.0 |
| LightGBM | 0.704 | 0.7 |
| CatBoost | 0.696 | 11.1 |
| AdaBoost | 0.672 | 0.5 |
| Logistic Regression | 0.660 | 9.4 |
| Decision Tree | 0.534 | 0.2 |
| SVC | 0.526 | 1.1 |

**Best model configuration (LightGBM):**
- `n_estimators`: 374
- `num_leaves`: 82
- `learning_rate`: 0.01175
- `subsample`: 0.6456
- `colsample_bytree`: 0.8059
- `min_child_samples`: 45
- `reg_alpha`: 4.26e-06
- `reg_lambda`: 1.98e-05

## Key findings

- Only 1,235 rows total, so the test split is ~247 rows. The untuned Gradient Boosting model scoring higher than the tuned LightGBM (0.733 vs 0.725) is most likely noise, not a real difference.
- Several columns needed manual cleanup of messy categories (`pain`, `peristalsis`, `capillary_refill_time`, `nasogastric_reflux`), and `lesion_2`/`lesion_3` were dropped since almost every row had the same value.
