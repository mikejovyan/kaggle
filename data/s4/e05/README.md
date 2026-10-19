# Regression with a Flood Prediction Dataset

**Competition:** [Regression with a Flood Prediction Dataset](https://www.kaggle.com/competitions/playground-series-s4e5)

A regression project predicting the probability of flooding based on environmental and infrastructural factors. Submissions are evaluated using R² score between the predicted value and the observed target `FloodProbability`.

## Dataset

```bash
kaggle competitions download -c playground-series-s4e5
```

- Training samples: 1,117,957
- Target: mean ± std = 0.505 ± 0.051, range [0.285, 0.725]
- Features: 22 total
  - Numerical: 21 (`AgriculturalPractices`, `ClimateChange`, `CoastalVulnerability`, `DamsQuality`, `Deforestation`, `DeterioratingInfrastructure`, `DrainageSystems`, `Encroachments`, `InadequatePlanning`, `IneffectiveDisasterPreparedness`, `Landslides`, `MonsoonIntensity`, `PoliticalFactors`, `PopulationScore`, `RiverManagement`, `Siltation`, `SumOfFeatures`, `TopographyDrainage`, `Urbanization`, `Watersheds`, `WetlandLoss`)
  - Ordinal: 1 (`IsSumBetween72And75`)
- Missing values: none

## Results

The Kaggle submission achieved a private score of `0.86686`, compared to the [leaderboard](https://www.kaggle.com/competitions/playground-series-s4e5/leaderboard) top score of `0.86905`.

Models ranked by test R² (80/20 train-test split):

| Model | R² | Train time (s) |
|-------|----|-----------------|
| CatBoost | 0.867 | 155.5 |
| CatBoost (Tuned) | 0.867 | 1506.4 |
| LightGBM | 0.867 | 13.6 |
| XGBoost | 0.867 | 12.0 |
| Ridge | 0.851 | 3.6 |
| Decision Tree | 0.726 | 63.4 |

**Best model configuration (CatBoost):**
- `iterations`: 419
- `depth`: 8
- `learning_rate`: 0.11595
- `l2_leaf_reg`: 4.6188
- `subsample`: 0.7908

## Key findings

- CatBoost, LightGBM, and XGBoost all converged to R² = 0.867 with default parameters, leaving little room for tuning gains; the tuned CatBoost matched the default on the local test set despite a 10x longer training time (1506.4s vs 155.5s)
- Two features were engineered: `SumOfFeatures` (row-wise sum of all inputs, useful for tree models) and `IsSumBetween72And75` (binary flag for a specific sum range, useful for linear models)
- Train and test distributions were indistinguishable (adversarial balanced accuracy = 0.50), indicating no distribution shift
- The target range is narrow (0.285–0.725), which likely constrains the ceiling on R²
- Ridge (0.851) trailed the boosting models by only 0.016 R², suggesting the engineered `IsSumBetween72And75` feature captured most of the non-linear structure needed for a linear model
