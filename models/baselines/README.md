# models/baselines/: Baseline Models

Simple models to establish performance floors and provide ensemble components.

## Models

### `random_forest.py`
- **Method**: Scikit-learn `RandomForestRegressor` (500 trees, unlimited depth, min_samples_leaf=5).
- **Features**: 23 features from `add_all_rf_features()`: time encodings, log demands, squared terms, lags, residual demand, interactions.
- **Validation**: 80/20 temporal split.
- **Test RMSE**: 751.84
- **Output**: archived

## Purpose

These baselines are intentionally simple. They serve as:
1. Lower-bound benchmarks for more complex models.
2. Diversity components in ensembles.
3. Sanity checks for the feature engineering pipeline.
