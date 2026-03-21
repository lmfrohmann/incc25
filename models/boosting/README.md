# models/boosting/: Gradient Boosting Models

High-performance gradient boosted tree models. CatBoost is the strongest single-framework model in this project.

## Models

### `catboost_final.py` (RMSE 681.06)
- **Method**: 10-seed CatBoost ensemble with expanding-window 5-fold CV.
- **Key hyperparameters**: depth=8, learning_rate=0.03, l2_leaf_reg=3.0, up to 3,000 iterations with early stopping (patience=100).
- **Bias correction**: Per-hour bias from 5-fold CV residuals.
- **Features**: Full `GBM_FEATURES` set (138 features).
- **Output**: `output/catboost_10seed_uniform_bias.csv`, `output/catboost_tuned_hourly_bias.csv`

### `catboost_optuna.py` (RMSE 686.75)
- **Method**: 30-trial Optuna TPE search over depth, learning_rate, l2_leaf_reg, subsample, colsample, min_child_samples.
- **CV**: 5-fold expanding-window, per-hour bias correction.
- **Output**: `output/catboost_optuna_bias.csv`

### `catboost_tuned.py`
- **Method**: Single-seed CatBoost with manual tuning.
- **Output**: archived

### `lightgbm.py`
- **Method**: LightGBM with early stopping (100 rounds patience).
- **Output**: archived (used as component in blends)

### `xgboost_catboost.py`
- **Method**: XGBoost and CatBoost trained independently for ensemble diversity.
- **Output**: archived (used as component in blends)

## Why CatBoost Wins

1. **Ordered boosting** prevents target leakage in time series.
2. **Symmetric trees** provide natural regularization.
3. **Multi-seed averaging** reduces variance without increasing bias.
4. **Expanding-window CV** respects temporal ordering.
