# models/econometric/: Econometric Models

Classical econometric and semi-parametric specifications designed for interpretability and ensemble diversity. These models make structurally different errors from gradient boosting, making them valuable for forecast combination.

All models use:
- `add_all_enhanced_features_combined(train, test)`, concatenated pipeline for rolling/lag features
- **5-fold expanding-window per-hour bias correction**
- Rolling statistics (13 features) and extended 24h/48h lags (8 features)

## Models

### `elastic_net.py` (RMSE 797.05)
- **Method**: Compares ElasticNetCV, LassoCV, and RidgeCV with `TimeSeriesSplit(n_splits=5)`. Ridge wins.
- **Features**: ~138 candidates including all fundamentals, Fourier harmonics, interactions, rolling stats, extended lags.
- **Output**: `output/elastic_net.csv`

### `regime_switching.py` (RMSE 788.93)
- **Method**: Three-regime linear model (pumping H10-15, generating H18-21, transition). Each regime has its own coefficient vector. Smooth logistic transitions.
- **Features**: 170 features: 44 base x 3 regimes + controls. Ridge regularization (alpha=100).
- **Output**: `output/regime_switching.csv`

### `structural_dispatch.py` (RMSE 861.37)
- **Method**: Ridge regression on Fourier harmonics interacted with residual demand, solar penetration, and implied electricity price.
- **Features**: 68 features.
- **Output**: `output/structural_dispatch.csv`

### `gam.py` (RMSE 832.16)
- **Method**: `pygam.LinearGAM` with penalized cubic splines and hour x residual_demand tensor product.
- **Features**: 32 smooth/linear terms.
- **Output**: `output/gam.csv`

## Ensemble Value

Individual econometric RMSE scores (788-862) are weaker than ML (681), but they contribute meaningfully to ensembles by making structurally different errors:

| Blend | RMSE |
|---|---:|
| 86% CatBoost + 14% Regime Switching | **677.22** |
| 100% CatBoost (best pure ML) | 681.06 |
