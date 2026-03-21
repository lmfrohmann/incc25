# models/ensembles/: Forecast Combination

Ensemble methods that combine outputs from multiple models to reduce variance and exploit error diversity.

## Models

### `econometric_blend.py` (RMSE 677.22)
- **Method**: Constrained optimization to find weights minimizing RMSE on a held-out validation set (last 90 days).
- **Best blend**: 86% CatBoost + 14% Regime Switching.
- **Also produces**: various econ/boosting weight ratio blends (RMSE 678-680).
- **Output**: `output/catboost_regime_blend_86_14.csv`, `output/econ*_boosting*_blend.csv`

### `catboost_multiseed.py` (RMSE 685.06)
- **Method**: 10 CatBoost models with different random seeds, averaged. Expanding-window bias correction.
- **Output**: `output/catboost_multiseed_bias.csv`

## Archived

`full_stack.py`, `stacked.py`, `blend_rf_lr.py` are in `archive/models/ensembles/`. The simple 2-model blend outperforms complex stacked architectures on this dataset.

## Key Insight

The best ensemble (RMSE 677.22) is a simple weighted average, not a complex stacked architecture. Combining structurally diverse models (econometric + ML) beats stacking similar ones.
