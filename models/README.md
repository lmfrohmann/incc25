# models/: Model Implementations

All forecasting models for the INCC pumped storage competition, organized by model family.

## Structure

```
models/
├── baselines/       Simple benchmark (Random Forest)
├── boosting/        Gradient boosting (CatBoost, LightGBM, XGBoost)
├── econometric/     Econometric specifications (Elastic Net, Regime Switching, GAM, Structural Dispatch)
└── ensembles/       Forecast combination (econometric blend, multi-seed)
```

## Performance Summary

| Family | Best Model | Test RMSE | R² |
|---|---|---:|---:|
| **Ensembles** | 86% CatBoost + 14% Regime | **677.22** | 0.8322 |
| **Boosting** | CatBoost 10-seed (uniform bias) | 681.06 | 0.8303 |
| **Econometric** | Elastic Net | 797.05 | 0.7676 |
| **Baselines** | Random Forest | 751.84 | 0.7932 |

## Running Models

Every model script is self-contained and runnable from the project root:

```bash
python models/<family>/<model>.py
```

Each script:
1. Loads data via `src.data`
2. Engineers features via `src.features` (combined train+test pipeline)
3. Trains and evaluates via time series cross-validation
4. Applies per-hour bias correction (5-fold expanding window)
5. Clips to physical capacity bounds
6. Saves predictions to `output/`

Econometric models support `--bootstrap` for 95% block bootstrap confidence intervals on coefficients.

See individual subfolder READMEs for model-specific details.
