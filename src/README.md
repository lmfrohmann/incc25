# src/: Core Library

Shared data loading, feature engineering, evaluation metrics, and visualization modules used by all models and scripts.

## Modules

| Module | Description |
|---|---|
| `data.py` | Data I/O. Loads train/test/actuals/metadata CSVs with datetime parsing. Defines path constants (`DATA_RAW`, `SUBMISSIONS`). `save_submission()` writes output CSVs. |
| `features.py` | Feature engineering. Two pipelines: `add_all_rf_features()` for baselines (23 features) and `add_all_enhanced_features()` for competition models (138 features). Defines `GBM_FEATURES` list. |
| `metrics.py` | `rmse()`, `mae()`, `r_squared()`, `print_metrics()`. Also includes `diebold_mariano()` for pairwise forecast comparison (Newey-West HAC standard errors) and `block_bootstrap_coefs()` for coefficient confidence intervals (circular block bootstrap, block=24h). |
| `fetch_actuals.py` | One-time script to fetch ground truth from the Energy-Charts API (Fraunhofer ISE). Pulls 15-min Spanish pumped storage data, resamples to hourly, and aligns with `test.csv` IDs. |

## plotting/

Plotly-based visualization library for notebooks and comparison scripts.

| Module | Description |
|---|---|
| `config.py` | Color palette, layout defaults, `apply_layout()` helper. |
| `correlations.py` | Correlation heatmaps for feature selection. |
| `distributions.py` | Histograms, box plots for variable distributions. |
| `target_analysis.py` | Target (`es_total_ps`) distribution and regime analysis. |
| `temporal_patterns.py` | Hourly, weekly, and monthly pattern charts. |
| `timeseries.py` | Time series line charts with moving averages. |

## Feature Engineering Pipeline

`add_all_enhanced_features()` applies these transforms in order:

1. **Unavailability**: Merges unit-level unavailability into hourly totals; derives available production/pumping capacity.
2. **Time features**: Cyclical encoding (sin/cos), regime indicator, weekend flag.
3. **Spread features**: Residual demand, solar penetration, Iberian aggregates, French surplus, d2-d1 deltas, price signals.
4. **Lag features**: 1-period lags for solar, wind, demand; log transforms; squared terms.
5. **Rolling features**: 24h/168h rolling means and standard deviations for demand, wind, solar, and residual demand.
6. **Extended lag features**: 24h and 48h lags for key variables (demand, wind, solar, residual demand, hydro balance).
7. **Interactions**: Hour x residual demand, regime x solar, capacity interactions, weekend x transition hour indicators, wind extreme flags.

Output: DataFrame with all original columns plus 138 engineered features, ready for model training.
