# notebooks/: Exploratory Analysis

Jupyter notebooks for data exploration, quality checks, pattern discovery, and model comparison.

## Notebooks

| Notebook | Purpose |
|---|---|
| `eda.ipynb` | Initial exploratory data analysis: distributions, correlations, missing values, target statistics. |
| `data_quality.ipynb` | Data quality audit: missing values, outliers, timestamp gaps, consistency checks across d1/d2 forecasts. |
| `temporal_patterns.ipynb` | Hourly, weekly, and seasonal patterns in pumped storage output. Identifies the pump (H10-15) / produce (H18-21) regime. |
| `supply_demand.ipynb` | Supply-demand balance analysis: residual demand, renewable penetration, cross-border flows and their relationship with storage dispatch. |
| `model_comparison.ipynb` | Top-6 model comparison against test actuals: time series overlays, scatter plots, error distributions, hourly RMSE, and weekly rolling RMSE. All charts are interactive Plotly. |
| `error_analysis.ipynb` | Detailed error analysis of the best blend submission: residual patterns, worst-hour breakdown, and error correlations with features. |
| `feature_dynamics.ipynb` | Feature importance dynamics across CV folds and temporal stability of key predictors. |
| `price_spread.ipynb` | Price spread analysis: gas/EUA cost curves, implied electricity price, and relationship with storage dispatch timing. |

## Key Discoveries

- **Autocorrelation**: lag-1h = 0.91, lag-24h = 0.75. Strong persistence and diurnal cycle.
- **Residual demand**: Correlation with target r = 0.80. Single strongest predictor.
- **Solar penetration**: Drives pumping behavior during midday hours.
- **Regime structure**: Clear pump/produce asymmetry by hour-of-day, motivating regime-switching models.

## Usage

```bash
jupyter notebook notebooks/
```

Requires: `jupyter`, `plotly`, and all dependencies in `requirements.txt`.
