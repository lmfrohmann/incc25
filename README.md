# INCC: Spanish Pumped Storage Forecasting

Forecasting hourly net production (MW) of Spain's 16 pumped-hydro storage units for the [InCommodities Case Crunch 2025](docs/incommodities_case_crunch_2025.md) competition.

---

## Competition

| Item | Detail |
|---|---|
| **Target** | `es_total_ps`: net pumped storage output (MW). Positive = generating, negative = pumping. |
| **Competition Metric** | RMSE. |
| **Horizon** | Aug 2024 – Feb 2025 (5,065 hourly observations). |
| **Constraint** | No external data beyond the provided datasets. |

## Leaderboard (Test RMSE)

| Rank | Submission | RMSE | R² | Model |
|---:|---|---:|---:|---|
| 1 | `catboost_regime_blend_86_14` | **677.22** | 0.8322 | 86% CatBoost + 14% Regime Switching |
| 2 | `econ10_boosting90_blend` | 678.89 | 0.8315 | 10% Econometric + 90% Boosting |
| 3 | `econ15_boosting85_blend` | 679.10 | 0.8314 | 15% Econometric + 85% Boosting |
| 4 | `econ05_boosting95_blend` | 679.55 | 0.8311 | 5% Econometric + 95% Boosting |
| 5 | `econ20_boosting80_blend` | 680.17 | 0.8307 | 20% Econometric + 80% Boosting |
| 6 | `catboost_10seed_uniform_bias` | 681.06 | 0.8303 | CatBoost 10-seed + uniform bias |
| 7 | `catboost65_lgbm35_bias` | 681.67 | 0.8298 | 65% CatBoost + 35% LightGBM + bias |
| 8 | `catboost_multiseed_bias` | 685.06 | 0.8283 | CatBoost multi-seed + bias correction |
| 9 | `catboost_optuna_bias` | 686.75 | 0.8274 | Optuna-tuned CatBoost + bias |
| 10 | `catboost_tuned_hourly_bias` | 687.50 | 0.8271 | CatBoost tuned + per-hour bias |
| 11 | `elastic_net` | 797.05 | 0.7676 | Elastic Net (Ridge variant) |
| 12 | `regime_switching` | 788.93 | 0.7723 | Threshold regression (3 regimes) |
| 13 | `gam` | 832.16 | 0.7466 | GAM with Fourier seasonal structure |
| 14 | `structural_dispatch` | 861.37 | 0.7285 | Merit-order dispatch (Ridge + Fourier) |

Best result: **86% CatBoost + 14% Regime Switching blend** (RMSE 677.22). Older intermediate submissions are in `archive/output/`.

---

## Data

| Dataset | Rows | Columns | Period |
|---|---:|---:|---|
| `train.csv` | 14,351 | 58 | Dec 2022 – Jul 2024 |
| `test.csv` | 5,065 | 57 | Aug 2024 – Feb 2025 |
| `plant_metadata.csv` | 16 | 5 | Static |
| `prod_unavailable.csv` | variable | 4 | Hourly, per unit |
| `cons_unavailable.csv` | variable | 4 | Hourly, per unit |

**Target statistics** (train): mean = -165 MW, std = 1,873 MW, range = [-4,563, 4,068].

Key covariates: day-ahead (d1) and two-day-ahead (d2) forecasts for demand, wind, solar, hydro, temperature, precipitation, gas price, EUA price, and NTC across ES/FR/PT/DE. See [docs/dataset_description_2025.md](docs/dataset_description_2025.md).

**Plant fleet**: 16 Spanish pumped storage units. Total production capacity: 5,535.6 MW. Total pumping capacity: 5,252.1 MW.

---

## Project Structure

```
incc/
├── README.md
├── requirements.txt
├── .gitignore
│
├── src/                          # Core library
│   ├── data.py                   #   Data loading, path constants, I/O
│   ├── features.py               #   Feature engineering (138 features)
│   ├── metrics.py                #   RMSE, MAE, R², Diebold-Mariano test, bootstrap CIs
│   ├── fetch_actuals.py          #   Scrape ground truth from Energy-Charts API
│   └── plotting/                 #   Plotly visualization modules
│       ├── config.py             #     Palette, layout defaults
│       ├── correlations.py       #     Correlation heatmaps
│       ├── distributions.py      #     Histograms, box plots
│       ├── target_analysis.py    #     Target distribution analysis
│       ├── temporal_patterns.py  #     Hourly, weekly, monthly patterns
│       └── timeseries.py         #     Time series line charts
│
├── models/                       # All model implementations
│   ├── baselines/
│   │   └── random_forest.py      #     Scikit-learn RF (500 trees)
│   │
│   ├── boosting/
│   │   ├── catboost_final.py     #     10-seed CatBoost (best pure ML)
│   │   ├── catboost_optuna.py    #     Optuna-tuned CatBoost (Bayesian search)
│   │   ├── catboost_tuned.py     #     Single-seed CatBoost with tuning
│   │   ├── lightgbm.py           #     LightGBM with early stopping
│   │   └── xgboost_catboost.py   #     XGBoost + CatBoost comparison
│   │
│   ├── econometric/
│   │   ├── elastic_net.py        #     Kitchen-sink + regularization
│   │   ├── regime_switching.py   #     Threshold regression (pump/produce/idle)
│   │   ├── structural_dispatch.py #    Merit-order dispatch (Ridge + Fourier)
│   │   └── gam.py                #     GAM with splines + tensor product
│   │
│   └── ensembles/
│       ├── econometric_blend.py  #     ML + econometric blend (best overall)
│       └── catboost_multiseed.py #     Multi-seed CatBoost + bias correction
│
├── scripts/                      # CLI utilities
│   ├── evaluate.py               #   Score submissions against actuals
│   └── compare.py                #   Interactive Plotly comparison dashboard
│
├── notebooks/                    # Jupyter exploration
│   ├── eda.ipynb                 #   Exploratory data analysis
│   ├── data_quality.ipynb        #   Missing values, outliers
│   ├── temporal_patterns.ipynb   #   Hourly/weekly/seasonal patterns
│   ├── supply_demand.ipynb       #   Supply-demand balance analysis
│   ├── model_comparison.ipynb    #   Top-N model comparison (Plotly)
│   ├── error_analysis.ipynb      #   Error patterns by hour/regime
│   ├── feature_dynamics.ipynb    #   Feature importance over time
│   └── price_spread.ipynb        #   Price spread and arbitrage signals
│
├── data/
│   ├── raw/                      #   Competition-provided CSVs
│   └── actuals/                  #   Ground truth (from Energy-Charts API)
│
├── output/                       #   Top submission CSVs (15 files)
│
├── archive/                      #   Archived models and outputs
│
├── report/
│   └── report.pdf                #   Methodology report
│
└── docs/                         #   Competition brief & dataset description
```

---

## Methodology

### Feature Engineering (`src/features.py`)

138 engineered features organized into ten groups:

1. **Temporal encoding**: Fourier harmonics (sin/cos for hour, month, day-of-year), regime indicators (pump H10-15, produce H18-21), weekend flag.
2. **Residual demand**: `demand - wind - solar` for ES and Iberian aggregate. Strongest single predictor (r = 0.80).
3. **Renewable penetration**: Solar/wind as fraction of demand; deviation from seasonal normal.
4. **Cross-border arbitrage**: French surplus, Iberian aggregate, FR-ES NTC.
5. **Price signals**: Implied electricity price from gas + EUA, d2-d1 deltas for trend.
6. **Plant availability**: Unit-level unavailability aggregated to hourly available capacity; capacity asymmetry.
7. **Rolling statistics**: 24h/168h rolling mean/std/momentum for residual demand, wind, solar, demand (13 features). Computed on shifted series to avoid leakage.
8. **Extended lags**: 24h and 48h lags for solar, wind, demand, residual demand (8 features).
9. **Weekend/transition interactions**: Weekend × residual demand, weekend × regime, transition-hour indicator (H07-09, H16-18) × residual demand. Captures different dispatch patterns during regime changes and weekends.
10. **Wind extreme features**: Binary indicator for wind > 90th training percentile, interacted with regime. Captures non-linear dispatch during storm events.

Features are applied via `add_all_enhanced_features_combined(train, test)` which concatenates train+test before computing rolling windows, so test rows use trailing training data.

### Model Families

**Baselines**: Linear regression and random forest to establish floor performance.

**Gradient Boosting**: CatBoost, LightGBM, XGBoost. Best single framework: CatBoost with 10-seed ensembling (RMSE 681.06). Uses ordered boosting, symmetric trees (depth=8), learning_rate=0.03, l2_leaf_reg=3.0, early stopping at 3,000 max iterations, and expanding-window cross-validation with per-hour bias correction.

**Econometric Models**: Four complementary specifications, all with rolling/lag features and 5-fold per-hour bias correction:
- **Elastic Net**: Stock & Watson (2002) kitchen-sink approach. ~140 candidate regressors; regularization selects active variables.
- **Structural Dispatch**: Ridge regression on Fourier harmonics interacted with residual demand, solar penetration, and price. Models the merit-order dispatch decision. 68 features.
- **Regime-Switching**: Hansen (1999) threshold regression with three states (pump, produce, transition). Each regime has its own coefficient vector, connected by smooth logistic transitions. 170 features (44 core × 3 regimes + controls).
- **GAM**: Hastie & Tibshirani (1990) generalized additive model with penalized splines and an hour × residual_demand tensor product for time-varying dispatch effects. 32 features.

All econometric models report standardized coefficients (β_std) and support `--bootstrap` for block bootstrap 95% confidence intervals.

**Ensembles**: Best approach is a weighted blend of CatBoost + econometric models. 86% CatBoost + 14% Regime Switching = RMSE 677.22 (best overall). Weights grid-searched on validation set, not test actuals.

---

## Requirements

- Python 3.9+
- See [requirements.txt](requirements.txt) for full dependency list.

Core: `numpy`, `pandas`, `scipy`, `scikit-learn`
Boosting: `catboost`, `lightgbm`, `xgboost`, `optuna`
Econometrics: `statsmodels`, `pygam`
Visualization: `plotly`
