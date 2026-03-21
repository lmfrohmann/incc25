# output/: Submission Files

Top model predictions in competition submission format (`id`, `es_total_ps`).

## Active Submissions by Test RMSE

| Rank | File | RMSE | Model |
|---:|---|---:|---|
| 1 | `catboost_regime_blend_86_14.csv` | **677.22** | 86% CatBoost + 14% Regime Switching |
| 2 | `econ10_boosting90_blend.csv` | 678.89 | 10% Econometric + 90% Boosting |
| 3 | `econ15_boosting85_blend.csv` | 679.10 | 15% Econometric + 85% Boosting |
| 4 | `econ05_boosting95_blend.csv` | 679.55 | 5% Econometric + 95% Boosting |
| 5 | `econ20_boosting80_blend.csv` | 680.17 | 20% Econometric + 80% Boosting |
| 6 | `catboost_10seed_uniform_bias.csv` | 681.06 | CatBoost 10-seed + uniform bias |
| 7 | `catboost65_lgbm35_bias.csv` | 681.67 | 65% CatBoost + 35% LightGBM + bias |
| 8 | `catboost_multiseed_bias.csv` | 685.06 | CatBoost multi-seed + bias |
| 9 | `catboost_optuna_bias.csv` | 686.75 | Optuna-tuned CatBoost + bias |
| 10 | `catboost_tuned_hourly_bias.csv` | 687.50 | CatBoost tuned + per-hour bias |
| 11 | `catboost65_lgbm35_blend.csv` | 690.02 | 65% CatBoost + 35% LightGBM |
| 12 | `elastic_net.csv` | 797.05 | Elastic Net (Ridge variant) |
| 13 | `regime_switching.csv` | 788.93 | Threshold regression (3 regimes) |
| 14 | `gam.csv` | 832.16 | GAM with splines + tensor product |
| 15 | `structural_dispatch.csv` | 861.37 | Merit-order dispatch (Ridge + Fourier) |

## Evaluating

```bash
python scripts/evaluate.py output/*.csv
python scripts/evaluate.py --dm-test output/catboost_regime_blend_86_14.csv output/catboost_10seed_uniform_bias.csv
```

## Archive

`archive/output/` contains historical intermediate submissions from the model development process.
