# scripts/: CLI Utilities

Command-line tools for evaluating and comparing model submissions.

## Scripts

### `evaluate.py`
Score one or more submission CSVs against test actuals.

```bash
python scripts/evaluate.py output/*.csv
python scripts/evaluate.py output/catboost_regime_blend_86_14.csv
python scripts/evaluate.py --dm-test output/catboost_regime_blend_86_14.csv output/catboost_10seed_uniform_bias.csv
```

Reports: RMSE, MAE, R², and number of matched observations. The `--dm-test` flag runs a pairwise Diebold-Mariano test (Newey-West HAC standard errors).

### `compare.py`
Interactive Plotly comparison dashboard for visual model diagnostics.

```bash
python scripts/compare.py output/catboost_regime_blend_86_14.csv output/catboost_10seed_uniform_bias.csv
```

Generates four interactive plots:
1. **Time series**: Actuals vs. predictions with 24h moving average.
2. **Scatter**: Predicted vs. actual with 45-degree reference line.
3. **Error distribution**: Histogram of (predicted - actual) per model.
4. **Hourly error**: Mean error by hour-of-day.
