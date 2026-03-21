"""Evaluate submission CSVs against test actuals.

Usage:
    python scripts/evaluate.py output/*.csv
    python scripts/evaluate.py --dm-test output/model_a.csv output/model_b.csv
"""

import sys
import os
import glob

# Allow imports from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data import load_actuals
from src.metrics import rmse, mae, r_squared, adjusted_r_squared, diebold_mariano

import pandas as pd
import numpy as np


def load_predictions(path, actuals):
    """Load a submission and merge with actuals, returning aligned arrays."""
    preds = pd.read_csv(path)
    merged = preds.merge(actuals, on="id")
    y_true = merged["es_total_ps_actual"]
    y_pred = merged["es_total_ps"]
    valid = y_true.notna() & y_pred.notna()
    return y_true[valid].values, y_pred[valid].values


def evaluate_submission(path, actuals, n_features=111):
    """Evaluate a single submission file and return metrics dict."""
    y_true, y_pred = load_predictions(path, actuals)
    return {
        "file": os.path.basename(path),
        "rmse": rmse(y_true, y_pred),
        "mae": mae(y_true, y_pred),
        "r2": r_squared(y_true, y_pred),
        "adj_r2": adjusted_r_squared(y_true, y_pred, n_features),
        "n": len(y_true),
    }


def run_dm_test(paths, actuals):
    """Run pairwise Diebold-Mariano tests between all submissions."""
    # Load all predictions
    models = {}
    for path in paths:
        name = os.path.basename(path)
        try:
            y_true, y_pred = load_predictions(path, actuals)
            models[name] = (y_true, y_pred)
        except Exception as e:
            print(f"  Skipping {name}: {e}")

    if len(models) < 2:
        print("Need at least 2 submissions for DM test.")
        return

    names = list(models.keys())
    y_true = list(models.values())[0][0]  # same actuals for all

    # Print individual RMSEs first
    print(f"\n{'Model':<40} {'RMSE':>10}")
    print("-" * 52)
    for name in names:
        r = rmse(y_true, models[name][1])
        print(f"{name:<40} {r:>10.2f}")

    # Pairwise DM tests
    print(f"\n{'Model 1':<25} {'Model 2':<25} {'DM stat':>8} {'p-value':>8} {'Result'}")
    print("-" * 95)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            result = diebold_mariano(y_true, models[names[i]][1], models[names[j]][1])
            print(f"{names[i]:<25} {names[j]:<25} "
                  f"{result['statistic']:>+8.3f} {result['p_value']:>8.4f} "
                  f"{result['conclusion']}")
    print()


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    # Check for --dm-test flag
    dm_test = "--dm-test" in sys.argv
    args = [a for a in sys.argv[1:] if a != "--dm-test"]

    # Expand globs (for shells that don't do it automatically)
    paths = []
    for arg in args:
        expanded = glob.glob(arg)
        paths.extend(expanded if expanded else [arg])

    actuals = load_actuals()

    if dm_test:
        run_dm_test(paths, actuals)
        return

    results = []
    for path in paths:
        if not os.path.isfile(path):
            print(f"  Skipping {path} (not found)")
            continue
        try:
            result = evaluate_submission(path, actuals)
            results.append(result)
        except Exception as e:
            print(f"  Error evaluating {path}: {e}")

    if not results:
        print("No valid submissions to evaluate.")
        sys.exit(1)

    # Sort by RMSE (best first)
    results.sort(key=lambda r: r["rmse"])

    # Print table
    print(f"\n{'File':<40} {'RMSE':>10} {'MAE':>10} {'R²':>10} {'Adj. R²':>10} {'N':>6}")
    print("-" * 90)
    for r in results:
        print(f"{r['file']:<40} {r['rmse']:>10.2f} {r['mae']:>10.2f} {r['r2']:>10.4f} {r['adj_r2']:>10.4f} {r['n']:>6}")
    print()


if __name__ == "__main__":
    main()
