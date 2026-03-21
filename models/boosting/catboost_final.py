"""Final Competition Submission — Spanish Pumped Storage Forecasting."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

from src.data import load_train, load_test, save_submission
from src.metrics import print_metrics
from src.features import (
    add_all_enhanced_features_combined,
    GBM_FEATURES, clip_predictions,
)

TARGET = "es_total_ps"
SEEDS = [42, 123, 456, 789, 2024, 11, 22, 33, 44, 55]
N_CV_FOLDS = 5
MIN_TRAIN_FRAC = 0.4  # first fold uses at least 40% of data for training

# CatBoost hyperparameters (selected via grid search on validation)
CAT_PARAMS = dict(
    learning_rate=0.03,
    depth=8,
    l2_leaf_reg=3.0,
    subsample=0.8,
    colsample_bylevel=0.8,
    random_seed=42,
    verbose=0,
)


def estimate_hourly_bias_cv(X, y, hours, features, n_folds=N_CV_FOLDS):
    """Estimate per-hour prediction bias via expanding-window CV."""
    n = len(X)
    min_train = int(n * MIN_TRAIN_FRAC)
    fold_size = (n - min_train) // n_folds
    all_residuals = []
    all_hours = []

    for fold in range(n_folds):
        val_start = min_train + fold * fold_size
        val_end = min_train + (fold + 1) * fold_size if fold < n_folds - 1 else n
        tr_idx = list(range(val_start))
        va_idx = list(range(val_start, val_end))

        X_tr, y_tr = X.iloc[tr_idx], y.iloc[tr_idx]
        X_va, y_va = X.iloc[va_idx], y.iloc[va_idx]

        cat = CatBoostRegressor(
            iterations=3000, early_stopping_rounds=100, **CAT_PARAMS
        )
        cat.fit(X_tr[features], y_tr, eval_set=(X_va[features], y_va))

        preds = cat.predict(X_va[features])
        residuals = y_va.values - preds
        fold_rmse = np.sqrt(np.mean(residuals ** 2))
        print(f"  Fold {fold + 1}/{n_folds}: "
              f"train={len(tr_idx)}, val={len(va_idx)}, RMSE={fold_rmse:.1f}")
        all_residuals.extend(residuals)
        all_hours.extend(hours.iloc[va_idx].values)

    all_residuals = np.array(all_residuals)
    all_hours = np.array(all_hours)

    hourly_bias = {}
    print("\n  Per-hour bias:")
    for h in range(24):
        mask = all_hours == h
        if mask.sum() > 0:
            hourly_bias[h] = float(np.mean(all_residuals[mask]))
            print(f"    H{h:02d}: {hourly_bias[h]:+.1f} MW (n={mask.sum()})")
        else:
            hourly_bias[h] = 0.0
    return hourly_bias


def find_best_iterations(X_train, y_train, X_val, y_val, features):
    """Find optimal boosting iterations via early stopping on temporal val."""
    cat = CatBoostRegressor(
        iterations=3000, early_stopping_rounds=100, **CAT_PARAMS
    )
    cat.fit(X_train[features], y_train, eval_set=(X_val[features], y_val))
    best_iter = cat.get_best_iteration()
    val_rmse = np.sqrt(np.mean((y_val.values - cat.predict(X_val[features])) ** 2))
    print(f"  Best iteration: {best_iter}, val RMSE: {val_rmse:.1f}")
    return best_iter


def train_multiseed_ensemble(X, y, features, n_iterations, seeds=SEEDS):
    """Train N CatBoost models with different random seeds on the same data."""
    models = []
    for seed in seeds:
        params = {**CAT_PARAMS, "random_seed": seed}
        cat = CatBoostRegressor(iterations=n_iterations, **params)
        cat.fit(X[features], y)
        models.append(cat)
        print(f"  Seed {seed} trained ({n_iterations} iterations)")
    return models


def predict_ensemble(models, X, features):
    """Average predictions across all models in the ensemble."""
    preds = np.array([m.predict(X[features]) for m in models])
    return preds.mean(axis=0)


# ============================================================================
# Main
# ============================================================================
if __name__ == "__main__":

    # 1. Load and engineer features
    print("=" * 60)
    print("Loading data and engineering features...")
    print("=" * 60)
    train, test = add_all_enhanced_features_combined(load_train(), load_test())

    train_clean = train.dropna(subset=[TARGET]).reset_index(drop=True)
    available = [f for f in GBM_FEATURES if f in train_clean.columns]
    print(f"Training samples: {len(train_clean)}")
    print(f"Test samples:     {len(test)}")
    print(f"Features:         {len(available)}")

    X_full = train_clean[available]
    y_full = train_clean[TARGET]

    # 2. Determine best iteration count (temporal validation)
    # NOTE: The last 90 days used here for best_iter selection overlap with the
    # last fold of the expanding-window bias CV below. This means the bias
    # estimate from the last fold is not fully independent of iteration selection.
    # Accepted trade-off: fixing this would require shrinking the bias CV data
    # or the validation window, reducing statistical power in both cases.
    print("\n" + "=" * 60)
    print("Step 1: Finding optimal iteration count (last 90 days held out)")
    print("=" * 60)
    val_cutoff = train_clean["datetime_start"].max() - pd.Timedelta(days=90)
    train_set = train_clean[train_clean["datetime_start"] <= val_cutoff]
    val_set = train_clean[train_clean["datetime_start"] > val_cutoff]
    best_iter = find_best_iterations(
        train_set, train_set[TARGET], val_set, val_set[TARGET], available
    )

    # 3. Estimate per-hour bias via k-fold expanding-window CV
    print("\n" + "=" * 60)
    print(f"Step 2: Estimating per-hour bias ({N_CV_FOLDS}-fold expanding-window CV)")
    print("=" * 60)
    hourly_bias = estimate_hourly_bias_cv(
        train_clean, y_full, train_clean["hour"], available
    )

    # 4. Train final 10-seed ensemble on full training data
    print("\n" + "=" * 60)
    print(f"Step 3: Training {len(SEEDS)}-seed ensemble on full training data")
    print("=" * 60)
    models = train_multiseed_ensemble(train_clean, y_full, available, best_iter)

    # 5. Predict test set
    print("\n" + "=" * 60)
    print("Step 4: Generating predictions")
    print("=" * 60)
    raw_preds = predict_ensemble(models, test, available)

    # Apply per-hour bias correction
    corrected_preds = raw_preds.copy()
    for h in range(24):
        mask = test["hour"] == h
        corrected_preds[mask] += hourly_bias.get(h, 0.0)
    final_preds = clip_predictions(corrected_preds, test)

    avg_bias = np.mean(list(hourly_bias.values()))
    print(f"  Raw mean:              {raw_preds.mean():.1f} MW")
    print(f"  + Per-hour bias (avg {avg_bias:+.1f}): {corrected_preds.mean():.1f} MW")
    print(f"  After clipping:        {final_preds.mean():.1f} MW")

    # 6. Save
    test_out = test[["id"]].copy()
    test_out[TARGET] = final_preds
    save_submission(test_out, "catboost_tuned_hourly_bias.csv")

    # Also save original uniform-bias version for comparison
    uniform_bias = avg_bias
    uniform_preds = clip_predictions(raw_preds + uniform_bias, test)
    test_uniform = test[["id"]].copy()
    test_uniform[TARGET] = uniform_preds
    save_submission(test_uniform, "catboost_10seed_uniform_bias.csv")

    print("\n" + "=" * 60)
    print("Done. Submissions: output/catboost_tuned_hourly_bias.csv, output/catboost_10seed_uniform_bias.csv")
    print("=" * 60)
