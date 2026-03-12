"""Optuna-Tuned CatBoost — Systematic Hyperparameter Optimization."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np
import pandas as pd
import optuna
from catboost import CatBoostRegressor

from src.data import load_train, load_test, save_submission
from src.features import (
    add_all_enhanced_features_combined, GBM_FEATURES, clip_predictions,
)

TARGET = "es_total_ps"
SEEDS = [42, 123, 456, 789, 2024, 11, 22, 33, 44, 55]
N_CV_FOLDS = 5
MIN_TRAIN_FRAC = 0.4
N_OPTUNA_TRIALS = 30


def expanding_window_cv(X, y, features, params, n_folds=N_CV_FOLDS):
    """Run expanding-window CV, return mean RMSE and per-hour residuals."""
    n = len(X)
    min_train = int(n * MIN_TRAIN_FRAC)
    fold_size = (n - min_train) // n_folds
    fold_rmses = []

    for fold in range(n_folds):
        val_start = min_train + fold * fold_size
        val_end = min_train + (fold + 1) * fold_size if fold < n_folds - 1 else n
        tr_idx = list(range(val_start))
        va_idx = list(range(val_start, val_end))

        X_tr, y_tr = X.iloc[tr_idx], y.iloc[tr_idx]
        X_va, y_va = X.iloc[va_idx], y.iloc[va_idx]

        cat = CatBoostRegressor(
            iterations=3000, early_stopping_rounds=100,
            verbose=0, **params
        )
        cat.fit(X_tr[features], y_tr, eval_set=(X_va[features], y_va))
        preds = cat.predict(X_va[features])
        fold_rmse = np.sqrt(np.mean((y_va.values - preds) ** 2))
        fold_rmses.append(fold_rmse)

    return np.mean(fold_rmses)


def estimate_hourly_bias_cv(X, y, hours, features, params, n_folds=N_CV_FOLDS):
    """Estimate per-hour bias from expanding-window CV with given params."""
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
            iterations=3000, early_stopping_rounds=100,
            verbose=0, **params
        )
        cat.fit(X_tr[features], y_tr, eval_set=(X_va[features], y_va))
        preds = cat.predict(X_va[features])
        all_residuals.extend(y_va.values - preds)
        all_hours.extend(hours.iloc[va_idx].values)

    all_residuals = np.array(all_residuals)
    all_hours = np.array(all_hours)

    hourly_bias = {}
    for h in range(24):
        mask = all_hours == h
        hourly_bias[h] = float(np.mean(all_residuals[mask])) if mask.sum() > 0 else 0.0
    return hourly_bias


if __name__ == "__main__":

    # 1. Load and engineer features
    print("=" * 60)
    print("Loading data and engineering features...")
    print("=" * 60)
    train, test = add_all_enhanced_features_combined(load_train(), load_test())
    train_clean = train.dropna(subset=[TARGET]).reset_index(drop=True)
    available = [f for f in GBM_FEATURES if f in train_clean.columns]

    X_full = train_clean[available + ["hour"]]  # keep hour for bias
    y_full = train_clean[TARGET]
    print(f"Training samples: {len(train_clean)}, Features: {len(available)}")

    # 2. Optuna hyperparameter search
    print("\n" + "=" * 60)
    print(f"Step 1: Optuna search ({N_OPTUNA_TRIALS} trials, {N_CV_FOLDS}-fold CV)")
    print("=" * 60)

    def objective(trial):
        params = {
            "depth": trial.suggest_int("depth", 5, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 10.0),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.6, 1.0),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
            "random_seed": 42,
            "loss_function": "RMSE",
        }
        return expanding_window_cv(train_clean, y_full, available, params)

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=N_OPTUNA_TRIALS, show_progress_bar=True)

    best_params = study.best_params
    best_params["random_seed"] = 42
    best_params["loss_function"] = "RMSE"
    print(f"\n  Best CV RMSE: {study.best_value:.2f}")
    print(f"  Best params:")
    for k, v in best_params.items():
        print(f"    {k}: {v}")

    # 3. Per-hour bias from CV with best params
    print("\n" + "=" * 60)
    print("Step 2: Estimating per-hour bias with best params")
    print("=" * 60)
    hourly_bias = estimate_hourly_bias_cv(
        train_clean, y_full, train_clean["hour"], available, best_params
    )
    for h in range(24):
        print(f"  H{h:02d}: {hourly_bias[h]:+.1f} MW")

    # 4. Find best iteration count
    print("\n" + "=" * 60)
    print("Step 3: Finding optimal iterations (last 90 days held out)")
    print("=" * 60)
    val_cutoff = train_clean["datetime_start"].max() - pd.Timedelta(days=90)
    train_set = train_clean[train_clean["datetime_start"] <= val_cutoff]
    val_set = train_clean[train_clean["datetime_start"] > val_cutoff]

    cat_val = CatBoostRegressor(
        iterations=3000, early_stopping_rounds=100, verbose=0, **best_params
    )
    cat_val.fit(train_set[available], train_set[TARGET],
                eval_set=(val_set[available], val_set[TARGET]))
    best_iter = cat_val.get_best_iteration()
    val_rmse = np.sqrt(np.mean(
        (val_set[TARGET].values - cat_val.predict(val_set[available])) ** 2
    ))
    print(f"  Best iteration: {best_iter}, val RMSE: {val_rmse:.1f}")

    # 5. Train 10-seed ensemble
    print("\n" + "=" * 60)
    print(f"Step 4: Training {len(SEEDS)}-seed ensemble on full data")
    print("=" * 60)
    models = []
    for seed in SEEDS:
        params = {**best_params, "random_seed": seed}
        cat = CatBoostRegressor(iterations=best_iter, verbose=0, **params)
        cat.fit(train_clean[available], y_full)
        models.append(cat)
        print(f"  Seed {seed} trained ({best_iter} iterations)")

    # 6. Predict and save
    print("\n" + "=" * 60)
    print("Step 5: Generating predictions")
    print("=" * 60)
    raw_preds = np.mean([m.predict(test[available]) for m in models], axis=0)

    corrected_preds = raw_preds.copy()
    for h in range(24):
        mask = test["hour"] == h
        corrected_preds[mask] += hourly_bias.get(h, 0.0)
    final_preds = clip_predictions(corrected_preds, test)

    print(f"  Raw mean:       {raw_preds.mean():.1f} MW")
    print(f"  After bias:     {corrected_preds.mean():.1f} MW")
    print(f"  After clipping: {final_preds.mean():.1f} MW")

    test_out = test[["id"]].copy()
    test_out[TARGET] = final_preds
    save_submission(test_out, "catboost_optuna_bias.csv")

    print("\n" + "=" * 60)
    print("Done. Submission: output/catboost_optuna_bias.csv")
    print("=" * 60)
