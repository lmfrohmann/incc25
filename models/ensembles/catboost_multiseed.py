"""Advanced CatBoost: multi-seed ensemble, bias correction, and hyperparameter grid."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
import lightgbm as lgb

from src.data import load_train, load_test, save_submission
from src.metrics import print_metrics
from src.features import add_all_enhanced_features_combined, GBM_FEATURES, clip_predictions

TARGET = "es_total_ps"

# 1. Load data
print("Loading data...")
train, test = add_all_enhanced_features_combined(load_train(), load_test())

train_clean = train.dropna(subset=[TARGET]).reset_index(drop=True)
available_features = [f for f in GBM_FEATURES if f in train_clean.columns]

X_full = train_clean[available_features]
y_full = train_clean[TARGET]

# Temporal val
val_cutoff = train_clean["datetime_start"].max() - pd.Timedelta(days=90)
train_set = train_clean[train_clean["datetime_start"] <= val_cutoff].reset_index(drop=True)
val_set = train_clean[train_clean["datetime_start"] > val_cutoff].reset_index(drop=True)
X_train, y_train = train_set[available_features], train_set[TARGET]
X_val, y_val = val_set[available_features], val_set[TARGET]

# NOTE: Known limitation — the same validation set is used for both config
# selection (grid search) and performance reporting. This introduces selection
# bias in reported metrics. Acceptable for a competition setting but would
# need a separate hold-out or nested CV for unbiased evaluation.

# 2. Hyperparameter grid search
print("\n--- Hyperparameter Search ---")
configs = [
    {"depth": 8, "l2_leaf_reg": 3.0, "lr": 0.03, "subsample": 0.8},
    {"depth": 6, "l2_leaf_reg": 5.0, "lr": 0.03, "subsample": 0.8},
    {"depth": 8, "l2_leaf_reg": 1.0, "lr": 0.03, "subsample": 0.85},
    {"depth": 7, "l2_leaf_reg": 3.0, "lr": 0.02, "subsample": 0.8},
    {"depth": 9, "l2_leaf_reg": 5.0, "lr": 0.03, "subsample": 0.75},
    {"depth": 8, "l2_leaf_reg": 3.0, "lr": 0.05, "subsample": 0.8},
    {"depth": 6, "l2_leaf_reg": 1.0, "lr": 0.03, "subsample": 0.9},
    {"depth": 8, "l2_leaf_reg": 3.0, "lr": 0.03, "subsample": 0.7},
]

best_config = None
best_rmse = float("inf")
best_iter = 0

for i, cfg in enumerate(configs):
    cat = CatBoostRegressor(
        iterations=3000, learning_rate=cfg["lr"], depth=cfg["depth"],
        l2_leaf_reg=cfg["l2_leaf_reg"], subsample=cfg["subsample"],
        colsample_bylevel=0.8, early_stopping_rounds=100,
        random_seed=42, verbose=0,
    )
    cat.fit(X_train, y_train, eval_set=(X_val, y_val))
    preds = cat.predict(X_val)
    r = np.sqrt(np.mean((y_val.values - preds) ** 2))
    print(f"  Config {i+1}: depth={cfg['depth']}, l2={cfg['l2_leaf_reg']}, "
          f"lr={cfg['lr']}, sub={cfg['subsample']} -> RMSE={r:.1f} (iter={cat.get_best_iteration()})")
    if r < best_rmse:
        best_rmse = r
        best_config = cfg
        best_iter = cat.get_best_iteration()

print(f"\nBest config: {best_config} (RMSE={best_rmse:.1f}, iter={best_iter})")

# Scale best_iter proportionally for full-data retrain to avoid data leakage
best_iter = int(best_iter * len(X_full) / len(X_train))
print(f"Scaled iter for full data: {best_iter}")

# 3. Multi-seed ensemble with best config
print("\n--- Multi-Seed CatBoost Ensemble ---")
seeds = [42, 123, 456, 789, 2024]
val_preds_all = []
test_preds_all = []

X_test = test[available_features]

for seed in seeds:
    # Train on full data
    cat = CatBoostRegressor(
        iterations=best_iter, learning_rate=best_config["lr"],
        depth=best_config["depth"], l2_leaf_reg=best_config["l2_leaf_reg"],
        subsample=best_config["subsample"], colsample_bylevel=0.8,
        random_seed=seed, verbose=0,
    )
    cat.fit(X_full, y_full)

    test_p = cat.predict(X_test)
    test_preds_all.append(test_p)

    # Also validate (train on train_set only)
    cat_val = CatBoostRegressor(
        iterations=best_iter, learning_rate=best_config["lr"],
        depth=best_config["depth"], l2_leaf_reg=best_config["l2_leaf_reg"],
        subsample=best_config["subsample"], colsample_bylevel=0.8,
        random_seed=seed, verbose=0,
    )
    cat_val.fit(X_train, y_train)
    val_p = cat_val.predict(X_val)
    val_preds_all.append(val_p)

    r = np.sqrt(np.mean((y_val.values - val_p) ** 2))
    print(f"  Seed {seed}: val RMSE={r:.1f}")

# Ensemble of seeds
ensemble_val = np.mean(val_preds_all, axis=0)
ensemble_test = np.mean(test_preds_all, axis=0)

print_metrics(y_val, ensemble_val, label="Multi-seed CatBoost (val)")

# 4. Bias correction using validation period
# Note: ensemble_val comes from cat_val models trained on train-only data (not full),
# so this bias estimation is not contaminated by val rows.
val_bias = np.mean(y_val.values - ensemble_val)
print(f"\nValidation bias: {val_bias:.1f} MW")

# Per-hour bias correction
val_set_copy = val_set.copy()
val_set_copy["pred"] = ensemble_val
val_set_copy["error"] = val_set_copy[TARGET] - val_set_copy["pred"]
hourly_bias = val_set_copy.groupby("hour")["error"].mean()
print("\nHourly bias corrections:")
print(hourly_bias.round(1).to_string())

# 5. Generate final submissions
test_data = test.copy()

# Version 1: Multi-seed ensemble (raw)
test_data[TARGET] = clip_predictions(ensemble_test, test_data)
save_submission(test_data[["id", TARGET]], "catboost_multiseed.csv")

# Version 2: With uniform bias correction
test_data[TARGET] = clip_predictions(ensemble_test + val_bias, test_data)
save_submission(test_data[["id", TARGET]], "cat_multiseed_bias.csv")

# Version 3: With per-hour bias correction
test_corrected = ensemble_test.copy()
for h in range(24):
    mask = test_data["hour"] == h
    if h in hourly_bias.index:
        test_corrected[mask.values] += hourly_bias[h]
test_data[TARGET] = clip_predictions(test_corrected, test_data)
save_submission(test_data[["id", TARGET]], "cat_multiseed_hourly_bias.csv")

# Version 4: Blend multi-seed CatBoost + LightGBM (0.65/0.35 from CV)
from models import DEFAULT_LGB_PARAMS
lgb_params = {**DEFAULT_LGB_PARAMS}

# Use best_iter from the original CatBoost as a proxy for LightGBM
dt = lgb.Dataset(X_train, label=y_train)
dv = lgb.Dataset(X_val, label=y_val, reference=dt)
lgb_m = lgb.train(lgb_params, dt, num_boost_round=3000,
                   valid_sets=[dv], valid_names=["val"],
                   callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)])
lgb_full = lgb.train(lgb_params, lgb.Dataset(X_full, label=y_full),
                      num_boost_round=lgb_m.best_iteration)
lgb_test = lgb_full.predict(X_test)

blend_65 = 0.65 * ensemble_test + 0.35 * lgb_test
test_data[TARGET] = clip_predictions(blend_65, test_data)
save_submission(test_data[["id", TARGET]], "catboost65_lgbm35_blend.csv")

# Blend with bias correction
blend_65_bias = blend_65 + val_bias
test_data[TARGET] = clip_predictions(blend_65_bias, test_data)
save_submission(test_data[["id", TARGET]], "catboost65_lgbm35_bias.csv")

print("\nDone.")
