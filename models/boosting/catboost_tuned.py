"""Tuned CatBoost with k-fold time-series CV for robust OOF + final ensemble."""

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

LGB_PARAMS = {
    "objective": "regression", "metric": "rmse", "boosting_type": "gbdt",
    "learning_rate": 0.03, "num_leaves": 127, "min_child_samples": 20,
    "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 5,
    "reg_alpha": 0.1, "reg_lambda": 1.0, "verbose": -1, "seed": 42,
}

if __name__ == "__main__":

    # 1. Load and engineer features
    print("Loading data...")
    train, test = add_all_enhanced_features_combined(load_train(), load_test())

    train_clean = train.dropna(subset=[TARGET]).reset_index(drop=True)
    available_features = [f for f in GBM_FEATURES if f in train_clean.columns]

    X_full = train_clean[available_features]
    y_full = train_clean[TARGET]

    # 2. Time-series k-fold CV (5 expanding window folds)
    print("\n--- Time-Series 5-Fold CV ---")
    n = len(train_clean)
    min_train_size = int(n * 0.4)  # at least 40% for first fold
    fold_size = (n - min_train_size) // 5

    oof_cat = np.full(n, np.nan)
    oof_lgb = np.full(n, np.nan)
    cat_iters = []
    lgb_iters = []

    for fold in range(5):
        val_start = min_train_size + fold * fold_size
        val_end = min_train_size + (fold + 1) * fold_size if fold < 4 else n
        train_idx = list(range(val_start))
        val_idx = list(range(val_start, val_end))

        X_tr, y_tr = X_full.iloc[train_idx], y_full.iloc[train_idx]
        X_va, y_va = X_full.iloc[val_idx], y_full.iloc[val_idx]

        # CatBoost
        cat = CatBoostRegressor(
            iterations=3000, learning_rate=0.03, depth=7, l2_leaf_reg=5.0,
            subsample=0.8, colsample_bylevel=0.8, early_stopping_rounds=100,
            random_seed=42, verbose=0,
        )
        cat.fit(X_tr, y_tr, eval_set=(X_va, y_va))
        oof_cat[val_idx] = cat.predict(X_va)
        cat_iters.append(cat.get_best_iteration())

        # LightGBM
        dt = lgb.Dataset(X_tr, label=y_tr)
        dv = lgb.Dataset(X_va, label=y_va, reference=dt)
        lgb_m = lgb.train(LGB_PARAMS, dt, num_boost_round=3000,
                           valid_sets=[dv], valid_names=["val"],
                           callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)])
        oof_lgb[val_idx] = lgb_m.predict(X_va)
        lgb_iters.append(lgb_m.best_iteration)

        cat_rmse = np.sqrt(np.mean((y_va.values - oof_cat[val_idx]) ** 2))
        lgb_rmse = np.sqrt(np.mean((y_va.values - oof_lgb[val_idx]) ** 2))
        print(f"  Fold {fold+1}: train={len(train_idx)}, val={len(val_idx)}, "
              f"CatBoost RMSE={cat_rmse:.1f} (iter={cat_iters[-1]}), "
              f"LightGBM RMSE={lgb_rmse:.1f} (iter={lgb_iters[-1]})")

    # OOF scores
    valid_mask = ~np.isnan(oof_cat)
    print_metrics(y_full[valid_mask], oof_cat[valid_mask], label="CatBoost (OOF)")
    print_metrics(y_full[valid_mask], oof_lgb[valid_mask], label="LightGBM (OOF)")

    # Optimal blend weight via grid search on OOF
    best_w, best_rmse = 1.0, float("inf")
    for w in np.arange(0.0, 1.01, 0.05):
        blend = w * oof_cat[valid_mask] + (1 - w) * oof_lgb[valid_mask]
        r = np.sqrt(np.mean((y_full[valid_mask].values - blend) ** 2))
        if r < best_rmse:
            best_w, best_rmse = w, r

    print(f"\nBest blend: CatBoost weight={best_w:.2f}, OOF RMSE={best_rmse:.1f}")

    # Also try adding clipping
    blend_oof = best_w * oof_cat[valid_mask] + (1 - best_w) * oof_lgb[valid_mask]
    blend_clipped = clip_predictions(blend_oof, train_clean.iloc[np.where(valid_mask)[0]])
    rmse_clipped = np.sqrt(np.mean((y_full[valid_mask].values - blend_clipped) ** 2))
    print(f"Clipped OOF RMSE: {rmse_clipped:.1f}")

    # 3. Train final models on full data
    print("\n--- Training final models on full data ---")

    avg_cat_iter = int(np.mean(cat_iters))
    avg_lgb_iter = int(np.mean(lgb_iters))

    cat_final = CatBoostRegressor(
        iterations=avg_cat_iter, learning_rate=0.03, depth=7, l2_leaf_reg=5.0,
        subsample=0.8, colsample_bylevel=0.8, random_seed=42, verbose=0,
    )
    cat_final.fit(X_full, y_full)

    lgb_final = lgb.train(
        LGB_PARAMS, lgb.Dataset(X_full, label=y_full),
        num_boost_round=avg_lgb_iter,
    )

    # 4. Predict test
    print("\n--- Predicting test set ---")
    test_data = test.copy()
    X_test = test_data[available_features]

    cat_test = cat_final.predict(X_test)
    lgb_test = lgb_final.predict(X_test)

    # CatBoost solo (best single model)
    test_data[TARGET] = clip_predictions(cat_test, test_data)
    save_submission(test_data[["id", TARGET]], "catboost_tuned.csv")

    # Optimized blend
    blend_test = best_w * cat_test + (1 - best_w) * lgb_test
    test_data[TARGET] = clip_predictions(blend_test, test_data)
    save_submission(test_data[["id", TARGET]], "cat_lgb_blend.csv")

    # Also save median of top models (robust to outliers)
    # Load all individual predictions
    from src.data import SUBMISSIONS
    preds_list = []
    for fname in ["catboost_model.csv", "xgb_model.csv", "lgbm_model.csv"]:
        fpath = os.path.join(SUBMISSIONS, fname)
        if not os.path.exists(fpath):
            print(f"  Warning: {fname} not found, skipping")
            continue
        p = pd.read_csv(fpath)
        if len(p) != len(test_data):
            print(f"  Warning: {fname} has {len(p)} rows, expected {len(test_data)}, skipping")
            continue
        preds_list.append(p["es_total_ps"].values)

    # Add the new CatBoost tuned
    preds_list.append(cat_test)
    preds_list.append(lgb_test)

    if len(preds_list) >= 2:
        median_preds = np.median(preds_list, axis=0)
        test_data[TARGET] = clip_predictions(median_preds, test_data)
        save_submission(test_data[["id", TARGET]], "median_ensemble.csv")
    else:
        print("  Not enough models for median ensemble, skipping.")

    print("\nDone.")
