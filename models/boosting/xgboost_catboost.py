"""XGBoost and CatBoost models for ensemble diversity."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np
import pandas as pd
import xgboost as xgb
from catboost import CatBoostRegressor

from src.data import load_train, load_test, save_submission
from src.metrics import print_metrics
from src.features import add_all_enhanced_features_combined, GBM_FEATURES, clip_predictions

TARGET = "es_total_ps"

if __name__ == "__main__":

    # 1. Load and engineer features
    print("Loading data...")
    train, test = add_all_enhanced_features_combined(load_train(), load_test())

    train_clean = train.dropna(subset=[TARGET]).reset_index(drop=True)

    available_features = [f for f in GBM_FEATURES if f in train_clean.columns]

    # Time-series validation
    val_cutoff = train_clean["datetime_start"].max() - pd.Timedelta(days=90)
    train_set = train_clean[train_clean["datetime_start"] <= val_cutoff]
    val_set = train_clean[train_clean["datetime_start"] > val_cutoff]

    X_train = train_set[available_features]
    y_train = train_set[TARGET]
    X_val = val_set[available_features]
    y_val = val_set[TARGET]
    X_full = train_clean[available_features]
    y_full = train_clean[TARGET]

    # 2. XGBoost
    print("\n=== XGBoost ===")
    xgb_params = {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "learning_rate": 0.03,
        "max_depth": 8,
        "min_child_weight": 20,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "tree_method": "hist",
        "seed": 42,
        "verbosity": 0,
    }

    dtrain_xgb = xgb.DMatrix(X_train, label=y_train)
    dval_xgb = xgb.DMatrix(X_val, label=y_val)
    dfull_xgb = xgb.DMatrix(X_full, label=y_full)

    xgb_model = xgb.train(
        xgb_params, dtrain_xgb,
        num_boost_round=3000,
        evals=[(dtrain_xgb, "train"), (dval_xgb, "val")],
        early_stopping_rounds=100,
        verbose_eval=200,
    )

    xgb_val_preds = xgb_model.predict(dval_xgb)
    xgb_val_clipped = clip_predictions(xgb_val_preds, val_set)
    print_metrics(y_val, xgb_val_clipped, label="XGBoost (val, clipped)")

    # Retrain on full and predict test
    xgb_full = xgb.train(xgb_params, dfull_xgb, num_boost_round=xgb_model.best_iteration)

    test_data = test.copy()
    dtest_xgb = xgb.DMatrix(test_data[available_features])
    xgb_test_preds = clip_predictions(xgb_full.predict(dtest_xgb), test_data)
    test_data[TARGET] = xgb_test_preds
    save_submission(test_data[["id", TARGET]], "xgb_model.csv")

    # 3. CatBoost
    print("\n=== CatBoost ===")
    cat_model = CatBoostRegressor(
        iterations=3000,
        learning_rate=0.03,
        depth=8,
        l2_leaf_reg=3.0,
        subsample=0.8,
        colsample_bylevel=0.8,
        early_stopping_rounds=100,
        random_seed=42,
        verbose=200,
    )

    cat_model.fit(
        X_train, y_train,
        eval_set=(X_val, y_val),
    )

    cat_val_preds = cat_model.predict(X_val)
    cat_val_clipped = clip_predictions(cat_val_preds, val_set)
    print_metrics(y_val, cat_val_clipped, label="CatBoost (val, clipped)")

    # Retrain on full
    best_iter = cat_model.get_best_iteration()
    cat_full = CatBoostRegressor(
        iterations=best_iter,
        learning_rate=0.03,
        depth=8,
        l2_leaf_reg=3.0,
        subsample=0.8,
        colsample_bylevel=0.8,
        random_seed=42,
        verbose=0,
    )
    cat_full.fit(X_full, y_full)

    cat_test_preds = clip_predictions(cat_full.predict(test_data[available_features]), test_data)
    test_data[TARGET] = cat_test_preds
    save_submission(test_data[["id", TARGET]], "catboost_model.csv")

    print("\nDone. Submissions saved: xgb_model.csv, catboost_model.csv")
