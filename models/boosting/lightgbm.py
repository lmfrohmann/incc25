"""LightGBM model with full feature engineering and time-series CV."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np
import pandas as pd
import lightgbm as lgb

from src.data import load_train, load_test, save_submission
from src.metrics import print_metrics
from src.features import add_all_enhanced_features_combined, GBM_FEATURES, clip_predictions

TARGET = "es_total_ps"
SUBMISSION_NAME = "lgbm_model.csv"

# 1. Load and engineer features
print("Loading data...")
train, test = add_all_enhanced_features_combined(load_train(), load_test())

# Drop rows missing target
train_clean = train.dropna(subset=[TARGET]).reset_index(drop=True)

# 2. Time-series cross-validation
# Use last 3 months of training as validation (~2160 hours)
val_cutoff = train_clean["datetime_start"].max() - pd.Timedelta(days=90)
train_set = train_clean[train_clean["datetime_start"] <= val_cutoff]
val_set = train_clean[train_clean["datetime_start"] > val_cutoff]

print(f"Train: {len(train_set)} rows ({train_set.datetime_start.min()} to {train_set.datetime_start.max()})")
print(f"Val:   {len(val_set)} rows ({val_set.datetime_start.min()} to {val_set.datetime_start.max()})")

# Filter to features that exist
available_features = [f for f in GBM_FEATURES if f in train_clean.columns]
missing = [f for f in GBM_FEATURES if f not in train_clean.columns]
if missing:
    print(f"Warning: missing features: {missing}")

X_train = train_set[available_features]
y_train = train_set[TARGET]
X_val = val_set[available_features]
y_val = val_set[TARGET]

# 3. Train LightGBM
params = {
    "objective": "regression",
    "metric": "rmse",
    "boosting_type": "gbdt",
    "learning_rate": 0.03,
    "num_leaves": 127,
    "max_depth": -1,
    "min_child_samples": 20,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "verbose": -1,
    "n_jobs": -1,
    "seed": 42,
}

dtrain = lgb.Dataset(X_train, label=y_train)
dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)

print("\nTraining LightGBM...")
model = lgb.train(
    params,
    dtrain,
    num_boost_round=3000,
    valid_sets=[dtrain, dval],
    valid_names=["train", "val"],
    callbacks=[
        lgb.early_stopping(stopping_rounds=100),
        lgb.log_evaluation(period=200),
    ],
)

# Validation metrics
val_preds = model.predict(X_val)
val_preds_clipped = clip_predictions(val_preds, val_set)
print_metrics(y_val, val_preds, label="LightGBM (val, raw)")
print_metrics(y_val, val_preds_clipped, label="LightGBM (val, clipped)")

# Feature importance
importance = pd.DataFrame({
    "feature": available_features,
    "importance": model.feature_importance(importance_type="gain"),
}).sort_values("importance", ascending=False)
print("\nTop 20 features (gain):")
print(importance.head(20).to_string(index=False))

# 4. Retrain on full data and predict test
print("\nRetraining on full training data...")
X_full = train_clean[available_features]
y_full = train_clean[TARGET]
dtrain_full = lgb.Dataset(X_full, label=y_full)

model_full = lgb.train(
    params,
    dtrain_full,
    num_boost_round=model.best_iteration,
)

# Predict test
test_data = test.copy()
test_preds = model_full.predict(test_data[available_features])
test_preds = clip_predictions(test_preds, test_data)
test_data[TARGET] = test_preds

save_submission(test_data[["id", TARGET]], SUBMISSION_NAME)
print(f"\nBest iteration: {model.best_iteration}")
print(test_data[["id", TARGET]].describe())
