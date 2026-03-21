"""Random Forest baseline model."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np
from sklearn.ensemble import RandomForestRegressor

from src.data import load_train, load_test, save_submission
from src.metrics import print_metrics
from src.features import add_all_rf_features, RF_FEATURES

TARGET = "es_total_ps"
SUBMISSION_NAME = "rf_baseline.csv"

# 1. Load and engineer features
train = add_all_rf_features(load_train())
test = add_all_rf_features(load_test())

train_clean = train.dropna(subset=[TARGET] + RF_FEATURES).reset_index(drop=True)

# 2. Temporal validation split (80% train / 20% validation)
n = len(train_clean)
split = int(0.8 * n)
train_set = train_clean.iloc[:split]
val_set = train_clean.iloc[split:]

# 3. Train and validate
rf = RandomForestRegressor(
    n_estimators=500,
    max_features=len(RF_FEATURES) // 3,  # match R's mtry = floor(p/3)
    min_samples_leaf=5,                   # match R's nodesize = 5
    random_state=123,
    n_jobs=-1,
)
rf.fit(train_set[RF_FEATURES], train_set[TARGET])
print_metrics(val_set[TARGET], rf.predict(val_set[RF_FEATURES]), label="RF (validation)")

# 4. Retrain on full data and predict test
rf.fit(train_clean[RF_FEATURES], train_clean[TARGET])

test_data = test[["id"] + RF_FEATURES].copy().ffill().bfill()
test_data[TARGET] = rf.predict(test_data[RF_FEATURES])

save_submission(test_data[["id", TARGET]], SUBMISSION_NAME)
