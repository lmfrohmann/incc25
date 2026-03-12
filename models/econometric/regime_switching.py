"""Regime-Switching Threshold Regression — Econometric Approach."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from src.data import load_train, load_test, save_submission
from src.features import (
    add_all_enhanced_features_combined, clip_predictions,
    TOTAL_PROD_CAPACITY, TOTAL_PUMP_CAPACITY,
)

TARGET = "es_total_ps"
SUBMISSION_NAME = "regime_switching.csv"


# Core features used in each regime
CORE_FEATURES = [
    "es_residualdemand_f_d1",
    "es_residualdemand_f_d2",
    "solar_penetration_d1",
    "implied_elec_price_d1",
    "fr_surplus_d1",
    "iberian_residual_d1",
    "es_hydro_inflow_f_d1",
    "es_hydro_balance_f_d1",
    "es_hydro_ror_f_d1",
    "es_hydro_res_f_d1",
    "es_temp_f_d1",
    "es_precip_f_d1",
    "delta_es_residual",
    "delta_es_solar",
    "solar_dev_d1",
    "wind_dev_d1",
    "es_wind_f_d1",
    "es_solar_f_d1",
    "fr_demand_f_d1",
    "pt_demand_f_d1",
    "de_demand_f_d1",
    "es_gas_market_price_d1",
    "eua_price",
    # Rolling statistics (13)
    "es_residualdemand_f_d1_rm24", "es_residualdemand_f_d1_rstd24",
    "es_residualdemand_f_d1_rm168", "es_residualdemand_f_d1_mom24",
    "es_wind_f_d1_rm24", "es_wind_f_d1_rstd24", "es_wind_f_d1_mom24",
    "es_solar_f_d1_rm24", "es_solar_f_d1_rstd24", "es_solar_f_d1_mom24",
    "es_demand_f_d1_rm24", "es_demand_f_d1_rstd24", "es_demand_f_d1_rm168",
    # Extended lags (8)
    "es_solar_f_d1_lag24", "es_solar_f_d1_lag48",
    "es_wind_f_d1_lag24", "es_wind_f_d1_lag48",
    "es_demand_f_d1_lag24", "es_demand_f_d1_lag48",
    "es_residualdemand_f_d1_lag24", "es_residualdemand_f_d1_lag48",
]


def logistic(x, c, gamma):
    """Smooth transition function: logistic centered at c with steepness gamma."""
    return 1.0 / (1.0 + np.exp(-gamma * (x - c)))


def build_regime_features(df, core_features, resid_median=None, resid_std=None):
    """Construct regime-switching feature matrix."""
    available = [f for f in core_features if f in df.columns]
    X_base = df[available].fillna(0).copy()

    # Regime indicators

    # Method 1: Hour-based hard regimes
    hour = df["hour"].values
    pump_regime = ((hour >= 10) & (hour <= 15)).astype(float)
    produce_regime = ((hour >= 18) & (hour <= 21)).astype(float)

    # Method 2: Smooth transition based on residual demand
    resid = df["es_residualdemand_f_d1"].fillna(0).values
    if resid_median is None:
        resid_median = np.median(resid)
    if resid_std is None:
        resid_std = np.std(resid)
    # Probability of being in "produce" regime (high residual demand)
    gamma = 2.0 / resid_std  # steepness parameter
    produce_smooth = logistic(resid, resid_median + 0.3 * resid_std, gamma)
    pump_smooth = 1.0 - produce_smooth

    # Build regime-interacted features
    X_out = pd.DataFrame(index=df.index)

    # Fourier diurnal components (4 harmonics)
    for k in range(1, 5):
        X_out[f"hour_sin_{k}"] = np.sin(2 * np.pi * k * hour / 24)
        X_out[f"hour_cos_{k}"] = np.cos(2 * np.pi * k * hour / 24)

    # Calendar
    X_out["month_sin"] = df["month_sin"]
    X_out["month_cos"] = df["month_cos"]
    X_out["is_weekend"] = df["is_weekend"]
    X_out["doy_sin"] = df["doy_sin"]
    X_out["doy_cos"] = df["doy_cos"]

    # Capacity constraints
    if "avail_prod_capacity" in df.columns:
        X_out["avail_prod"] = df["avail_prod_capacity"]
        X_out["avail_pump"] = df["avail_pump_capacity"]
    else:
        X_out["avail_prod"] = TOTAL_PROD_CAPACITY
        X_out["avail_pump"] = TOTAL_PUMP_CAPACITY

    # Base features (regime-invariant)
    for col in available:
        X_out[f"base_{col}"] = X_base[col].values

    # Pump regime features (hour-based)
    for col in available:
        X_out[f"pump_{col}"] = X_base[col].values * pump_regime

    # Produce regime features (hour-based)
    for col in available:
        X_out[f"prod_{col}"] = X_base[col].values * produce_regime

    # Smooth-transition regime features
    for col in available[:10]:  # Top 10 features only (avoid explosion)
        X_out[f"smooth_pump_{col}"] = X_base[col].values * pump_smooth
        X_out[f"smooth_prod_{col}"] = X_base[col].values * produce_smooth

    # Regime indicators themselves
    X_out["pump_regime"] = pump_regime
    X_out["produce_regime"] = produce_regime
    X_out["produce_smooth"] = produce_smooth

    regime_stats = {"resid_median": resid_median, "resid_std": resid_std}
    return X_out, regime_stats


# Main
if __name__ == "__main__":

    np.random.seed(42)

    print("=" * 70)
    print("REGIME-SWITCHING THRESHOLD REGRESSION")
    print("=" * 70)

    # Load data
    print("\nLoading data...")
    train, test = add_all_enhanced_features_combined(load_train(), load_test())
    train_clean = train.dropna(subset=[TARGET]).reset_index(drop=True)

    X_full, train_regime_stats = build_regime_features(train_clean, CORE_FEATURES)
    X_test, _ = build_regime_features(test, CORE_FEATURES, **train_regime_stats)
    y_full = train_clean[TARGET]

    # Fill remaining NaNs
    X_full = X_full.fillna(0)
    X_test = X_test.fillna(0)

    feature_names = list(X_full.columns)
    print(f"Features: {len(feature_names)} ({len(CORE_FEATURES)} core × 3 regimes + controls)")

    # Temporal validation
    val_cutoff = train_clean["datetime_start"].max() - pd.Timedelta(days=90)
    train_mask = train_clean["datetime_start"] <= val_cutoff
    val_mask = train_clean["datetime_start"] > val_cutoff

    X_tr, y_tr = X_full[train_mask], y_full[train_mask]
    X_va, y_va = X_full[val_mask], y_full[val_mask]

    # Alpha search
    print("\n--- Ridge alpha search ---")
    best_rmse = float("inf")
    best_alpha = None

    for alpha in [1.0, 5.0, 10.0, 50.0, 100.0, 500.0, 1000.0]:
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_va_s = scaler.transform(X_va)

        model = Ridge(alpha=alpha)
        model.fit(X_tr_s, y_tr)
        preds = model.predict(X_va_s)
        preds_clipped = clip_predictions(preds, train_clean[val_mask])
        r = np.sqrt(np.mean((y_va.values - preds_clipped) ** 2))
        print(f"  alpha={alpha:>7.1f}: val RMSE = {r:.1f}")
        if r < best_rmse:
            best_rmse = r
            best_alpha = alpha

    print(f"  Best alpha: {best_alpha} (RMSE={best_rmse:.1f})")

    # Train final model on full data
    print(f"\n--- Training final model (alpha={best_alpha}) ---")
    scaler_final = StandardScaler()
    X_full_s = scaler_final.fit_transform(X_full)
    X_test_s = scaler_final.transform(X_test)

    final_model = Ridge(alpha=best_alpha)
    final_model.fit(X_full_s, y_full)

    # Per-hour bias estimation (5-fold expanding window)
    print("\n--- Per-hour bias estimation (5-fold expanding window) ---")
    n = len(X_full)
    min_train = int(n * 0.4)
    n_folds = 5
    fold_size = (n - min_train) // n_folds
    all_residuals = []
    all_hours = []

    for fold in range(n_folds):
        v_start = min_train + fold * fold_size
        v_end = min_train + (fold + 1) * fold_size if fold < n_folds - 1 else n
        tr_idx = list(range(v_start))
        va_idx = list(range(v_start, v_end))

        sc = StandardScaler()
        X_tr_f = sc.fit_transform(X_full.iloc[tr_idx])
        X_va_f = sc.transform(X_full.iloc[va_idx])

        m = Ridge(alpha=best_alpha)
        m.fit(X_tr_f, y_full.iloc[tr_idx])
        p = m.predict(X_va_f)
        residuals = y_full.iloc[va_idx].values - p
        rmse_fold = np.sqrt(np.mean(residuals ** 2))
        print(f"  Fold {fold+1}/{n_folds}: train={len(tr_idx)}, val={len(va_idx)}, RMSE={rmse_fold:.1f}")
        all_residuals.extend(residuals)
        all_hours.extend(train_clean["hour"].iloc[va_idx].values)

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
    avg_bias = float(np.mean(list(hourly_bias.values())))
    print(f"  Average per-hour bias: {avg_bias:+.1f}")

    # Predict
    print("\n--- Generating predictions ---")
    raw_preds = final_model.predict(X_test_s)
    corrected_preds = raw_preds.copy()
    for h in range(24):
        mask = test["hour"] == h
        corrected_preds[mask] += hourly_bias.get(h, 0.0)
    final_preds = clip_predictions(corrected_preds, test)

    print(f"  Raw mean:              {raw_preds.mean():.1f} MW")
    print(f"  + Per-hour bias (avg {avg_bias:+.1f}): {corrected_preds.mean():.1f} MW")
    print(f"  After clipping:        {final_preds.mean():.1f} MW")

    # Regime analysis
    print("\n--- Regime coefficient analysis (standardized β) ---")
    print("    Standardized coefficients: directly comparable across features.")
    print("    A 1-std increase in the feature changes the prediction by β_std MW.\n")
    coef_names = feature_names
    coefs_std = final_model.coef_  # standardized (X was scaled, y was not)

    print("  Feature                         | Base β_std | Pump β_std | Prod β_std")
    print("  " + "-" * 75)
    for feat in CORE_FEATURES[:10]:
        base_key = f"base_{feat}"
        pump_key = f"pump_{feat}"
        prod_key = f"prod_{feat}"
        vals = {}
        for key in [base_key, pump_key, prod_key]:
            if key in coef_names:
                idx = coef_names.index(key)
                vals[key] = coefs_std[idx]
            else:
                vals[key] = 0.0
        print(f"  {feat:<35s} | {vals[base_key]:>+10.2f} | {vals[pump_key]:>+10.2f} | {vals[prod_key]:>+10.2f}")

    # Bootstrap confidence intervals (optional)
    if "--bootstrap" in sys.argv:
        from src.metrics import block_bootstrap_coefs
        print("\n--- Bootstrap 95% confidence intervals (200 replications) ---")
        print("    Block bootstrap (block=24h) preserves temporal dependence.\n")

        X_arr = scaler_final.transform(X_full.values)
        y_arr = y_full.values

        def fit_fn(X_b, y_b):
            m = Ridge(alpha=best_alpha)
            m.fit(X_b, y_b)
            return m.coef_

        boot = block_bootstrap_coefs(X_arr, y_arr, fit_fn, n_bootstrap=200)
        print(f"  Valid replications: {boot['n_valid']}/200\n")

        # Show CIs for the top base features
        print(f"  {'Feature':<35s} {'β_std':>8s} {'95% CI':>20s} {'Sig?':>5s}")
        print("  " + "-" * 72)
        for feat in CORE_FEATURES[:10]:
            base_key = f"base_{feat}"
            if base_key in coef_names:
                idx = coef_names.index(base_key)
                beta = coefs_std[idx]
                lo = boot["ci_lower"][idx]
                hi = boot["ci_upper"][idx]
                sig = "*" if boot["significant"][idx] else ""
                print(f"  {base_key:<35s} {beta:>+8.2f} [{lo:>+8.2f}, {hi:>+8.2f}] {sig:>5s}")

    # Save
    test_out = test[["id"]].copy()
    test_out[TARGET] = final_preds
    save_submission(test_out, SUBMISSION_NAME)

    print(f"\nDone. Submission: output/{SUBMISSION_NAME}")
