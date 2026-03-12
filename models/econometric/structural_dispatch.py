"""Structural Dispatch Model — Econometric Merit-Order Approach."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, HuberRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from src.data import load_train, load_test, save_submission
from src.features import (
    add_all_enhanced_features_combined, clip_predictions,
    TOTAL_PROD_CAPACITY, TOTAL_PUMP_CAPACITY,
)

TARGET = "es_total_ps"
SUBMISSION_NAME = "structural_dispatch.csv"


# 1. Economic feature construction

def build_dispatch_features(df):
    """Construct features motivated by the dispatch optimization problem."""
    X = pd.DataFrame(index=df.index)

    # A. Core economic drivers

    # Residual demand: the key merit-order signal
    # When residual demand is high → expensive plants set price → produce
    # When residual demand is low → cheap power → pump
    X["residual_demand_d1"] = df["es_residualdemand_f_d1"]
    X["residual_demand_d2"] = df["es_residualdemand_f_d2"]

    # Quadratic residual demand (merit order is convex)
    X["residual_demand_sq"] = df["es_residualdemand_f_d1"] ** 2

    # Solar penetration: when solar floods the market, prices crash → pump
    X["solar_penetration"] = df["solar_penetration_d1"]

    # RES ratio: total renewables / demand
    X["res_ratio"] = df["es_res_ratio_d1"]

    # Implied electricity price from gas (marginal plant cost)
    X["implied_elec_price"] = df["implied_elec_price_d1"]

    # B. Cross-border arbitrage signals

    # France surplus drives exports to Spain → lowers prices → pump
    X["fr_surplus"] = df["fr_surplus_d1"]

    # Iberian residual demand (ES + PT integrated market)
    X["iberian_residual"] = df["iberian_residual_d1"]

    # Net transfer capacity (cross-border flow constraints)
    X["fr_es_ntc"] = df["fr_es_ntc_d1"].fillna(0) if "fr_es_ntc_d1" in df.columns else 0.0
    X["es_fr_ntc"] = df["es_fr_ntc_d1"].fillna(0) if "es_fr_ntc_d1" in df.columns else 0.0
    # Net flow direction signal: positive = France can export to Spain
    X["ntc_net"] = X["fr_es_ntc"] - X["es_fr_ntc"]

    # C. Hydro state variables

    # Hydro inflow and balance affect reservoir management strategy
    X["hydro_inflow"] = df["es_hydro_inflow_f_d1"]
    X["hydro_balance"] = df["es_hydro_balance_f_d1"]

    # Run-of-river: baseload hydro that reduces need for pumped storage
    X["hydro_ror_es"] = df["es_hydro_ror_f_d1"]
    X["hydro_res_es"] = df["es_hydro_res_f_d1"]

    # D. Diurnal Fourier components
    # The pump/produce cycle follows a deterministic daily pattern
    # modulated by fundamentals. Fourier terms capture this.
    hour = df["hour"]
    for k in range(1, 5):  # 4 harmonics (captures ~99% of diurnal variance)
        X[f"hour_sin_{k}"] = np.sin(2 * np.pi * k * hour / 24)
        X[f"hour_cos_{k}"] = np.cos(2 * np.pi * k * hour / 24)

    # E. Fourier × fundamental interactions
    # The *amplitude* of the diurnal cycle depends on fundamentals:
    # more solar → bigger midday dip (more pumping)
    # higher residual demand → bigger evening peak (more producing)
    X["sin1_x_solar"] = X["hour_sin_1"] * X["solar_penetration"]
    X["cos1_x_solar"] = X["hour_cos_1"] * X["solar_penetration"]
    X["sin1_x_residual"] = X["hour_sin_1"] * X["residual_demand_d1"]
    X["cos1_x_residual"] = X["hour_cos_1"] * X["residual_demand_d1"]
    X["sin1_x_price"] = X["hour_sin_1"] * X["implied_elec_price"]
    X["cos1_x_price"] = X["hour_cos_1"] * X["implied_elec_price"]

    # F. Capacity constraints
    if "avail_prod_capacity" in df.columns:
        X["avail_prod"] = df["avail_prod_capacity"]
        X["avail_pump"] = df["avail_pump_capacity"]
        X["capacity_ratio"] = df["avail_prod_capacity"] / TOTAL_PROD_CAPACITY
    else:
        X["avail_prod"] = TOTAL_PROD_CAPACITY
        X["avail_pump"] = TOTAL_PUMP_CAPACITY
        X["capacity_ratio"] = 1.0

    # G. Seasonal / calendar
    X["month_sin"] = df["month_sin"]
    X["month_cos"] = df["month_cos"]
    X["is_weekend"] = df["is_weekend"]
    X["doy_sin"] = df["doy_sin"]
    X["doy_cos"] = df["doy_cos"]

    # H. Weather
    X["temp"] = df["es_temp_f_d1"]
    X["precip"] = df["es_precip_f_d1"]
    X["wind_speed"] = df["es_wind_speed_f_d1"]

    # I. D2−D1 deltas (forward-looking information)
    X["delta_residual"] = df["delta_es_residual"]
    X["delta_solar"] = df["delta_es_solar"]
    X["delta_demand"] = df["delta_es_demand"]

    # J. Deviation from normal (forecast surprise)
    X["solar_dev"] = df["solar_dev_d1"]
    X["wind_dev"] = df["wind_dev_d1"]

    # K. Rolling statistics (13 features)
    rolling_cols = [
        "es_residualdemand_f_d1_rm24", "es_residualdemand_f_d1_rstd24",
        "es_residualdemand_f_d1_rm168", "es_residualdemand_f_d1_mom24",
        "es_wind_f_d1_rm24", "es_wind_f_d1_rstd24", "es_wind_f_d1_mom24",
        "es_solar_f_d1_rm24", "es_solar_f_d1_rstd24", "es_solar_f_d1_mom24",
        "es_demand_f_d1_rm24", "es_demand_f_d1_rstd24", "es_demand_f_d1_rm168",
    ]
    for col in rolling_cols:
        if col in df.columns:
            X[col] = df[col].fillna(0)

    # L. Extended lags (8 features)
    lag_cols = [
        "es_solar_f_d1_lag24", "es_solar_f_d1_lag48",
        "es_wind_f_d1_lag24", "es_wind_f_d1_lag48",
        "es_demand_f_d1_lag24", "es_demand_f_d1_lag48",
        "es_residualdemand_f_d1_lag24", "es_residualdemand_f_d1_lag48",
    ]
    for col in lag_cols:
        if col in df.columns:
            X[col] = df[col].fillna(0)

    # M. Fourier × rolling interactions
    if "es_residualdemand_f_d1_rm24" in X.columns:
        X["sin1_x_rolling_resid"] = X["hour_sin_1"] * X["es_residualdemand_f_d1_rm24"]
        X["cos1_x_rolling_resid"] = X["hour_cos_1"] * X["es_residualdemand_f_d1_rm24"]

    return X


# 2. Model: Two-stage structural estimation

def build_structural_model(alpha_ridge=10.0):
    """Stage 1 + 2 combined via a scaled Ridge regression."""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("ridge", Ridge(alpha=alpha_ridge)),
    ])


def build_robust_model(epsilon=1.5):
    """Huber regression: M-estimator that is robust to outliers."""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("huber", HuberRegressor(epsilon=epsilon, max_iter=500)),
    ])


# 3. Main
if __name__ == "__main__":

    np.random.seed(42)

    print("=" * 70)
    print("STRUCTURAL DISPATCH MODEL")
    print("=" * 70)

    # Load and engineer features
    print("\nLoading data and engineering features...")
    train, test = add_all_enhanced_features_combined(load_train(), load_test())
    train_clean = train.dropna(subset=[TARGET]).reset_index(drop=True)

    X_train_full = build_dispatch_features(train_clean)
    X_test = build_dispatch_features(test)
    y_full = train_clean[TARGET]

    # Handle NaNs (fill with 0 — NTCs, hydro inflow)
    X_train_full = X_train_full.fillna(0)
    X_test = X_test.fillna(0)

    feature_names = list(X_train_full.columns)
    print(f"Features: {len(feature_names)}")

    # Temporal validation
    val_cutoff = train_clean["datetime_start"].max() - pd.Timedelta(days=90)
    train_mask = train_clean["datetime_start"] <= val_cutoff
    val_mask = train_clean["datetime_start"] > val_cutoff

    X_tr, y_tr = X_train_full[train_mask], y_full[train_mask]
    X_va, y_va = X_train_full[val_mask], y_full[val_mask]

    # Test multiple alpha values
    print("\n--- Ridge alpha search ---")
    best_rmse = float("inf")
    best_alpha = None

    for alpha in [0.1, 1.0, 5.0, 10.0, 50.0, 100.0, 500.0]:
        model = build_structural_model(alpha_ridge=alpha)
        model.fit(X_tr, y_tr)
        preds = model.predict(X_va)
        preds_clipped = clip_predictions(preds, train_clean[val_mask])
        r = np.sqrt(np.mean((y_va.values - preds_clipped) ** 2))
        print(f"  alpha={alpha:>6.1f}: val RMSE = {r:.1f}")
        if r < best_rmse:
            best_rmse = r
            best_alpha = alpha

    print(f"  Best alpha: {best_alpha} (RMSE={best_rmse:.1f})")

    # Huber regression (robust to outliers)
    print("\n--- Huber robust regression ---")
    for eps in [1.2, 1.35, 1.5, 2.0]:
        model = build_robust_model(epsilon=eps)
        model.fit(X_tr, y_tr)
        preds = model.predict(X_va)
        preds_clipped = clip_predictions(preds, train_clean[val_mask])
        r = np.sqrt(np.mean((y_va.values - preds_clipped) ** 2))
        print(f"  epsilon={eps:.2f}: val RMSE = {r:.1f}")

    # Final model: Ridge with best alpha, trained on full data
    print(f"\n--- Training final Ridge (alpha={best_alpha}) on full data ---")
    final_model = build_structural_model(alpha_ridge=best_alpha)
    final_model.fit(X_train_full, y_full)

    # Per-hour bias correction (5-fold expanding window)
    print("\n--- Per-hour bias estimation (5-fold expanding window) ---")
    n = len(X_train_full)
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

        m = build_structural_model(alpha_ridge=best_alpha)
        m.fit(X_train_full.iloc[tr_idx], y_full.iloc[tr_idx])
        p = m.predict(X_train_full.iloc[va_idx])
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

    # Predict test
    print("\n--- Generating predictions ---")
    raw_preds = final_model.predict(X_test)
    corrected_preds = raw_preds.copy()
    for h in range(24):
        mask = test["hour"] == h
        corrected_preds[mask] += hourly_bias.get(h, 0.0)
    final_preds = clip_predictions(corrected_preds, test)

    print(f"  Raw mean:              {raw_preds.mean():.1f} MW")
    print(f"  + Per-hour bias (avg {avg_bias:+.1f}): {corrected_preds.mean():.1f} MW")
    print(f"  After clipping:        {final_preds.mean():.1f} MW")

    # Coefficient analysis (interpretability!)
    print("\n--- Top 20 coefficients by standardized magnitude ---")
    print("    β_std: effect in MW of a 1-std increase in the feature.")
    print("    Directly comparable across features (scale-free).\n")
    ridge = final_model.named_steps["ridge"]
    coef_df = pd.DataFrame({
        "feature": feature_names,
        "beta_std": ridge.coef_,
    })
    coef_df["abs_beta"] = np.abs(coef_df["beta_std"])
    coef_df = coef_df.sort_values("abs_beta", ascending=False)
    for _, row in coef_df.head(20).iterrows():
        sign = "+" if row["beta_std"] > 0 else "-"
        print(f"  {sign} {row['feature']:<30s} β_std={row['beta_std']:>+8.1f} MW")

    # Bootstrap confidence intervals (optional)
    if "--bootstrap" in sys.argv:
        from src.metrics import block_bootstrap_coefs
        print("\n--- Bootstrap 95% confidence intervals (200 replications) ---")
        print("    Block bootstrap (block=24h) preserves temporal dependence.\n")

        X_arr = final_model.named_steps["scaler"].transform(X_train_full.values)
        y_arr = y_full.values

        def fit_fn(X_b, y_b):
            from sklearn.linear_model import Ridge as R
            m = R(alpha=best_alpha)
            m.fit(X_b, y_b)
            return m.coef_

        boot = block_bootstrap_coefs(X_arr, y_arr, fit_fn, n_bootstrap=200)
        print(f"  Valid replications: {boot['n_valid']}/200\n")
        print(f"  {'Feature':<30s} {'β_std':>8s} {'95% CI':>20s} {'Sig?':>5s}")
        print("  " + "-" * 67)
        for i, (_, row) in enumerate(coef_df.head(20).iterrows()):
            feat_idx = feature_names.index(row["feature"])
            lo = boot["ci_lower"][feat_idx]
            hi = boot["ci_upper"][feat_idx]
            sig = "*" if boot["significant"][feat_idx] else ""
            print(f"  {row['feature']:<30s} {row['beta_std']:>+8.1f} "
                  f"[{lo:>+8.1f}, {hi:>+8.1f}] {sig:>5s}")

    # Save
    test_out = test[["id"]].copy()
    test_out[TARGET] = final_preds
    save_submission(test_out, SUBMISSION_NAME)

    print(f"\nDone. Submission: output/{SUBMISSION_NAME}")
