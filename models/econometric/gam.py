"""GAM with Fourier Seasonal Structure."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np
import pandas as pd
from pygam import LinearGAM, s, f, l, te
from sklearn.preprocessing import StandardScaler

from src.data import load_train, load_test, save_submission
from src.features import (
    add_all_enhanced_features_combined, clip_predictions,
    TOTAL_PROD_CAPACITY, TOTAL_PUMP_CAPACITY,
)

TARGET = "es_total_ps"
SUBMISSION_NAME = "gam.csv"


def build_gam_features(df):
    """Build feature matrix for GAM. Order matters — matches GAM term spec."""
    X = pd.DataFrame(index=df.index)

    # 0: hour (cyclic spline)
    X["hour"] = df["hour"].values.astype(float)

    # 1: residual demand d1 (key economic driver)
    X["residual_demand"] = df["es_residualdemand_f_d1"].fillna(0)

    # 2: solar penetration
    X["solar_penetration"] = df["solar_penetration_d1"].fillna(0)

    # 3: implied electricity price
    X["implied_elec_price"] = df["implied_elec_price_d1"].fillna(0)

    # 4: France surplus
    X["fr_surplus"] = df["fr_surplus_d1"].fillna(0)

    # 5: hydro inflow
    X["hydro_inflow"] = df["es_hydro_inflow_f_d1"].fillna(0)

    # 6: hydro balance
    X["hydro_balance"] = df["es_hydro_balance_f_d1"].fillna(0)

    # 7: temperature
    X["temp"] = df["es_temp_f_d1"].fillna(0)

    # 8: ES wind
    X["es_wind"] = df["es_wind_f_d1"].fillna(0)

    # 9: ES solar
    X["es_solar"] = df["es_solar_f_d1"].fillna(0)

    # 10: iberian residual demand
    X["iberian_residual"] = df["iberian_residual_d1"].fillna(0)

    # 11: gas price
    X["gas_price"] = df["es_gas_market_price_d1"].fillna(0)

    # 12: EUA price
    X["eua_price"] = df["eua_price"].fillna(0)

    # 13: month (seasonal)
    X["month"] = df["month"].values.astype(float)

    # 14: is_weekend
    X["is_weekend"] = df["is_weekend"].values.astype(float)

    # 15: available production capacity
    if "avail_prod_capacity" in df.columns:
        X["avail_prod"] = df["avail_prod_capacity"]
    else:
        X["avail_prod"] = float(TOTAL_PROD_CAPACITY)

    # 16: available pump capacity
    if "avail_pump_capacity" in df.columns:
        X["avail_pump"] = df["avail_pump_capacity"]
    else:
        X["avail_pump"] = float(TOTAL_PUMP_CAPACITY)

    # 17: delta residual (d2 - d1)
    X["delta_residual"] = df["delta_es_residual"].fillna(0)

    # 18: ES hydro reservoir production
    X["hydro_res"] = df["es_hydro_res_f_d1"].fillna(0)

    # 19: precipitation
    X["precip"] = df["es_precip_f_d1"].fillna(0)

    # 20: RES ratio
    X["res_ratio"] = df["es_res_ratio_d1"].fillna(0)

    # 21: NTC net
    ntc_fr_es = df["fr_es_ntc_d1"].fillna(0) if "fr_es_ntc_d1" in df.columns else 0
    ntc_es_fr = df["es_fr_ntc_d1"].fillna(0) if "es_fr_ntc_d1" in df.columns else 0
    X["ntc_net"] = ntc_fr_es - ntc_es_fr

    # 22-27: Rolling statistics (key signals)
    X["resid_rm24"] = df["es_residualdemand_f_d1_rm24"].fillna(0) if "es_residualdemand_f_d1_rm24" in df.columns else 0
    X["resid_rstd24"] = df["es_residualdemand_f_d1_rstd24"].fillna(0) if "es_residualdemand_f_d1_rstd24" in df.columns else 0
    X["resid_rm168"] = df["es_residualdemand_f_d1_rm168"].fillna(0) if "es_residualdemand_f_d1_rm168" in df.columns else 0
    X["resid_mom24"] = df["es_residualdemand_f_d1_mom24"].fillna(0) if "es_residualdemand_f_d1_mom24" in df.columns else 0
    X["demand_rm24"] = df["es_demand_f_d1_rm24"].fillna(0) if "es_demand_f_d1_rm24" in df.columns else 0
    X["wind_rm24"] = df["es_wind_f_d1_rm24"].fillna(0) if "es_wind_f_d1_rm24" in df.columns else 0

    # 28-31: Extended lags
    X["resid_lag24"] = df["es_residualdemand_f_d1_lag24"].fillna(0) if "es_residualdemand_f_d1_lag24" in df.columns else 0
    X["resid_lag48"] = df["es_residualdemand_f_d1_lag48"].fillna(0) if "es_residualdemand_f_d1_lag48" in df.columns else 0
    X["demand_lag24"] = df["es_demand_f_d1_lag24"].fillna(0) if "es_demand_f_d1_lag24" in df.columns else 0
    X["demand_lag48"] = df["es_demand_f_d1_lag48"].fillna(0) if "es_demand_f_d1_lag48" in df.columns else 0

    return X


def build_gam_terms():
    """Build GAM term specification."""
    return (
        s(0, n_splines=25, spline_order=3)   # hour
        + s(1, n_splines=20)           # residual demand
        + s(2, n_splines=15)           # solar penetration
        + s(3, n_splines=15)           # implied price
        + s(4, n_splines=15)           # France surplus
        + s(5, n_splines=12)           # hydro inflow
        + s(6, n_splines=12)           # hydro balance
        + s(7, n_splines=12)           # temperature
        + s(8, n_splines=15)           # ES wind
        + s(9, n_splines=15)           # ES solar
        + s(10, n_splines=15)          # iberian residual
        + s(11, n_splines=12)          # gas price
        + s(12, n_splines=10)          # EUA price
        + s(13, n_splines=12)          # month
        + l(14)                        # is_weekend (binary → linear)
        + l(15)                        # avail_prod (near-constant → linear)
        + l(16)                        # avail_pump (near-constant → linear)
        + s(17, n_splines=12)          # delta residual
        + s(18, n_splines=12)          # hydro reservoir
        + s(19, n_splines=10)          # precipitation
        + s(20, n_splines=12)          # RES ratio
        + s(21, n_splines=10)          # NTC net
        # Rolling statistics
        + s(22, n_splines=12)          # resid_rm24
        + s(23, n_splines=10)          # resid_rstd24
        + s(24, n_splines=12)          # resid_rm168
        + s(25, n_splines=10)          # resid_mom24
        + s(26, n_splines=12)          # demand_rm24
        + s(27, n_splines=10)          # wind_rm24
        # Extended lags
        + s(28, n_splines=12)          # resid_lag24
        + s(29, n_splines=10)          # resid_lag48
        + s(30, n_splines=12)          # demand_lag24
        + s(31, n_splines=10)          # demand_lag48
        # Tensor product: hour × residual demand (time-varying dispatch)
        + te(0, 1, n_splines=[10, 10])
    )


# Main
if __name__ == "__main__":

    np.random.seed(42)

    print("=" * 70)
    print("GAM WITH FOURIER SEASONAL STRUCTURE")
    print("=" * 70)

    # Load data
    print("\nLoading data...")
    train, test = add_all_enhanced_features_combined(load_train(), load_test())
    train_clean = train.dropna(subset=[TARGET]).reset_index(drop=True)

    X_full = build_gam_features(train_clean)
    X_test = build_gam_features(test)
    y_full = train_clean[TARGET].values

    feature_names = list(X_full.columns)
    print(f"Features: {len(feature_names)}")

    # Convert to numpy
    X_full_np = X_full.values
    X_test_np = X_test.values

    # Temporal validation
    val_cutoff = train_clean["datetime_start"].max() - pd.Timedelta(days=90)
    train_mask = (train_clean["datetime_start"] <= val_cutoff).values
    val_mask = (train_clean["datetime_start"] > val_cutoff).values

    X_tr, y_tr = X_full_np[train_mask], y_full[train_mask]
    X_va, y_va = X_full_np[val_mask], y_full[val_mask]

    # Lambda search
    print("\n--- Lambda (smoothing penalty) search ---")
    best_rmse = float("inf")
    best_lam = None
    terms = build_gam_terms()

    for lam_exp in [-1, 0, 1, 2, 3, 4]:
        lam = 10.0 ** lam_exp
        gam = LinearGAM(terms, lam=lam)
        gam.fit(X_tr, y_tr)
        preds = gam.predict(X_va)
        preds_clipped = clip_predictions(preds, train_clean[val_mask])
        r = np.sqrt(np.mean((y_va - preds_clipped) ** 2))
        print(f"  lam=1e{lam_exp}: val RMSE = {r:.1f}")
        if r < best_rmse:
            best_rmse = r
            best_lam = lam

    print(f"  Best lambda: {best_lam} (RMSE={best_rmse:.1f})")

    # Also try gridsearch
    print("\n--- Trying pyGAM gridsearch (automatic lambda tuning) ---")
    gam_gs = LinearGAM(terms)
    gam_gs.gridsearch(X_tr, y_tr, progress=False)
    preds_gs = gam_gs.predict(X_va)
    preds_gs_clipped = clip_predictions(preds_gs, train_clean[val_mask])
    rmse_gs = np.sqrt(np.mean((y_va - preds_gs_clipped) ** 2))
    print(f"  Gridsearch val RMSE: {rmse_gs:.1f}")

    # Use whichever is better
    if rmse_gs < best_rmse:
        print("  -> Using gridsearch model")
        best_rmse = rmse_gs
        use_gridsearch = True
    else:
        print(f"  -> Using fixed lambda={best_lam}")
        use_gridsearch = False

    # Train final model on full data
    print(f"\n--- Training final GAM on full data ---")
    if use_gridsearch:
        final_gam = LinearGAM(terms)
        final_gam.gridsearch(X_full_np, y_full, progress=False)
    else:
        final_gam = LinearGAM(terms, lam=best_lam)
        final_gam.fit(X_full_np, y_full)

    print(f"  GCV score: {final_gam.statistics_['GCV']:.1f}")
    print(f"  Pseudo R-squared: {final_gam.statistics_['pseudo_r2']['explained_deviance']:.4f}")

    # Per-hour bias estimation (5-fold expanding window)
    print("\n--- Per-hour bias estimation (5-fold expanding window) ---")
    n = len(X_full_np)
    min_train_n = int(n * 0.4)
    n_folds = 5
    fold_size = (n - min_train_n) // n_folds
    all_residuals = []
    all_hours = []

    for fold in range(n_folds):
        v_start = min_train_n + fold * fold_size
        v_end = min_train_n + (fold + 1) * fold_size if fold < n_folds - 1 else n
        tr_idx = list(range(v_start))
        va_idx = list(range(v_start, v_end))

        if use_gridsearch:
            m = LinearGAM(terms)
            m.gridsearch(X_full_np[tr_idx], y_full[tr_idx], progress=False)
        else:
            m = LinearGAM(terms, lam=best_lam)
            m.fit(X_full_np[tr_idx], y_full[tr_idx])

        p = m.predict(X_full_np[va_idx])
        residuals = y_full[va_idx] - p
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
    raw_preds = final_gam.predict(X_test_np)
    corrected_preds = raw_preds.copy()
    for h in range(24):
        mask = test["hour"] == h
        corrected_preds[mask] += hourly_bias.get(h, 0.0)
    final_preds = clip_predictions(corrected_preds, test)

    print(f"  Raw mean:              {raw_preds.mean():.1f} MW")
    print(f"  + Per-hour bias (avg {avg_bias:+.1f}): {corrected_preds.mean():.1f} MW")
    print(f"  After clipping:        {final_preds.mean():.1f} MW")

    # Partial dependence analysis
    # Iterate only over the first len(feature_names) terms; the tensor
    # product term at index 32 (te(hour, residual_demand)) is excluded
    # because it does not map to a single named feature.
    print("\n--- Partial dependence (feature effects) ---")
    for i in range(len(feature_names)):
        fname = feature_names[i]
        # Compute the range of the partial effect
        XX = final_gam.generate_X_grid(term=i, n=100)
        pdep = final_gam.partial_dependence(term=i, X=XX)
        effect_range = pdep.max() - pdep.min()
        print(f"  {fname:<25s}: effect range = {effect_range:>8.1f} MW")

    # Save
    test_out = test[["id"]].copy()
    test_out[TARGET] = final_preds
    save_submission(test_out, SUBMISSION_NAME)

    print(f"\nDone. Submission: output/{SUBMISSION_NAME}")
