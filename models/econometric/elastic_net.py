"""Elastic Net with Econometric Feature Engineering."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNetCV, LassoCV, RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit

from src.data import load_train, load_test, save_submission
from src.features import add_all_enhanced_features_combined, clip_predictions

TARGET = "es_total_ps"
SUBMISSION_NAME = "elastic_net.csv"


def build_econometric_features(df):
    """Construct a rich econometric feature matrix (~200 features)."""
    X = pd.DataFrame(index=df.index)
    hour = df["hour"].values

    # =====================================================================
    # A. FOURIER DIURNAL COMPONENTS (4 harmonics)
    # =====================================================================
    for k in range(1, 5):
        X[f"sin_{k}"] = np.sin(2 * np.pi * k * hour / 24)
        X[f"cos_{k}"] = np.cos(2 * np.pi * k * hour / 24)

    # =====================================================================
    # B. SEASONAL COMPONENTS
    # =====================================================================
    X["month_sin"] = df["month_sin"]
    X["month_cos"] = df["month_cos"]
    X["doy_sin"] = df["doy_sin"]
    X["doy_cos"] = df["doy_cos"]
    X["is_weekend"] = df["is_weekend"]

    # =====================================================================
    # C. FUNDAMENTAL ECONOMIC DRIVERS (levels)
    # =====================================================================
    fundamentals_d1 = [
        "es_demand_f_d1", "fr_demand_f_d1", "de_demand_f_d1", "pt_demand_f_d1",
        "es_wind_f_d1", "fr_wind_f_d1", "de_wind_f_d1", "pt_wind_f_d1",
        "es_solar_f_d1", "fr_solar_f_d1", "de_solar_f_d1", "pt_solar_f_d1",
        "es_hydro_ror_f_d1", "fr_hydro_ror_f_d1", "pt_hydro_ror_f_d1",
        "es_hydro_res_f_d1", "fr_hydro_res_f_d1",
        "es_hydro_inflow_f_d1", "es_hydro_balance_f_d1",
        "es_temp_f_d1", "es_precip_f_d1", "es_wind_speed_f_d1",
        "es_gas_market_price_d1", "eua_price",
    ]
    for col in fundamentals_d1:
        if col in df.columns:
            X[col] = df[col].fillna(0)

    # d2 features
    fundamentals_d2 = [
        "es_demand_f_d2", "fr_demand_f_d2", "de_demand_f_d2", "pt_demand_f_d2",
        "es_wind_f_d2", "fr_wind_f_d2", "de_wind_f_d2", "pt_wind_f_d2",
        "es_solar_f_d2", "fr_solar_f_d2", "de_solar_f_d2", "pt_solar_f_d2",
        "es_hydro_ror_f_d2", "fr_hydro_ror_f_d2", "pt_hydro_ror_f_d2",
        "es_hydro_res_f_d2", "fr_hydro_res_f_d2",
        "es_hydro_inflow_f_d2", "es_hydro_balance_f_d2",
        "es_temp_f_d2", "es_precip_f_d2", "es_wind_speed_f_d2",
        "es_gas_market_price_d2",
    ]
    for col in fundamentals_d2:
        if col in df.columns:
            X[col] = df[col].fillna(0)

    # NTC
    for col in ["fr_es_ntc_d1", "es_fr_ntc_d1"]:
        if col in df.columns:
            X[col] = df[col].fillna(0)

    # =====================================================================
    # D. ENGINEERED ECONOMIC VARIABLES
    # =====================================================================

    # Residual demand (key merit order signal)
    X["residual_demand_d1"] = df["es_residualdemand_f_d1"]
    X["residual_demand_d2"] = df["es_residualdemand_f_d2"]

    # Iberian aggregate
    X["iberian_residual"] = df["iberian_residual_d1"]

    # Implied electricity price
    X["implied_elec_price"] = df["implied_elec_price_d1"]

    # France surplus
    X["fr_surplus"] = df["fr_surplus_d1"]

    # Solar/wind penetration
    X["solar_penetration"] = df["solar_penetration_d1"]
    X["res_ratio"] = df["es_res_ratio_d1"]

    # Deltas (d2 − d1)
    X["delta_residual"] = df["delta_es_residual"]
    X["delta_solar"] = df["delta_es_solar"]
    X["delta_demand"] = df["delta_es_demand"]
    X["delta_wind"] = df["delta_es_wind"]

    # Deviations from normal
    X["solar_dev"] = df["solar_dev_d1"]
    X["wind_dev"] = df["wind_dev_d1"]

    # =====================================================================
    # E. POLYNOMIAL TERMS (nonlinear price-quantity relationships)
    # =====================================================================
    X["residual_demand_sq"] = X["residual_demand_d1"] ** 2
    X["residual_demand_cu"] = X["residual_demand_d1"] ** 3
    X["solar_pen_sq"] = X["solar_penetration"] ** 2
    X["implied_price_sq"] = X["implied_elec_price"] ** 2
    X["temp_sq"] = df["es_temp_f_d1"].fillna(0) ** 2

    # =====================================================================
    # F. CAPACITY CONSTRAINTS
    # =====================================================================
    if "avail_prod_capacity" in df.columns:
        X["avail_prod"] = df["avail_prod_capacity"]
        X["avail_pump"] = df["avail_pump_capacity"]
        X["capacity_asym"] = df["capacity_asymmetry"]
        X["total_prod_unavail"] = df["total_prod_unavail_mw"]
        X["total_cons_unavail"] = df["total_cons_unavail_mw"]

    # =====================================================================
    # G. HOUR × FUNDAMENTAL INTERACTIONS
    # =====================================================================
    # Key insight: the effect of solar on dispatch DEPENDS on the hour
    key_interactions = [
        "residual_demand_d1", "solar_penetration", "implied_elec_price",
        "fr_surplus", "iberian_residual",
    ]
    for feat in key_interactions:
        if feat in X.columns:
            # Fourier × fundamental (time-varying coefficients)
            X[f"sin1_x_{feat}"] = X["sin_1"] * X[feat]
            X[f"cos1_x_{feat}"] = X["cos_1"] * X[feat]
            X[f"sin2_x_{feat}"] = X["sin_2"] * X[feat]
            X[f"cos2_x_{feat}"] = X["cos_2"] * X[feat]

    # =====================================================================
    # H. CROSS-COUNTRY SPREADS (arbitrage signals)
    # =====================================================================
    if "es_demand_f_d1" in df.columns and "fr_demand_f_d1" in df.columns:
        # Demand spread: if France has more demand → less export to Spain
        X["fr_es_demand_spread"] = df["fr_demand_f_d1"] - df["es_demand_f_d1"]
        # Wind spread: if Germany has more wind → cheaper power flows south
        X["de_es_wind_spread"] = df["de_wind_f_d1"].fillna(0) - df["es_wind_f_d1"]

    # =====================================================================
    # I. LOG TRANSFORMS (diminishing returns)
    # =====================================================================
    for col in ["es_demand_f_d1", "fr_demand_f_d1", "de_demand_f_d1", "pt_demand_f_d1"]:
        if col in df.columns:
            X[f"log_{col}"] = np.log(df[col].clip(lower=1))

    X["log_eua"] = np.log(df["eua_price"].clip(lower=0.01))
    X["log_gas"] = np.log(df["es_gas_market_price_d1"].clip(lower=0.01).fillna(1))

    # =====================================================================
    # J. ROLLING STATISTICS (13 features)
    # =====================================================================
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

    # =====================================================================
    # K. EXTENDED LAGS (8 features: 24h and 48h lags)
    # =====================================================================
    lag_cols = [
        "es_solar_f_d1_lag24", "es_solar_f_d1_lag48",
        "es_wind_f_d1_lag24", "es_wind_f_d1_lag48",
        "es_demand_f_d1_lag24", "es_demand_f_d1_lag48",
        "es_residualdemand_f_d1_lag24", "es_residualdemand_f_d1_lag48",
    ]
    for col in lag_cols:
        if col in df.columns:
            X[col] = df[col].fillna(0)

    # =====================================================================
    # L. FOURIER × ROLLING INTERACTIONS (time-varying rolling effects)
    # =====================================================================
    rolling_interact = ["es_residualdemand_f_d1_rm24", "es_demand_f_d1_rm24"]
    for feat in rolling_interact:
        if feat in X.columns:
            X[f"sin1_x_{feat}"] = X["sin_1"] * X[feat]
            X[f"cos1_x_{feat}"] = X["cos_1"] * X[feat]

    return X.fillna(0)


# Main
if __name__ == "__main__":

    np.random.seed(42)

    print("=" * 70)
    print("ELASTIC NET ECONOMETRIC MODEL")
    print("=" * 70)

    # Load data
    print("\nLoading data and engineering features...")
    train, test = add_all_enhanced_features_combined(load_train(), load_test())
    train_clean = train.dropna(subset=[TARGET]).reset_index(drop=True)

    X_full = build_econometric_features(train_clean)
    X_test = build_econometric_features(test)
    y_full = train_clean[TARGET]

    feature_names = list(X_full.columns)
    print(f"Features: {len(feature_names)}")

    # Scale
    scaler = StandardScaler()
    X_full_s = scaler.fit_transform(X_full)
    X_test_s = scaler.transform(X_test)

    # Temporal validation
    val_cutoff = train_clean["datetime_start"].max() - pd.Timedelta(days=90)
    train_mask = train_clean["datetime_start"] <= val_cutoff
    val_mask = train_clean["datetime_start"] > val_cutoff

    X_tr_s = X_full_s[train_mask.values]
    X_va_s = X_full_s[val_mask.values]
    y_tr = y_full[train_mask]
    y_va = y_full[val_mask]

    # Compare: ElasticNet, LASSO, Ridge
    print("\n--- Model comparison (temporal validation) ---")

    # TimeSeriesSplit for CV within the training portion
    tscv = TimeSeriesSplit(n_splits=5)

    # Elastic Net CV
    print("\n[1] Elastic Net (α and l1_ratio via CV)...")
    enet = ElasticNetCV(
        l1_ratio=[0.1, 0.3, 0.5, 0.7, 0.9],
        alphas=np.logspace(-3, 2, 30),
        cv=tscv,
        max_iter=10000,
        n_jobs=-1,
        random_state=42,
    )
    enet.fit(X_tr_s, y_tr)
    enet_preds = enet.predict(X_va_s)
    enet_clipped = clip_predictions(enet_preds, train_clean[val_mask])
    enet_rmse = np.sqrt(np.mean((y_va.values - enet_clipped) ** 2))
    n_nonzero = np.sum(enet.coef_ != 0)
    print(f"  Best alpha={enet.alpha_:.4f}, l1_ratio={enet.l1_ratio_:.2f}")
    print(f"  Non-zero coefficients: {n_nonzero}/{len(feature_names)}")
    print(f"  Val RMSE: {enet_rmse:.1f}")

    # LASSO CV
    print("\n[2] LASSO (pure L1)...")
    lasso = LassoCV(
        alphas=np.logspace(-3, 2, 30),
        cv=tscv,
        max_iter=10000,
        n_jobs=-1,
        random_state=42,
    )
    lasso.fit(X_tr_s, y_tr)
    lasso_preds = lasso.predict(X_va_s)
    lasso_clipped = clip_predictions(lasso_preds, train_clean[val_mask])
    lasso_rmse = np.sqrt(np.mean((y_va.values - lasso_clipped) ** 2))
    lasso_nonzero = np.sum(lasso.coef_ != 0)
    print(f"  Best alpha={lasso.alpha_:.4f}")
    print(f"  Non-zero coefficients: {lasso_nonzero}/{len(feature_names)}")
    print(f"  Val RMSE: {lasso_rmse:.1f}")

    # Ridge CV
    print("\n[3] Ridge (pure L2)...")
    ridge = RidgeCV(alphas=np.logspace(-2, 4, 30), cv=tscv)
    ridge.fit(X_tr_s, y_tr)
    ridge_preds = ridge.predict(X_va_s)
    ridge_clipped = clip_predictions(ridge_preds, train_clean[val_mask])
    ridge_rmse = np.sqrt(np.mean((y_va.values - ridge_clipped) ** 2))
    print(f"  Best alpha={ridge.alpha_:.4f}")
    print(f"  Val RMSE: {ridge_rmse:.1f}")

    # Pick best
    best_model_name, best_model_rmse = min(
        [("ElasticNet", enet_rmse), ("LASSO", lasso_rmse), ("Ridge", ridge_rmse)],
        key=lambda x: x[1],
    )
    print(f"\n  Best model: {best_model_name} (RMSE={best_model_rmse:.1f})")

    # Retrain best on full data
    print(f"\n--- Retraining {best_model_name} on full data ---")
    if best_model_name == "ElasticNet":
        from sklearn.linear_model import ElasticNet as EN
        final = EN(alpha=enet.alpha_, l1_ratio=enet.l1_ratio_, max_iter=10000)
    elif best_model_name == "LASSO":
        from sklearn.linear_model import Lasso
        final = Lasso(alpha=lasso.alpha_, max_iter=10000)
    else:
        from sklearn.linear_model import Ridge as R
        final = R(alpha=ridge.alpha_)

    final.fit(X_full_s, y_full)

    # Per-hour bias correction (5-fold expanding window)
    print("\n--- Per-hour bias estimation (5-fold expanding window) ---")
    n = len(X_full_s)
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

        sc = StandardScaler()
        Xtr = sc.fit_transform(X_full.iloc[tr_idx])
        Xva = sc.transform(X_full.iloc[va_idx])

        if best_model_name == "ElasticNet":
            m = EN(alpha=enet.alpha_, l1_ratio=enet.l1_ratio_, max_iter=10000)
        elif best_model_name == "LASSO":
            m = Lasso(alpha=lasso.alpha_, max_iter=10000)
        else:
            m = R(alpha=ridge.alpha_)

        m.fit(Xtr, y_full.iloc[tr_idx])
        p = m.predict(Xva)
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
    raw_preds = final.predict(X_test_s)
    corrected_preds = raw_preds.copy()
    for h in range(24):
        mask = test["hour"] == h
        corrected_preds[mask] += hourly_bias.get(h, 0.0)
    final_preds = clip_predictions(corrected_preds, test)

    print(f"  Raw mean:              {raw_preds.mean():.1f} MW")
    print(f"  + Per-hour bias (avg {avg_bias:+.1f}): {corrected_preds.mean():.1f} MW")
    print(f"  After clipping:        {final_preds.mean():.1f} MW")

    # Feature importance (non-zero coefficients)
    print(f"\n--- Selected features ({best_model_name}) ---")
    coef_df = pd.DataFrame({
        "feature": feature_names,
        "coef": final.coef_,
        "abs_coef": np.abs(final.coef_),
    }).sort_values("abs_coef", ascending=False)

    nonzero = coef_df[coef_df["abs_coef"] > 1e-6]
    print(f"  {len(nonzero)} non-zero out of {len(feature_names)} features")
    print("\n  Top 25 selected features:")
    for _, row in nonzero.head(25).iterrows():
        sign = "+" if row["coef"] > 0 else "-"
        print(f"    {sign} {row['feature']:<40s} β={row['coef']:>+9.2f}")

    # Bootstrap confidence intervals (optional)
    if "--bootstrap" in sys.argv:
        from src.metrics import block_bootstrap_coefs
        print(f"\n--- Bootstrap 95% confidence intervals (200 replications) ---")
        print("    Block bootstrap (block=24h) preserves temporal dependence.\n")

        def fit_fn(X_b, y_b):
            if best_model_name == "Ridge":
                m = R(alpha=ridge.alpha_)
            elif best_model_name == "LASSO":
                m = Lasso(alpha=lasso.alpha_, max_iter=10000)
            else:
                m = EN(alpha=enet.alpha_, l1_ratio=enet.l1_ratio_, max_iter=10000)
            m.fit(X_b, y_b)
            return m.coef_

        boot = block_bootstrap_coefs(X_full_s, y_full.values, fit_fn,
                                     n_bootstrap=200)
        print(f"  Valid replications: {boot['n_valid']}/200\n")

        top_feats = nonzero.head(25)
        print(f"  {'Feature':<40s} {'β':>9s} {'95% CI':>22s} {'Sig?':>5s}")
        print("  " + "-" * 80)
        for _, row in top_feats.iterrows():
            idx = feature_names.index(row["feature"])
            lo = boot["ci_lower"][idx]
            hi = boot["ci_upper"][idx]
            sig = "*" if boot["significant"][idx] else ""
            print(f"  {row['feature']:<40s} {row['coef']:>+9.2f} "
                  f"[{lo:>+9.2f}, {hi:>+9.2f}] {sig:>5s}")

    # Save
    test_out = test[["id"]].copy()
    test_out[TARGET] = final_preds
    save_submission(test_out, SUBMISSION_NAME)

    print(f"\nDone. Submission: output/{SUBMISSION_NAME}")
