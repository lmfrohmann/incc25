"""Feature engineering helpers for the INCC pumped storage forecasting project."""

import os
import numpy as np
import pandas as pd

# Plant metadata constants (from plant_metadata.csv)
TOTAL_PROD_CAPACITY = 5535.6   # MW
TOTAL_PUMP_CAPACITY = 5252.1   # MW

# Original feature functions (preserved for baseline models)

def add_time_features(df):
    """Add month, weekday (R-style: Sun=1..Sat=7), and hour columns."""
    df = df.copy()
    df["month"] = df["datetime_start"].dt.month
    df["weekday"] = ((df["datetime_start"].dt.dayofweek + 1) % 7) + 1
    df["hour"] = df["datetime_start"].dt.hour
    return df


def add_log_demands(df):
    """Add log-transformed demand columns for ES, FR, DE, PT (day-ahead)."""
    df = df.copy()
    df["log_es_demand_f_d1"] = np.log(df["es_demand_f_d1"].clip(lower=0.01))
    df["log_fr_demand_f_d1"] = np.log(df["fr_demand_f_d1"].clip(lower=0.01))
    df["log_de_demand_f_d1"] = np.log(df["de_demand_f_d1"].clip(lower=0.01))
    df["log_pt_demand_f_d1"] = np.log(df["pt_demand_f_d1"].clip(lower=0.01))
    return df


def add_squared_terms(df):
    """Add squared terms for temperature and gas price (day-ahead)."""
    df = df.copy()
    df["es_temp_f_d1_sq"] = df["es_temp_f_d1"] ** 2
    df["es_gas_market_price_d1_sq"] = df["es_gas_market_price_d1"] ** 2
    return df


def add_lag_features(df):
    """Add 1-period lagged values for solar and wind forecasts (d1 and d2)."""
    df = df.copy()
    df["lag_es_solar_f_d1"] = df["es_solar_f_d1"].shift(1)
    df["lag_es_solar_f_d2"] = df["es_solar_f_d2"].shift(1)
    df["lag_es_wind_f_d1"] = df["es_wind_f_d1"].shift(1)
    df["lag_es_wind_f_d2"] = df["es_wind_f_d2"].shift(1)
    return df


def add_residual_demand(df):
    """Add residual demand = demand - wind - solar, plus squared term."""
    df = df.copy()
    df["es_residualdemand_f_d1"] = (
        df["es_demand_f_d1"] - df["es_wind_f_d1"] - df["es_solar_f_d1"]
    )
    df["es_residualdemand_f_d1_sq"] = df["es_residualdemand_f_d1"] ** 2
    return df


def add_interaction_terms(df):
    """Add interaction features: residual_demand*wind and temp*demand."""
    df = df.copy()
    df["interaction_residual_wind"] = (
        df["es_residualdemand_f_d1"] * df["es_wind_f_d1"]
    )
    df["interaction_temp_demand"] = df["es_temp_f_d1"] * df["es_demand_f_d1"]
    return df


def add_all_rf_features(df):
    """Apply the full feature engineering pipeline used by the Random Forest model."""
    df = add_time_features(df)
    df = add_log_demands(df)
    df = add_squared_terms(df)
    df = add_lag_features(df)
    df = add_residual_demand(df)
    df = add_interaction_terms(df)
    return df


# Enhanced feature engineering for competition models

def _load_unavailability_timeseries():
    """Load and pivot unavailability data into hourly total MW unavailable."""
    data_raw = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "raw")

    results = {}
    for kind in ("prod", "cons"):
        path = os.path.join(data_raw, f"{kind}_unavailable.csv")
        udf = pd.read_csv(path)
        udf["datetime_start"] = pd.to_datetime(udf["datetime_start"], utc=True)
        # Sum unavailable MW across all units per hour
        agg = udf.groupby("datetime_start")["unavailable"].sum().reset_index()
        agg = agg.rename(columns={"unavailable": f"total_{kind}_unavail_mw"})
        results[kind] = agg
    return results


def add_unavailability_features(df):
    """Merge hourly unavailability totals and derive available capacity."""
    df = df.copy()
    unavail = _load_unavailability_timeseries()

    for kind in ("prod", "cons"):
        df = df.merge(unavail[kind], on="datetime_start", how="left")
        col = f"total_{kind}_unavail_mw"
        df[col] = df[col].fillna(0)

    df["avail_prod_capacity"] = TOTAL_PROD_CAPACITY - df["total_prod_unavail_mw"]
    df["avail_pump_capacity"] = TOTAL_PUMP_CAPACITY - df["total_cons_unavail_mw"]
    df["capacity_asymmetry"] = df["avail_prod_capacity"] - df["avail_pump_capacity"]
    return df


def add_enhanced_time_features(df):
    """Cyclical time encoding + calendar features."""
    df = df.copy()
    df["month"] = df["datetime_start"].dt.month
    df["weekday"] = ((df["datetime_start"].dt.dayofweek + 1) % 7) + 1
    df["hour"] = df["datetime_start"].dt.hour
    df["is_weekend"] = (df["datetime_start"].dt.dayofweek >= 5).astype(int)
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    df["day_of_year"] = df["datetime_start"].dt.dayofyear
    df["doy_sin"] = np.sin(2 * np.pi * df["day_of_year"] / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * df["day_of_year"] / 365.25)
    # Pump/produce regime indicator: 1=produce hours, -1=pump hours, 0=transition
    df["regime"] = 0
    df.loc[df["hour"].between(10, 15), "regime"] = -1   # pump
    df.loc[df["hour"].between(18, 21), "regime"] = 1     # produce
    # Transition hour indicator (H07-09 pump ramp, H16-18 produce ramp — highest error hours)
    df["is_transition"] = 0
    df.loc[df["hour"].between(7, 9), "is_transition"] = 1
    df.loc[df["hour"].between(16, 18), "is_transition"] = 1
    return df


def add_spread_features(df):
    """Arbitrage / price signal features that drive pump-storage dispatch."""
    df = df.copy()
    # Implied electricity cost from gas (from docs formula)
    df["implied_elec_price_d1"] = (df["es_gas_market_price_d1"] + 0.2 * df["eua_price"]) / 0.5

    # Residual demands (d1 and d2)
    df["es_residualdemand_f_d1"] = df["es_demand_f_d1"] - df["es_wind_f_d1"] - df["es_solar_f_d1"]
    df["es_residualdemand_f_d2"] = df["es_demand_f_d2"] - df["es_wind_f_d2"] - df["es_solar_f_d2"]
    df["es_residualdemand_f_d1_sq"] = df["es_residualdemand_f_d1"] ** 2

    # Solar penetration ratio (how much of demand is covered by solar)
    df["solar_penetration_d1"] = df["es_solar_f_d1"] / df["es_demand_f_d1"].clip(lower=1)
    df["solar_penetration_d2"] = df["es_solar_f_d2"] / df["es_demand_f_d2"].clip(lower=1)

    # Total renewable infeed for Spain
    df["es_total_res_d1"] = df["es_wind_f_d1"] + df["es_solar_f_d1"]
    df["es_total_res_d2"] = df["es_wind_f_d2"] + df["es_solar_f_d2"]

    # RES ratio to demand
    df["es_res_ratio_d1"] = df["es_total_res_d1"] / df["es_demand_f_d1"].clip(lower=1)

    # Iberian aggregates (ES + PT)
    df["iberian_demand_d1"] = df["es_demand_f_d1"] + df["pt_demand_f_d1"]
    df["iberian_res_d1"] = df["es_wind_f_d1"] + df["es_solar_f_d1"] + df["pt_wind_f_d1"] + df["pt_solar_f_d1"]
    df["iberian_residual_d1"] = df["iberian_demand_d1"] - df["iberian_res_d1"]

    # France surplus/deficit (drives cross-border flows)
    df["fr_surplus_d1"] = (
        df["fr_wind_f_d1"] + df["fr_solar_f_d1"] + df["fr_hydro_ror_f_d1"] + df["fr_hydro_res_f_d1"]
        - df["fr_demand_f_d1"]
    )

    # d2 minus d1 deltas (trend signals — what's changing tomorrow)
    df["delta_es_demand"] = df["es_demand_f_d2"] - df["es_demand_f_d1"]
    df["delta_es_solar"] = df["es_solar_f_d2"] - df["es_solar_f_d1"]
    df["delta_es_wind"] = df["es_wind_f_d2"] - df["es_wind_f_d1"]
    df["delta_es_residual"] = df["es_residualdemand_f_d2"] - df["es_residualdemand_f_d1"]

    # Solar deviation from normal (d1)
    df["solar_dev_d1"] = df["es_solar_f_d1"] - df["es_solar_n_d1"]
    df["wind_dev_d1"] = df["es_wind_f_d1"] - df["es_wind_n_d1"]

    return df


def add_enhanced_interactions(df):
    """Key interaction terms driven by domain knowledge."""
    df = df.copy()
    # Hour interactions with key drivers
    df["hour_x_residual"] = df["hour"] * df["es_residualdemand_f_d1"]
    df["hour_x_solar"] = df["hour"] * df["es_solar_f_d1"]
    df["hour_x_solar_pen"] = df["hour"] * df["solar_penetration_d1"]
    # Regime interactions
    df["regime_x_residual"] = df["regime"] * df["es_residualdemand_f_d1"]
    df["regime_x_solar"] = df["regime"] * df["es_solar_f_d1"]
    # Original interactions
    df["interaction_residual_wind"] = df["es_residualdemand_f_d1"] * df["es_wind_f_d1"]
    df["interaction_temp_demand"] = df["es_temp_f_d1"] * df["es_demand_f_d1"]
    # Capacity interactions
    if "avail_prod_capacity" in df.columns:
        df["regime_x_avail_prod"] = df["regime"] * df["avail_prod_capacity"]
        df["regime_x_avail_pump"] = df["regime"] * df["avail_pump_capacity"]
    # Weekend interactions (weekend dispatch differs by ~85 MW)
    df["weekend_x_residual"] = df["is_weekend"] * df["es_residualdemand_f_d1"]
    df["weekend_x_regime"] = df["is_weekend"] * df["regime"]
    # Transition hour interactions (worst RMSE hours)
    df["transition_x_residual"] = df["is_transition"] * df["es_residualdemand_f_d1"]
    return df


def add_lag_features_enhanced(df):
    """Extended lags for key features."""
    df = df.copy()
    for col in ["es_solar_f_d1", "es_solar_f_d2", "es_wind_f_d1", "es_wind_f_d2"]:
        df[f"lag_{col}"] = df[col].shift(1)
    # Demand lag
    df["lag_es_demand_f_d1"] = df["es_demand_f_d1"].shift(1)
    # Log demands
    df["log_es_demand_f_d1"] = np.log(df["es_demand_f_d1"].clip(lower=1))
    df["log_fr_demand_f_d1"] = np.log(df["fr_demand_f_d1"].clip(lower=1))
    df["log_de_demand_f_d1"] = np.log(df["de_demand_f_d1"].clip(lower=1))
    df["log_pt_demand_f_d1"] = np.log(df["pt_demand_f_d1"].clip(lower=1))
    # Squared terms
    df["es_temp_f_d1_sq"] = df["es_temp_f_d1"] ** 2
    df["es_gas_market_price_d1_sq"] = df["es_gas_market_price_d1"] ** 2
    df["log_eua_price"] = np.log(df["eua_price"].clip(lower=0.01))
    return df


def add_rolling_features(df):
    """Add rolling statistics for key features. Uses .shift(1) to avoid leakage."""
    df = df.copy()
    rolling_cols = {
        "es_residualdemand_f_d1": ["mean_24", "std_24", "mean_168", "momentum_24"],
        "es_wind_f_d1": ["mean_24", "std_24", "momentum_24"],
        "es_solar_f_d1": ["mean_24", "std_24", "momentum_24"],
        "es_demand_f_d1": ["mean_24", "std_24", "mean_168"],
    }
    for col, ops in rolling_cols.items():
        shifted = df[col].shift(1)  # avoid leakage
        if "mean_24" in ops or "momentum_24" in ops:
            rm24 = shifted.rolling(24, min_periods=1).mean()
        if "mean_24" in ops:
            df[f"{col}_rm24"] = rm24
        if "std_24" in ops:
            df[f"{col}_rstd24"] = shifted.rolling(24, min_periods=1).std()
        if "mean_168" in ops:
            df[f"{col}_rm168"] = shifted.rolling(168, min_periods=1).mean()
        if "momentum_24" in ops:
            df[f"{col}_mom24"] = df[col] - rm24
    return df


def add_extended_lag_features(df):
    """Add 24h and 48h lags for key features."""
    df = df.copy()
    for col in ["es_solar_f_d1", "es_wind_f_d1", "es_demand_f_d1", "es_residualdemand_f_d1"]:
        df[f"{col}_lag24"] = df[col].shift(24)
        df[f"{col}_lag48"] = df[col].shift(48)
    return df


def add_all_enhanced_features(df):
    """Full enhanced feature pipeline for competition models."""
    df = add_unavailability_features(df)
    df = add_enhanced_time_features(df)
    df = add_spread_features(df)
    df = add_lag_features_enhanced(df)
    df = add_enhanced_interactions(df)
    df = add_rolling_features(df)
    df = add_extended_lag_features(df)
    return df


def add_all_enhanced_features_combined(train_df, test_df):
    """Apply enhanced features separately to preserve CV integrity."""
    n_train = len(train_df)

    # Train: features from train data only (CV-safe)
    train_clean = add_all_enhanced_features(train_df)

    # Test: concat so rolling windows see trailing train context
    combined = pd.concat([train_df, test_df], ignore_index=True)
    combined = add_all_enhanced_features(combined)
    test_clean = combined.iloc[n_train:].reset_index(drop=True)

    # Wind extreme indicator: threshold from training data only
    wind_col = "es_wind_f_d1"
    if wind_col in train_clean.columns:
        wind_p90 = train_clean[wind_col].quantile(0.90)
        for df in (train_clean, test_clean):
            df["wind_extreme"] = (df[wind_col] > wind_p90).astype(int)
            df["wind_extreme_x_regime"] = df["wind_extreme"] * df["regime"]

    return train_clean.reset_index(drop=True), test_clean


# Feature lists

# Original feature lists (preserved for baseline models)
RF_FEATURES = [
    "es_temp_f_d1", "es_temp_f_d1_sq",
    "log_es_demand_f_d1", "log_fr_demand_f_d1",
    "log_de_demand_f_d1", "log_pt_demand_f_d1",
    "es_hydro_ror_f_d1", "fr_hydro_ror_f_d1", "es_hydro_inflow_f_d1",
    "fr_wind_f_d1", "pt_wind_f_d1",
    "lag_es_solar_f_d1", "lag_es_solar_f_d2",
    "lag_es_wind_f_d1", "lag_es_wind_f_d2",
    "es_residualdemand_f_d1", "es_residualdemand_f_d1_sq",
    "es_gas_market_price_d1", "es_gas_market_price_d1_sq",
    "interaction_residual_wind", "interaction_temp_demand",
    "month", "weekday", "hour",
]

LR_FEATURES = [
    "es_temp_f_d1", "es_temp_f_d1_sq",
    "log_es_demand_f_d1", "log_fr_demand_f_d1",
    "log_de_demand_f_d1", "log_pt_demand_f_d1",
    "es_hydro_ror_f_d1", "fr_hydro_ror_f_d1", "es_hydro_inflow_f_d1",
    "fr_wind_f_d1", "pt_wind_f_d1",
    "lag_es_solar_f_d1",
    "log_eua_price",
    "es_residualdemand_f_d1",
]

# Full feature set for GBM models — use all raw + engineered
GBM_FEATURES = [
    # Time features
    "hour", "month", "weekday", "is_weekend",
    "hour_sin", "hour_cos", "month_sin", "month_cos",
    "doy_sin", "doy_cos", "regime", "is_transition",
    # Raw d1 features (weather, generation, demand)
    "es_temp_f_d1", "es_precip_f_d1",
    "es_demand_f_d1", "fr_demand_f_d1", "de_demand_f_d1", "pt_demand_f_d1",
    "es_wind_f_d1", "fr_wind_f_d1", "de_wind_f_d1", "pt_wind_f_d1",
    "es_solar_f_d1", "fr_solar_f_d1", "de_solar_f_d1", "pt_solar_f_d1",
    "es_wind_n_d1", "es_solar_n_d1", "fr_solar_n_d1",
    "es_hydro_ror_f_d1", "fr_hydro_ror_f_d1", "pt_hydro_ror_f_d1",
    "es_hydro_res_f_d1", "fr_hydro_res_f_d1",
    "es_hydro_inflow_f_d1", "es_hydro_balance_f_d1",
    "es_wind_speed_f_d1",
    "es_gas_market_price_d1", "eua_price",
    "fr_es_ntc_d1", "es_fr_ntc_d1",
    # Raw d2 features
    "es_temp_f_d2", "es_precip_f_d2",
    "es_demand_f_d2", "fr_demand_f_d2", "de_demand_f_d2", "pt_demand_f_d2",
    "es_wind_f_d2", "fr_wind_f_d2", "de_wind_f_d2", "pt_wind_f_d2",
    "es_solar_f_d2", "fr_solar_f_d2", "de_solar_f_d2", "pt_solar_f_d2",
    "es_wind_n_d2", "es_solar_n_d2", "fr_solar_n_d2",
    "es_hydro_ror_f_d2", "fr_hydro_ror_f_d2", "pt_hydro_ror_f_d2",
    "es_hydro_res_f_d2", "fr_hydro_res_f_d2",
    "es_hydro_inflow_f_d2", "es_hydro_balance_f_d2",
    "es_wind_speed_f_d2",
    "es_gas_market_price_d2",
    # Unavailability features
    "total_prod_unavail_mw", "total_cons_unavail_mw",
    "avail_prod_capacity", "avail_pump_capacity", "capacity_asymmetry",
    # Engineered spread / arbitrage
    "implied_elec_price_d1",
    "es_residualdemand_f_d1", "es_residualdemand_f_d2", "es_residualdemand_f_d1_sq",
    "solar_penetration_d1", "solar_penetration_d2",
    "es_total_res_d1", "es_total_res_d2", "es_res_ratio_d1",
    "iberian_demand_d1", "iberian_res_d1", "iberian_residual_d1",
    "fr_surplus_d1",
    "delta_es_demand", "delta_es_solar", "delta_es_wind", "delta_es_residual",
    "solar_dev_d1", "wind_dev_d1",
    # Lags
    "lag_es_solar_f_d1", "lag_es_solar_f_d2",
    "lag_es_wind_f_d1", "lag_es_wind_f_d2",
    "lag_es_demand_f_d1",
    # Log transforms
    "log_es_demand_f_d1", "log_fr_demand_f_d1",
    "log_de_demand_f_d1", "log_pt_demand_f_d1",
    "log_eua_price",
    # Squared
    "es_temp_f_d1_sq", "es_gas_market_price_d1_sq",
    # Interactions
    "hour_x_residual", "hour_x_solar", "hour_x_solar_pen",
    "regime_x_residual", "regime_x_solar",
    "interaction_residual_wind", "interaction_temp_demand",
    "regime_x_avail_prod", "regime_x_avail_pump",
    "weekend_x_residual", "weekend_x_regime",
    "transition_x_residual",
    # Rolling features (13)
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
    # Wind extreme features
    "wind_extreme", "wind_extreme_x_regime",
]


def clip_predictions(preds, df=None):
    """Clip predictions to physical capacity bounds."""
    if df is not None and "avail_prod_capacity" in df.columns:
        upper = df["avail_prod_capacity"].values
        lower = -df["avail_pump_capacity"].values
        return np.clip(preds, lower, upper)
    return np.clip(preds, -TOTAL_PUMP_CAPACITY, TOTAL_PROD_CAPACITY)
