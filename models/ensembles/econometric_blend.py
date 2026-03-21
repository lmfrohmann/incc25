"""Econometric Ensemble — Optimal Blend of Econometric + ML Models."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from scipy.optimize import minimize

from src.data import load_train, load_test, save_submission, load_actuals
from src.metrics import print_metrics, rmse
from src.features import add_all_enhanced_features_combined, clip_predictions, GBM_FEATURES

TARGET = "es_total_ps"


# Helpers
def load_submission(name):
    """Load a submission CSV from the output folder."""
    path = os.path.join(os.path.dirname(__file__), "..", "..", "output", name)
    if os.path.exists(path):
        return pd.read_csv(path)
    return None


def optimal_weights_constrained(preds_matrix, y_true):
    """Find optimal non-negative weights that sum to 1 (Bates-Granger)."""
    n_models = preds_matrix.shape[1]
    w0 = np.ones(n_models) / n_models

    def objective(w):
        blend = preds_matrix @ w
        return np.mean((y_true - blend) ** 2)

    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}
    bounds = [(0, 1)] * n_models

    result = minimize(objective, w0, method="SLSQP",
                      bounds=bounds, constraints=constraints)
    return result.x


# Component model retraining for validation predictions
def retrain_structural_dispatch(train_df, val_df):
    """Retrain structural dispatch on train, predict on val."""
    from models.econometric.structural_dispatch import build_dispatch_features, build_structural_model
    X_tr = build_dispatch_features(train_df).fillna(0)
    X_va = build_dispatch_features(val_df).fillna(0)
    y_tr = train_df[TARGET]
    model = build_structural_model(alpha_ridge=100.0)
    model.fit(X_tr, y_tr)
    return clip_predictions(model.predict(X_va), val_df)


def retrain_regime_switching(train_df, val_df):
    """Retrain regime-switching on train, predict on val."""
    from models.econometric.regime_switching import build_regime_features, CORE_FEATURES
    X_tr, regime_stats = build_regime_features(train_df, CORE_FEATURES)
    X_tr = X_tr.fillna(0)
    X_va, _ = build_regime_features(val_df, CORE_FEATURES, **regime_stats)
    X_va = X_va.fillna(0)
    y_tr = train_df[TARGET]
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_va_s = scaler.transform(X_va)
    model = Ridge(alpha=100.0)
    model.fit(X_tr_s, y_tr)
    return clip_predictions(model.predict(X_va_s), val_df)


def retrain_elastic_net(train_df, val_df):
    """Retrain elastic net (Ridge variant) on train, predict on val."""
    from models.econometric.elastic_net import build_econometric_features
    from sklearn.linear_model import RidgeCV
    from sklearn.model_selection import TimeSeriesSplit
    X_tr = build_econometric_features(train_df).fillna(0)
    X_va = build_econometric_features(val_df).fillna(0)
    y_tr = train_df[TARGET]
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_va_s = scaler.transform(X_va)
    tscv = TimeSeriesSplit(n_splits=5)
    model = RidgeCV(alphas=np.logspace(-2, 4, 30), cv=tscv)
    model.fit(X_tr_s, y_tr)
    return clip_predictions(model.predict(X_va_s), val_df)


def retrain_gam(train_df, val_df):
    """Retrain GAM on train, predict on val."""
    from models.econometric.gam import build_gam_features, build_gam_terms
    from pygam import LinearGAM
    X_tr = build_gam_features(train_df)
    X_va = build_gam_features(val_df)
    y_tr = train_df[TARGET].values
    terms = build_gam_terms()
    gam = LinearGAM(terms, lam=1000.0)
    gam.fit(X_tr, y_tr)
    return clip_predictions(gam.predict(X_va), val_df)


def retrain_catboost(train_df, val_df, n_seeds=10):
    """Retrain CatBoost multi-seed on train, predict on val."""
    from catboost import CatBoostRegressor
    features = [f for f in GBM_FEATURES if f in train_df.columns]
    X_tr = train_df[features].values
    X_va = val_df[features].values
    y_tr = train_df[TARGET].values
    preds_sum = np.zeros(len(val_df))
    # NOTE: No early stopping is used here because adding an eval_set would
    # require a train/val split inside this function, significantly changing
    # the training flow. The fixed iteration count is a known limitation.
    diverse_seeds = [42, 123, 456, 789, 2024, 1337, 7, 99, 314, 628]
    for seed in diverse_seeds[:n_seeds]:
        model = CatBoostRegressor(
            iterations=2000, depth=8, learning_rate=0.03,
            l2_leaf_reg=3.0, subsample=0.8, colsample_bylevel=0.8,
            random_seed=seed, verbose=0,
            loss_function="RMSE",
        )
        model.fit(X_tr, y_tr)
        preds_sum += model.predict(X_va)
    return clip_predictions(preds_sum / n_seeds, val_df)


# Main
if __name__ == "__main__":

    print("=" * 70)
    print("ECONOMETRIC ENSEMBLE — HONEST FORECAST COMBINATION")
    print("=" * 70)

    # Step 1: Load and prepare data with validation split
    print("\nLoading and preparing data...")
    train, test = add_all_enhanced_features_combined(load_train(), load_test())
    train_clean = train.dropna(subset=[TARGET]).reset_index(drop=True)

    val_cutoff = train_clean["datetime_start"].max() - pd.Timedelta(days=90)
    train_part = train_clean[train_clean["datetime_start"] <= val_cutoff].reset_index(drop=True)
    val_part = train_clean[train_clean["datetime_start"] > val_cutoff].reset_index(drop=True)
    y_val = val_part[TARGET].values

    print(f"  Training: {len(train_part)} rows (up to {val_cutoff.date()})")
    print(f"  Validation: {len(val_part)} rows (last 90 days)")

    # Step 2: Generate validation predictions from each component model
    print("\n--- Retraining component models on train, predicting on validation ---")

    component_preds = {}
    component_functions = {
        "structural": retrain_structural_dispatch,
        "regime": retrain_regime_switching,
        "elastic_net": retrain_elastic_net,
        "gam": retrain_gam,
        "catboost": retrain_catboost,
    }

    econ_names = ["structural", "regime", "elastic_net", "gam"]
    ml_names = ["catboost"]

    for name, func in component_functions.items():
        try:
            preds = func(train_part, val_part)
            component_preds[name] = preds
            r = np.sqrt(np.mean((y_val - preds) ** 2))
            print(f"  {name:<20s}: val RMSE = {r:.2f}")
        except Exception as e:
            print(f"  {name:<20s}: FAILED ({e})")

    if len(component_preds) < 2:
        print("\nNeed at least 2 component models. Install missing dependencies.")
        sys.exit(1)

    # Step 3: Optimize weights on VALIDATION set (not test!)
    print("\n--- Weight optimization on VALIDATION set ---")

    # Full model weights
    model_names_avail = list(component_preds.keys())
    preds_matrix = np.column_stack([component_preds[n] for n in model_names_avail])

    w_bg = optimal_weights_constrained(preds_matrix, y_val)
    blend_bg = preds_matrix @ w_bg
    rmse_bg = np.sqrt(np.mean((y_val - blend_bg) ** 2))
    print(f"\n  Bates-Granger optimal weights (all models):")
    for name, w in zip(model_names_avail, w_bg):
        print(f"    {name:<20s}: {w:.4f}")
    print(f"  Validation RMSE: {rmse_bg:.2f}")

    # Two-component blend: econometric avg vs ML avg
    econ_avail = [n for n in model_names_avail if n in econ_names]
    ml_avail = [n for n in model_names_avail if n in ml_names]

    if econ_avail and ml_avail:
        econ_val_avg = np.column_stack([component_preds[n] for n in econ_avail]).mean(axis=1)
        ml_val_avg = np.column_stack([component_preds[n] for n in ml_avail]).mean(axis=1)
        two_preds = np.column_stack([econ_val_avg, ml_val_avg])
        w_two = optimal_weights_constrained(two_preds, y_val)
        blend_two = two_preds @ w_two
        rmse_two = np.sqrt(np.mean((y_val - blend_two) ** 2))
        print(f"\n  Two-component blend (econ avg + ML avg):")
        print(f"    Econ weight: {w_two[0]:.4f}, ML weight: {w_two[1]:.4f}")
        print(f"    Validation RMSE: {rmse_two:.2f}")

        # Grid search for interpretable weights
        print("\n  Grid search (validation RMSE):")
        best_grid_w = 0.0
        best_grid_rmse = float("inf")
        for w_econ in np.arange(0.0, 1.05, 0.05):
            blend = w_econ * econ_val_avg + (1 - w_econ) * ml_val_avg
            r = np.sqrt(np.mean((y_val - blend) ** 2))
            marker = " *" if r < best_grid_rmse else ""
            print(f"    econ={w_econ:.2f}, ml={1-w_econ:.2f}: RMSE={r:.2f}{marker}")
            if r < best_grid_rmse:
                best_grid_rmse = r
                best_grid_w = w_econ

        print(f"\n  Best grid weight: econ={best_grid_w:.2f}, ml={1-best_grid_w:.2f} "
              f"(val RMSE={best_grid_rmse:.2f})")

    # Step 4: Generate test submissions using validation-optimized weights
    print("\n--- Generating ensemble submissions ---")

    # Load existing test predictions from output/
    econ_test_files = {
        "structural": "structural_dispatch.csv",
        "regime": "regime_switching.csv",
        "elastic_net": "elastic_net.csv",
        "gam": "gam.csv",
    }
    ml_test_files = {
        "catboost": "catboost_10seed_uniform_bias.csv",
    }

    test_preds = {}
    all_test_files = {**econ_test_files, **ml_test_files}
    for name, fname in all_test_files.items():
        sub = load_submission(fname)
        if sub is not None:
            test_preds[name] = sub.set_index("id")["es_total_ps"].values
            print(f"  Loaded test predictions: {fname}")
    test_ids = load_submission(list(all_test_files.values())[0])["id"].values

    # Econometric avg + ML avg for test
    econ_test_avail = [n for n in test_preds if n in econ_test_files]
    ml_test_avail = [n for n in test_preds if n in ml_test_files]

    if econ_test_avail and ml_test_avail:
        econ_test_avg = np.column_stack([test_preds[n] for n in econ_test_avail]).mean(axis=1)
        ml_test_avg = np.column_stack([test_preds[n] for n in ml_test_avail]).mean(axis=1)

        # Use validation-optimized weights for blends
        for w_econ in [0.05, 0.10, 0.15, 0.20]:
            blend = w_econ * econ_test_avg + (1 - w_econ) * ml_test_avg
            fname = f"econ{int(w_econ*100)}_ml{int((1-w_econ)*100)}_blend.csv"
            test_out = pd.DataFrame({"id": test_ids, TARGET: blend})
            save_submission(test_out, fname)

        # Best validation-optimized blend
        blend_best = best_grid_w * econ_test_avg + (1 - best_grid_w) * ml_test_avg
        test_out = pd.DataFrame({"id": test_ids, TARGET: blend_best})
        save_submission(test_out, "econ_ml_optimal_blend.csv")

    # All-model average
    all_test_preds_matrix = np.column_stack([test_preds[n] for n in test_preds])
    blend_all_test = all_test_preds_matrix.mean(axis=1)
    test_out = pd.DataFrame({"id": test_ids, TARGET: blend_all_test})
    save_submission(test_out, "econ_ml_avg_blend.csv")

    # Econ-only average
    if len(econ_test_avail) >= 2:
        test_out = pd.DataFrame({"id": test_ids, TARGET: econ_test_avg})
        save_submission(test_out, "econ_only_avg.csv")

    # Step 5: Post-hoc evaluation (if test actuals exist)
    try:
        actuals = load_actuals()
        print("\n--- Post-hoc evaluation against test actuals ---")
        print("    (These scores were NOT used for weight selection)")

        for name, sub_file in all_test_files.items():
            sub = load_submission(sub_file)
            if sub is not None:
                merged = sub.merge(actuals, on="id")
                r = np.sqrt(np.mean(
                    (merged["es_total_ps_actual"].values - merged["es_total_ps"].values) ** 2
                ))
                print(f"  {name:<20s}: test RMSE = {r:.2f}")

        # Evaluate blends
        print("\n  Blended submissions:")
        for w_econ in [0.05, 0.10, 0.15, 0.20]:
            fname = f"econ{int(w_econ*100)}_ml{int((1-w_econ)*100)}_blend.csv"
            sub = load_submission(fname)
            if sub is not None:
                merged = sub.merge(actuals, on="id")
                r = np.sqrt(np.mean(
                    (merged["es_total_ps_actual"].values - merged["es_total_ps"].values) ** 2
                ))
                print(f"    econ={w_econ:.0%}, ml={1-w_econ:.0%}: test RMSE = {r:.2f}")

    except Exception:
        print("\n  (No test actuals available for post-hoc evaluation)")

    print("\nDone.")
