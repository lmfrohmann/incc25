"""Evaluation metrics and statistical tests for the INCC project."""

import numpy as np
from scipy import stats


def rmse(y_true, y_pred):
    """Root Mean Squared Error — the primary competition metric."""
    return np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2))


def mae(y_true, y_pred):
    """Mean Absolute Error."""
    return np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred)))


def r_squared(y_true, y_pred):
    """Coefficient of determination (R²)."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - ss_res / ss_tot


def adjusted_r_squared(y_true, y_pred, n_features):
    """Adjusted R² — penalises model complexity."""
    r2 = r_squared(y_true, y_pred)
    n = len(y_true)
    return 1 - (1 - r2) * (n - 1) / (n - n_features - 1)


def print_metrics(y_true, y_pred, label="Model", n_features=None):
    """Print RMSE, MAE, R², and optionally Adj. R² for a set of predictions."""
    print(f"\n{label} Performance Metrics:")
    print(f"  RMSE:      {rmse(y_true, y_pred):.4f}")
    print(f"  MAE:       {mae(y_true, y_pred):.4f}")
    print(f"  R-squared: {r_squared(y_true, y_pred):.4f}")
    if n_features is not None:
        print(f"  Adj. R²:   {adjusted_r_squared(y_true, y_pred, n_features):.4f}")


def _newey_west_var(d, max_lag=None):
    """Newey-West HAC variance estimator for a series d."""
    n = len(d)
    if max_lag is None:
        max_lag = int(np.floor(n ** (1 / 3)))  # Andrews (1991) rule
    d_demeaned = d - d.mean()
    gamma_0 = np.sum(d_demeaned ** 2) / n
    gamma_sum = 0.0
    for lag in range(1, max_lag + 1):
        weight = 1.0 - lag / (max_lag + 1)  # Bartlett kernel
        gamma_j = np.sum(d_demeaned[lag:] * d_demeaned[:-lag]) / n
        gamma_sum += 2 * weight * gamma_j
    return (gamma_0 + gamma_sum) / n


def diebold_mariano(y_true, y_pred_1, y_pred_2, loss="squared"):
    """Diebold-Mariano test for equal predictive accuracy."""
    y_true = np.asarray(y_true)
    y_pred_1 = np.asarray(y_pred_1)
    y_pred_2 = np.asarray(y_pred_2)

    e1 = y_true - y_pred_1
    e2 = y_true - y_pred_2

    if loss == "squared":
        d = e1 ** 2 - e2 ** 2
    elif loss == "absolute":
        d = np.abs(e1) - np.abs(e2)
    else:
        raise ValueError(f"Unknown loss: {loss}")

    n = len(d)
    d_bar = d.mean()
    var_d = _newey_west_var(d)

    if var_d < 1e-15:
        return {"statistic": 0.0, "p_value": 1.0, "mean_diff": d_bar,
                "conclusion": "Identical predictions"}

    dm_stat = d_bar / np.sqrt(var_d)
    p_value = 2.0 * stats.norm.sf(np.abs(dm_stat))  # two-sided

    if p_value < 0.01:
        sig = "significant at 1%"
    elif p_value < 0.05:
        sig = "significant at 5%"
    elif p_value < 0.10:
        sig = "significant at 10%"
    else:
        sig = "not significant"

    better = "Model 1" if d_bar < 0 else "Model 2"

    return {
        "statistic": dm_stat,
        "p_value": p_value,
        "mean_diff": d_bar,
        "conclusion": f"{better} is better ({sig})",
    }


def block_bootstrap_coefs(X, y, fit_fn, n_bootstrap=200, block_size=24,
                          seed=42):
    """Block bootstrap for coefficient confidence intervals."""
    rng = np.random.RandomState(seed)
    n, p = X.shape
    n_blocks = int(np.ceil(n / block_size))
    coef_samples = np.zeros((n_bootstrap, p))

    for b in range(n_bootstrap):
        # Draw random block start indices (circular)
        starts = rng.randint(0, n, size=n_blocks)
        indices = []
        for s in starts:
            indices.extend(range(s, min(s + block_size, n)))
        indices = np.array(indices[:n])  # trim to original length

        X_boot = X[indices]
        y_boot = y[indices]
        try:
            coef_samples[b] = fit_fn(X_boot, y_boot)
        except Exception:
            coef_samples[b] = np.nan

    # Drop failed replications
    valid = ~np.any(np.isnan(coef_samples), axis=1)
    coef_samples = coef_samples[valid]

    ci_lower = np.percentile(coef_samples, 2.5, axis=0)
    ci_upper = np.percentile(coef_samples, 97.5, axis=0)
    significant = ~((ci_lower <= 0) & (ci_upper >= 0))

    return {
        "coefs_mean": coef_samples.mean(axis=0),
        "coefs_std": coef_samples.std(axis=0),
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "significant": significant,
        "n_valid": valid.sum(),
    }
