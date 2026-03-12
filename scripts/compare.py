"""Compare submission CSVs against test actuals with interactive plots.

Usage:
    python scripts/compare.py output/catboost_10seed_uniform_bias.csv output/catboost_regime_blend_86_14.csv
"""

import sys
import os
import math

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.data import load_actuals, load_test
from src.metrics import rmse, mae, r_squared, adjusted_r_squared
from src.plotting.config import PALETTE, COLORS, apply_layout


def load_submissions(paths):
    """Load submission CSVs and return {name: DataFrame} dict."""
    subs = {}
    for p in paths:
        name = os.path.splitext(os.path.basename(p))[0]
        subs[name] = pd.read_csv(p)
    return subs


def build_comparison_df(actuals, test, submissions):
    """Merge actuals, test datetimes, and all submissions into one DataFrame."""
    df = actuals.merge(test[["id", "datetime_start"]], on="id")
    df["hour"] = df["datetime_start"].dt.hour
    for name, sub in submissions.items():
        df = df.merge(sub.rename(columns={"es_total_ps": name}), on="id")
    return df.sort_values("datetime_start").reset_index(drop=True)


def print_metrics_table(df, names, n_features=111):
    """Print a metrics table to console."""
    actual = df["es_total_ps_actual"].values
    print(f"\n{'Submission':<35s} | {'RMSE':>8s} | {'MAE':>8s} | {'R²':>8s} | {'Adj. R²':>8s}")
    print("-" * 80)
    for name in names:
        pred = df[name].values
        print(
            f"{name:<35s} | {rmse(actual, pred):>8.2f} | "
            f"{mae(actual, pred):>8.2f} | {r_squared(actual, pred):>8.4f} | "
            f"{adjusted_r_squared(actual, pred, n_features):>8.4f}"
        )


def plot_timeseries(df, names):
    """Time series: actuals vs each submission."""
    fig = go.Figure()
    fig.add_trace(go.Scattergl(
        x=df["datetime_start"], y=df["es_total_ps_actual"],
        mode="lines", name="Actual",
        line=dict(color=COLORS["neutral"], width=1), opacity=0.6,
    ))
    for i, name in enumerate(names):
        rolled = df[name].rolling(24, center=True).mean()
        fig.add_trace(go.Scattergl(
            x=df["datetime_start"], y=rolled,
            mode="lines", name=f"{name} (24h MA)",
            line=dict(color=PALETTE[i % len(PALETTE)], width=2),
        ))
    actual_ma = df["es_total_ps_actual"].rolling(24, center=True).mean()
    fig.add_trace(go.Scattergl(
        x=df["datetime_start"], y=actual_ma,
        mode="lines", name="Actual (24h MA)",
        line=dict(color="black", width=2, dash="dot"),
    ))
    apply_layout(fig, title="Submissions vs Actuals — Time Series",
                 xaxis_title="Time", yaxis_title="MW")
    return fig


def plot_scatter(df, names):
    """Scatter: predicted vs actual with 45-degree line, one subplot per submission."""
    ncols = min(len(names), 3)
    nrows = math.ceil(len(names) / ncols)
    fig = make_subplots(rows=nrows, cols=ncols, subplot_titles=names)

    actual = df["es_total_ps_actual"].values
    lo, hi = actual.min(), actual.max()

    for i, name in enumerate(names):
        r, c = divmod(i, ncols)
        pred = df[name].values
        fig.add_trace(go.Scattergl(
            x=actual, y=pred, mode="markers",
            marker=dict(color=PALETTE[i % len(PALETTE)], size=2, opacity=0.3),
            name=name, showlegend=False,
        ), row=r + 1, col=c + 1)
        fig.add_trace(go.Scatter(
            x=[lo, hi], y=[lo, hi], mode="lines",
            line=dict(color="black", dash="dash", width=1),
            showlegend=False,
        ), row=r + 1, col=c + 1)
        fig.update_xaxes(title_text="Actual", row=r + 1, col=c + 1)
        fig.update_yaxes(title_text="Predicted", row=r + 1, col=c + 1)

    apply_layout(fig, title="Predicted vs Actual",
                 height=400 * nrows, width=400 * ncols)
    return fig


def plot_error_distribution(df, names):
    """Histogram of (predicted - actual) per submission."""
    fig = go.Figure()
    actual = df["es_total_ps_actual"].values
    for i, name in enumerate(names):
        errors = df[name].values - actual
        fig.add_trace(go.Histogram(
            x=errors, name=name, opacity=0.6,
            marker_color=PALETTE[i % len(PALETTE)], nbinsx=80,
        ))
    fig.add_vline(x=0, line_dash="dash", line_color="black")
    apply_layout(fig, title="Error Distribution (Predicted − Actual)",
                 xaxis_title="Error (MW)", yaxis_title="Count")
    fig.update_layout(barmode="overlay")
    return fig


def plot_hourly_error(df, names):
    """Mean error by hour-of-day to show systematic biases."""
    fig = go.Figure()
    actual = df["es_total_ps_actual"]
    for i, name in enumerate(names):
        hourly_err = (df[name] - actual).groupby(df["hour"]).mean()
        fig.add_trace(go.Bar(
            x=hourly_err.index, y=hourly_err.values, name=name,
            marker_color=PALETTE[i % len(PALETTE)], opacity=0.7,
        ))
    fig.add_hline(y=0, line_dash="dash", line_color="black")
    apply_layout(fig, title="Mean Error by Hour of Day",
                 xaxis_title="Hour", yaxis_title="Mean Error (MW)")
    fig.update_layout(barmode="group")
    return fig


def main():
    if len(sys.argv) < 2:
        print("Usage: python src/compare_submissions.py <submission.csv> [submission2.csv ...]")
        sys.exit(1)

    actuals = load_actuals()
    test = load_test()
    submissions = load_submissions(sys.argv[1:])
    names = list(submissions.keys())

    df = build_comparison_df(actuals, test, submissions)
    print_metrics_table(df, names)

    plot_timeseries(df, names).show()
    plot_scatter(df, names).show()
    plot_error_distribution(df, names).show()
    plot_hourly_error(df, names).show()


if __name__ == "__main__":
    main()
