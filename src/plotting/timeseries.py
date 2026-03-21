"""Time series visualisation utilities."""

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .config import (
    PALETTE, COLORS, FEATURE_GROUPS, TARGET,
    apply_layout, filter_existing_columns,
)


def plot_timeseries(df, columns, title=None, rolling_window=None, height=500):
    """Plot one or more columns over datetime_start."""
    if isinstance(columns, str):
        columns = [columns]
    columns = filter_existing_columns(df, columns)
    if not columns:
        raise ValueError("None of the requested columns exist in the DataFrame.")

    fig = go.Figure()
    for i, col in enumerate(columns):
        color = PALETTE[i % len(PALETTE)]
        fig.add_trace(go.Scattergl(
            x=df["datetime_start"], y=df[col],
            mode="lines", name=col, line=dict(color=color, width=1),
            opacity=0.8,
        ))
        if rolling_window:
            rolled = df[col].rolling(rolling_window, center=True).mean()
            fig.add_trace(go.Scattergl(
                x=df["datetime_start"], y=rolled,
                mode="lines",
                name=f"{col} (MA-{rolling_window})",
                line=dict(color=color, width=2, dash="dot"),
            ))

    apply_layout(fig, title=title or ", ".join(columns),
                 xaxis_title="Time", yaxis_title="Value", height=height)
    return fig


def plot_target(df, rolling_window=24, height=500):
    """Plot the target variable with generation/pumping colour coding."""
    fig = go.Figure()

    gen_mask = df[TARGET] >= 0
    pump_mask = df[TARGET] < 0

    fig.add_trace(go.Scattergl(
        x=df.loc[gen_mask, "datetime_start"],
        y=df.loc[gen_mask, TARGET],
        mode="markers", marker=dict(color=COLORS["positive"], size=2),
        name="Generation (>= 0)",
    ))
    fig.add_trace(go.Scattergl(
        x=df.loc[pump_mask, "datetime_start"],
        y=df.loc[pump_mask, TARGET],
        mode="markers", marker=dict(color=COLORS["negative"], size=2),
        name="Pumping (< 0)",
    ))

    if rolling_window:
        rolled = df[TARGET].rolling(rolling_window, center=True).mean()
        fig.add_trace(go.Scattergl(
            x=df["datetime_start"], y=rolled,
            mode="lines",
            name=f"Rolling mean ({rolling_window}h)",
            line=dict(color=COLORS["primary"], width=2),
        ))

    apply_layout(fig, title="Target: Spanish Pumped Storage Production",
                 xaxis_title="Time", yaxis_title="MW", height=height)
    return fig


def plot_feature_group(df, group_name, rolling_window=None, height=500):
    """Plot all features in a predefined group."""
    if group_name not in FEATURE_GROUPS:
        raise ValueError(f"Unknown group '{group_name}'. Choose from: {list(FEATURE_GROUPS)}")
    cols = filter_existing_columns(df, FEATURE_GROUPS[group_name])
    return plot_timeseries(df, cols,
                           title=f"Feature Group: {group_name.title()}",
                           rolling_window=rolling_window, height=height)


def plot_d1_vs_d2(df, base_feature, height=450):
    """Compare day-ahead (d1) vs two-day-ahead (d2) forecasts."""
    d1 = f"{base_feature}_d1"
    d2 = f"{base_feature}_d2"
    cols = filter_existing_columns(df, [d1, d2])
    if len(cols) < 2:
        raise ValueError(f"Need both {d1} and {d2} in the DataFrame.")

    fig = go.Figure()
    fig.add_trace(go.Scattergl(
        x=df["datetime_start"], y=df[d1],
        mode="lines", name="Day-ahead (d1)", line=dict(width=1),
    ))
    fig.add_trace(go.Scattergl(
        x=df["datetime_start"], y=df[d2],
        mode="lines", name="Two-day-ahead (d2)", line=dict(width=1, dash="dash"),
    ))
    apply_layout(fig, title=f"Forecast Comparison: {base_feature}",
                 xaxis_title="Time", yaxis_title="Value", height=height)
    return fig


def plot_train_test_split(train_df, test_df, columns=None, height=450):
    """Overlay train and test periods for selected columns (or target)."""
    columns = columns or [TARGET]
    if isinstance(columns, str):
        columns = [columns]

    fig = go.Figure()
    for col in columns:
        if col in train_df.columns:
            fig.add_trace(go.Scattergl(
                x=train_df["datetime_start"], y=train_df[col],
                mode="lines", name=f"{col} (train)",
                line=dict(width=1, color=COLORS["primary"]),
            ))
        if col in test_df.columns:
            fig.add_trace(go.Scattergl(
                x=test_df["datetime_start"], y=test_df[col],
                mode="lines", name=f"{col} (test)",
                line=dict(width=1, color=COLORS["secondary"]),
            ))

    apply_layout(fig, title="Train / Test Split",
                 xaxis_title="Time", yaxis_title="Value", height=height)
    return fig
