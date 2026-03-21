"""Distribution and statistical summary plots."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .config import PALETTE, COLORS, TARGET, apply_layout, filter_existing_columns


def plot_histogram(df, columns, nbins=80, title=None, height=450):
    """Overlaid histograms for one or more columns."""
    if isinstance(columns, str):
        columns = [columns]
    columns = filter_existing_columns(df, columns)

    fig = go.Figure()
    for i, col in enumerate(columns):
        fig.add_trace(go.Histogram(
            x=df[col], nbinsx=nbins, name=col,
            marker_color=PALETTE[i % len(PALETTE)], opacity=0.65,
        ))
    fig.update_layout(barmode="overlay")
    apply_layout(fig, title=title or "Distribution", xaxis_title="Value",
                 yaxis_title="Count", height=height)
    return fig


def plot_target_distribution(df, height=450):
    """Histogram of target split by generation / pumping."""
    gen = df.loc[df[TARGET] >= 0, TARGET]
    pump = df.loc[df[TARGET] < 0, TARGET]

    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=gen, nbinsx=60, name="Generation (>= 0)",
        marker_color=COLORS["positive"], opacity=0.7,
    ))
    fig.add_trace(go.Histogram(
        x=pump, nbinsx=60, name="Pumping (< 0)",
        marker_color=COLORS["negative"], opacity=0.7,
    ))
    fig.update_layout(barmode="overlay")
    apply_layout(fig, title="Target Distribution: Generation vs Pumping",
                 xaxis_title="MW", yaxis_title="Count", height=height)
    return fig


def plot_boxplots(df, columns, title=None, height=500):
    """Side-by-side box plots (useful for comparing feature scales)."""
    if isinstance(columns, str):
        columns = [columns]
    columns = filter_existing_columns(df, columns)

    fig = go.Figure()
    for i, col in enumerate(columns):
        fig.add_trace(go.Box(
            y=df[col], name=col,
            marker_color=PALETTE[i % len(PALETTE)],
        ))
    apply_layout(fig, title=title or "Box Plots",
                 yaxis_title="Value", height=height)
    return fig


def plot_violin(df, column, groupby_col=None, title=None, height=450):
    """Violin plot, optionally grouped by a categorical column."""
    fig = go.Figure()
    if groupby_col and groupby_col in df.columns:
        for i, (name, grp) in enumerate(df.groupby(groupby_col)):
            fig.add_trace(go.Violin(
                y=grp[column], name=str(name),
                marker_color=PALETTE[i % len(PALETTE)],
                box_visible=True, meanline_visible=True,
            ))
    else:
        fig.add_trace(go.Violin(
            y=df[column], name=column,
            marker_color=COLORS["primary"],
            box_visible=True, meanline_visible=True,
        ))
    apply_layout(fig, title=title or f"Violin: {column}",
                 yaxis_title=column, height=height)
    return fig


def plot_missing_values(df, title="Missing Values", height=500):
    """Heatmap showing the percentage of missing values per column."""
    pct = df.isnull().mean().sort_values(ascending=False)
    pct = pct[pct > 0]
    if pct.empty:
        print("No missing values found.")
        return None

    fig = go.Figure(go.Bar(
        x=pct.values * 100, y=pct.index, orientation="h",
        marker_color=COLORS["negative"],
    ))
    apply_layout(fig, title=title, xaxis_title="% Missing",
                 yaxis_title="Feature", height=max(height, len(pct) * 22))
    fig.update_layout(yaxis=dict(autorange="reversed"))
    return fig
