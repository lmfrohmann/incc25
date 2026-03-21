"""Correlation and scatter-plot utilities."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from statsmodels.tsa.stattools import acf, pacf

from .config import (
    PALETTE, COLORS, TARGET, FEATURE_GROUPS,
    apply_layout, filter_existing_columns,
)


def plot_correlation_heatmap(df, columns=None, method="pearson", title=None,
                             height=700, width=900):
    """Interactive correlation heatmap."""
    if columns is None:
        num_df = df.select_dtypes("number")
    else:
        columns = filter_existing_columns(df, columns)
        num_df = df[columns]

    corr = num_df.corr(method=method)

    fig = go.Figure(go.Heatmap(
        z=corr.values,
        x=corr.columns.tolist(),
        y=corr.index.tolist(),
        colorscale="RdBu_r",
        zmid=0, zmin=-1, zmax=1,
        text=np.round(corr.values, 2),
        texttemplate="%{text}",
        textfont=dict(size=9),
        hovertemplate="%{x} vs %{y}: %{z:.3f}<extra></extra>",
    ))
    apply_layout(fig,
                 title=title or f"Correlation Matrix ({method.title()})",
                 height=height, width=width)
    fig.update_layout(xaxis_tickangle=-45)
    return fig


def plot_target_correlations(df, top_n=20, method="pearson", height=500):
    """Bar chart of top-N feature correlations with the target."""
    if TARGET not in df.columns:
        raise ValueError(f"Target column '{TARGET}' not in DataFrame.")

    corr = df.select_dtypes("number").corr(method=method)[TARGET].drop(TARGET)
    corr = corr.reindex(corr.abs().sort_values(ascending=False).index)[:top_n]

    colors = [COLORS["positive"] if v >= 0 else COLORS["negative"] for v in corr.values]

    fig = go.Figure(go.Bar(
        x=corr.values, y=corr.index, orientation="h",
        marker_color=colors,
        text=np.round(corr.values, 3),
        textposition="outside",
    ))
    apply_layout(fig,
                 title=f"Top {top_n} Correlations with Target ({method.title()})",
                 xaxis_title="Correlation", yaxis_title="Feature",
                 height=max(height, top_n * 25))
    fig.update_layout(yaxis=dict(autorange="reversed"))
    return fig


def plot_scatter(df, x_col, y_col=None, color_col=None, trendline=True,
                 title=None, height=500):
    """Scatter plot of x vs y (defaults y to target), with optional OLS trendline."""
    y_col = y_col or TARGET
    fig = go.Figure()

    if color_col and color_col in df.columns:
        for i, (name, grp) in enumerate(df.groupby(color_col)):
            fig.add_trace(go.Scattergl(
                x=grp[x_col], y=grp[y_col],
                mode="markers", name=str(name),
                marker=dict(size=3, color=PALETTE[i % len(PALETTE)], opacity=0.5),
            ))
    else:
        fig.add_trace(go.Scattergl(
            x=df[x_col], y=df[y_col],
            mode="markers", name=f"{x_col} vs {y_col}",
            marker=dict(size=3, color=COLORS["primary"], opacity=0.4),
        ))

    if trendline:
        mask = df[[x_col, y_col]].dropna().index
        x_clean = df.loc[mask, x_col]
        y_clean = df.loc[mask, y_col]
        if len(x_clean) > 2:
            coeffs = np.polyfit(x_clean, y_clean, 1)
            x_line = np.linspace(x_clean.min(), x_clean.max(), 100)
            fig.add_trace(go.Scatter(
                x=x_line, y=np.polyval(coeffs, x_line),
                mode="lines", name=f"OLS (slope={coeffs[0]:.2f})",
                line=dict(color=COLORS["secondary"], width=2, dash="dash"),
            ))

    apply_layout(fig, title=title or f"{x_col} vs {y_col}",
                 xaxis_title=x_col, yaxis_title=y_col, height=height)
    return fig


def plot_acf_pacf(series, lags=72, title=None, height=450):
    """ACF/PACF dual plot for a time series."""
    s = pd.Series(series).dropna().values
    nlags = min(lags, len(s) // 2 - 1)
    acf_vals = acf(s, nlags=nlags, fft=True)
    pacf_vals = pacf(s, nlags=min(nlags, 200))  # pacf needs shorter lag

    fig = make_subplots(rows=1, cols=2, subplot_titles=["ACF", "PACF"])

    for col_idx, (vals, label) in enumerate(
        [(acf_vals, "ACF"), (pacf_vals, "PACF")], start=1
    ):
        fig.add_trace(go.Bar(
            x=list(range(len(vals))), y=vals,
            marker_color=COLORS["primary"], opacity=0.7,
            showlegend=False,
        ), row=1, col=col_idx)
        # 95% significance band
        ci = 1.96 / np.sqrt(len(s))
        fig.add_hline(y=ci, line_dash="dash", line_color="red",
                      line_width=1, row=1, col=col_idx)
        fig.add_hline(y=-ci, line_dash="dash", line_color="red",
                      line_width=1, row=1, col=col_idx)
        fig.update_xaxes(title_text="Lag", row=1, col=col_idx)
        fig.update_yaxes(title_text=label, row=1, col=col_idx)

    apply_layout(fig, title=title or "ACF / PACF", height=height, width=1000)
    return fig


def plot_scatter_matrix(df, columns, height=800, width=900):
    """Scatter-plot matrix (SPLOM) for a small set of features."""
    columns = filter_existing_columns(df, columns)
    if len(columns) > 8:
        columns = columns[:8]

    fig = px.scatter_matrix(
        df[columns].dropna(),
        dimensions=columns,
        opacity=0.3,
        height=height, width=width,
    )
    fig.update_traces(diagonal_visible=True, marker=dict(size=2))
    apply_layout(fig, title="Scatter Matrix", height=height, width=width)
    return fig
