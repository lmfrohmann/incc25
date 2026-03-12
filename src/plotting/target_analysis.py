"""Target-specific and feature-vs-target analysis for pumped storage EDA."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .config import PALETTE, COLORS, TARGET, apply_layout, filter_existing_columns


def plot_regime_analysis(df, height=500):
    """Analyse pumping vs generation regimes: fraction of time, intensity."""
    dt = df["datetime_start"]
    df = df.copy()
    df["hour"] = dt.dt.hour

    gen = df[TARGET] > 0
    pump = df[TARGET] < 0
    idle = df[TARGET] == 0

    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=["Regime Frequency by Hour",
                                        "Mean Intensity by Hour"])

    for mask, name, color in [
        (gen, "Generation", COLORS["positive"]),
        (pump, "Pumping", COLORS["negative"]),
        (idle, "Idle", COLORS["neutral"]),
    ]:
        freq = df[mask].groupby("hour").size() / df.groupby("hour").size()
        freq = freq.fillna(0)
        fig.add_trace(go.Bar(
            x=freq.index, y=freq.values * 100, name=name,
            marker_color=color, legendgroup=name,
        ), row=1, col=1)

    gen_mean = df[gen].groupby("hour")[TARGET].mean()
    pump_mean = df[pump].groupby("hour")[TARGET].mean()
    fig.add_trace(go.Scatter(
        x=gen_mean.index, y=gen_mean.values,
        mode="lines+markers", name="Avg Generation",
        line=dict(color=COLORS["positive"], width=2),
        legendgroup="Generation", showlegend=False,
    ), row=1, col=2)
    fig.add_trace(go.Scatter(
        x=pump_mean.index, y=pump_mean.values,
        mode="lines+markers", name="Avg Pumping",
        line=dict(color=COLORS["negative"], width=2),
        legendgroup="Pumping", showlegend=False,
    ), row=1, col=2)

    fig.update_layout(barmode="stack")
    apply_layout(fig, title="Pumped Storage Operating Regimes",
                 height=height, width=1100)
    fig.update_xaxes(title_text="Hour", dtick=1, row=1, col=1)
    fig.update_xaxes(title_text="Hour", dtick=1, row=1, col=2)
    fig.update_yaxes(title_text="% of hours", row=1, col=1)
    fig.update_yaxes(title_text="MW", row=1, col=2)
    return fig


def plot_feature_vs_target(df, features, ncols=3, height_per_row=350):
    """Grid of scatter plots: each feature vs target."""
    if isinstance(features, str):
        features = [features]
    features = filter_existing_columns(df, features)
    nrows = int(np.ceil(len(features) / ncols))

    fig = make_subplots(rows=nrows, cols=ncols,
                        subplot_titles=features,
                        horizontal_spacing=0.06, vertical_spacing=0.08)

    for idx, feat in enumerate(features):
        r = idx // ncols + 1
        c = idx % ncols + 1
        fig.add_trace(go.Scattergl(
            x=df[feat], y=df[TARGET],
            mode="markers",
            marker=dict(size=2, color=COLORS["primary"], opacity=0.3),
            showlegend=False,
        ), row=r, col=c)
        fig.update_xaxes(title_text=feat, title_font_size=9, row=r, col=c)
        fig.update_yaxes(title_text="MW" if c == 1 else "", row=r, col=c)

    apply_layout(fig, title="Features vs Target (es_total_ps)",
                 height=nrows * height_per_row,
                 width=1100)
    return fig


def plot_residual_demand_profile(df, height=500):
    """Show how residual demand (demand - wind - solar) relates to target."""
    df = df.copy()
    if "es_residualdemand_f_d1" not in df.columns:
        if all(c in df.columns for c in ["es_demand_f_d1", "es_wind_f_d1", "es_solar_f_d1"]):
            df["es_residualdemand_f_d1"] = (
                df["es_demand_f_d1"] - df["es_wind_f_d1"] - df["es_solar_f_d1"]
            )
        else:
            raise ValueError("Need demand, wind, and solar d1 columns for residual demand.")

    fig = go.Figure()

    # Color by regime
    gen_mask = df[TARGET] >= 0
    fig.add_trace(go.Scattergl(
        x=df.loc[gen_mask, "es_residualdemand_f_d1"],
        y=df.loc[gen_mask, TARGET],
        mode="markers", name="Generation",
        marker=dict(size=3, color=COLORS["positive"], opacity=0.4),
    ))
    fig.add_trace(go.Scattergl(
        x=df.loc[~gen_mask, "es_residualdemand_f_d1"],
        y=df.loc[~gen_mask, TARGET],
        mode="markers", name="Pumping",
        marker=dict(size=3, color=COLORS["negative"], opacity=0.4),
    ))

    apply_layout(fig, title="Residual Demand vs Pumped Storage Production",
                 xaxis_title="Residual Demand (MW)",
                 yaxis_title="es_total_ps (MW)", height=height)
    return fig


def plot_error_by_regime(errors_df, regime_col, title=None, height=500):
    """Box/violin of errors grouped by a regime column."""
    fig = go.Figure()
    groups = sorted(errors_df[regime_col].dropna().unique())
    for i, grp in enumerate(groups):
        sub = errors_df[errors_df[regime_col] == grp]
        fig.add_trace(go.Violin(
            y=sub["error"], name=str(grp),
            box_visible=True, meanline_visible=True,
            marker_color=PALETTE[i % len(PALETTE)],
        ))
    apply_layout(fig,
                 title=title or f"Error Distribution by {regime_col}",
                 xaxis_title=regime_col, yaxis_title="Error (MW)",
                 height=height)
    return fig


def plot_rolling_rmse(df, actual_col, pred_cols, window=168, height=450):
    """Rolling RMSE time series for one or more prediction columns."""
    if isinstance(pred_cols, str):
        pred_cols = [pred_cols]

    fig = go.Figure()
    for i, col in enumerate(pred_cols):
        sq_err = (df[col] - df[actual_col]) ** 2
        rolling = np.sqrt(sq_err.rolling(window, center=True).mean())
        fig.add_trace(go.Scatter(
            x=df["datetime_start"], y=rolling,
            mode="lines", name=col,
            line=dict(color=PALETTE[i % len(PALETTE)], width=2),
        ))

    apply_layout(fig,
                 title=f"Rolling RMSE ({window}h window)",
                 xaxis_title="Date", yaxis_title="RMSE (MW)",
                 height=height)
    return fig


def plot_conditional_scatter(df, x_col, y_col, condition_col, bins=4,
                             title=None, height=500):
    """Scatter colored by quantile of a conditioning variable."""
    tmp = df[[x_col, y_col, condition_col]].dropna().copy()
    tmp["_bin"] = pd.qcut(tmp[condition_col], q=bins, duplicates="drop")
    bin_labels = sorted(tmp["_bin"].unique())

    fig = go.Figure()
    for i, bl in enumerate(bin_labels):
        sub = tmp[tmp["_bin"] == bl]
        fig.add_trace(go.Scattergl(
            x=sub[x_col], y=sub[y_col],
            mode="markers", name=str(bl),
            marker=dict(size=3, color=PALETTE[i % len(PALETTE)], opacity=0.5),
        ))

    apply_layout(fig,
                 title=title or f"{x_col} vs {y_col} (colored by {condition_col})",
                 xaxis_title=x_col, yaxis_title=y_col, height=height)
    return fig


def plot_unavailability_timeline(prod_unavail, cons_unavail, height=450):
    """Timeline of production and consumption unavailability by unit."""
    fig = go.Figure()

    for kind_df, kind_name, color in [
        (prod_unavail, "Production", COLORS["negative"]),
        (cons_unavail, "Consumption", COLORS["primary"]),
    ]:
        if kind_df is None or kind_df.empty:
            continue
        udf = kind_df.copy()
        udf["datetime_start"] = pd.to_datetime(udf["datetime_start"], utc=True)
        agg = udf.groupby("datetime_start")["unavailable"].sum().reset_index()
        fig.add_trace(go.Scattergl(
            x=agg["datetime_start"], y=agg["unavailable"],
            mode="lines", name=f"{kind_name} Unavail.",
            line=dict(color=color, width=1),
            fill="tozeroy",
        ))

    apply_layout(fig, title="Aggregated Capacity Unavailability",
                 xaxis_title="Time", yaxis_title="Unavailable MW", height=height)
    return fig


def plot_plant_capacity(metadata, height=400):
    """Bar chart comparing plant capacities (production and pump load)."""
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=metadata["unit_name"],
        y=metadata["capacity_per_unit"] * metadata["units"],
        name="Total Production Capacity (MW)",
        marker_color=COLORS["positive"],
    ))
    fig.add_trace(go.Bar(
        x=metadata["unit_name"],
        y=metadata["pump_load_per_unit"] * metadata["pump_units"],
        name="Total Pump Load (MW)",
        marker_color=COLORS["negative"],
    ))
    fig.update_layout(barmode="group")
    apply_layout(fig, title="Plant Capacities: Production vs Pump Load",
                 xaxis_title="Plant", yaxis_title="MW", height=height)
    fig.update_xaxes(tickangle=-45)
    return fig
