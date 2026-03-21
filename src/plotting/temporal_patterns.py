"""Temporal pattern analysis — hourly, daily, weekly, monthly profiles."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from .config import PALETTE, COLORS, TARGET, apply_layout


def _ensure_time_cols(df):
    """Add hour/weekday/month columns if missing, returning a copy."""
    df = df.copy()
    dt = df["datetime_start"]
    if "hour" not in df.columns:
        df["hour"] = dt.dt.hour
    if "weekday" not in df.columns:
        df["weekday"] = dt.dt.day_name()
    if "weekday_num" not in df.columns:
        df["weekday_num"] = dt.dt.dayofweek  # Mon=0
    if "month" not in df.columns:
        df["month"] = dt.dt.month
    if "month_name" not in df.columns:
        df["month_name"] = dt.dt.month_name()
    return df


def plot_hourly_profile(df, column=None, groupby=None, agg="mean", height=450):
    """Average hourly profile. Optionally group by month/weekday."""
    column = column or TARGET
    df = _ensure_time_cols(df)

    fig = go.Figure()

    if groupby == "month":
        for m in sorted(df["month"].unique()):
            sub = df[df["month"] == m]
            profile = sub.groupby("hour")[column].agg(agg)
            fig.add_trace(go.Scatter(
                x=profile.index, y=profile.values,
                mode="lines+markers", name=f"Month {m}",
                line=dict(color=PALETTE[(m - 1) % len(PALETTE)]),
            ))
    elif groupby == "weekday":
        day_order = ["Monday", "Tuesday", "Wednesday", "Thursday",
                     "Friday", "Saturday", "Sunday"]
        for i, day in enumerate(day_order):
            sub = df[df["weekday"] == day]
            if sub.empty:
                continue
            profile = sub.groupby("hour")[column].agg(agg)
            fig.add_trace(go.Scatter(
                x=profile.index, y=profile.values,
                mode="lines+markers", name=day,
                line=dict(color=PALETTE[i % len(PALETTE)]),
            ))
    else:
        profile = df.groupby("hour")[column].agg(agg)
        fig.add_trace(go.Scatter(
            x=profile.index, y=profile.values,
            mode="lines+markers", name=f"{agg}({column})",
            line=dict(color=COLORS["primary"], width=2),
            fill="tozeroy", fillcolor="rgba(31,119,180,0.15)",
        ))

    group_label = f" by {groupby}" if groupby else ""
    apply_layout(fig,
                 title=f"Hourly Profile: {column} ({agg}){group_label}",
                 xaxis_title="Hour of Day", yaxis_title=column, height=height)
    fig.update_xaxes(dtick=1)
    return fig


def plot_weekly_profile(df, column=None, agg="mean", height=450):
    """Average profile across days of the week."""
    column = column or TARGET
    df = _ensure_time_cols(df)

    day_order = ["Monday", "Tuesday", "Wednesday", "Thursday",
                 "Friday", "Saturday", "Sunday"]
    profile = df.groupby("weekday").agg(
        val=(column, agg),
        order=("weekday_num", "first"),
    ).sort_values("order")

    colors = [COLORS["primary"]] * 5 + [COLORS["secondary"]] * 2

    fig = go.Figure(go.Bar(
        x=profile.index, y=profile["val"],
        marker_color=colors[:len(profile)],
    ))
    apply_layout(fig,
                 title=f"Weekly Profile: {column} ({agg})",
                 xaxis_title="Day", yaxis_title=column, height=height)
    return fig


def plot_monthly_profile(df, column=None, show_box=True, height=500):
    """Monthly box plots or bar chart."""
    column = column or TARGET
    df = _ensure_time_cols(df)

    fig = go.Figure()
    if show_box:
        for m in sorted(df["month"].unique()):
            sub = df[df["month"] == m]
            fig.add_trace(go.Box(
                y=sub[column], name=sub["month_name"].iloc[0],
                marker_color=PALETTE[(m - 1) % len(PALETTE)],
            ))
    else:
        profile = df.groupby("month")[column].mean()
        fig.add_trace(go.Bar(
            x=[pd.Timestamp(2024, m, 1).month_name() for m in profile.index],
            y=profile.values,
            marker_color=[PALETTE[(m - 1) % len(PALETTE)] for m in profile.index],
        ))

    apply_layout(fig,
                 title=f"Monthly Profile: {column}",
                 xaxis_title="Month", yaxis_title=column, height=height)
    return fig


def plot_hourly_heatmap(df, column=None, agg="mean", height=450):
    """Heatmap: hour of day vs day of week (or month)."""
    column = column or TARGET
    df = _ensure_time_cols(df)

    pivot = df.pivot_table(values=column, index="weekday_num",
                           columns="hour", aggfunc=agg)
    day_labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

    fig = go.Figure(go.Heatmap(
        z=pivot.values,
        x=pivot.columns.tolist(),
        y=[day_labels[i] for i in pivot.index],
        colorscale="RdBu_r",
        zmid=0,
        hovertemplate="Hour %{x}, %{y}: %{z:.0f} MW<extra></extra>",
    ))
    apply_layout(fig,
                 title=f"Heatmap: {column} ({agg}) — Hour vs Weekday",
                 xaxis_title="Hour of Day", yaxis_title="Day of Week",
                 height=height)
    fig.update_xaxes(dtick=1)
    return fig


def plot_monthly_hourly_heatmap(df, column=None, agg="mean", height=500):
    """Heatmap: hour of day vs month."""
    column = column or TARGET
    df = _ensure_time_cols(df)

    pivot = df.pivot_table(values=column, index="month",
                           columns="hour", aggfunc=agg)
    month_labels = [pd.Timestamp(2024, m, 1).strftime("%b") for m in pivot.index]

    fig = go.Figure(go.Heatmap(
        z=pivot.values,
        x=pivot.columns.tolist(),
        y=month_labels,
        colorscale="RdBu_r",
        zmid=0,
        hovertemplate="Hour %{x}, %{y}: %{z:.0f}<extra></extra>",
    ))
    apply_layout(fig,
                 title=f"Heatmap: {column} ({agg}) — Hour vs Month",
                 xaxis_title="Hour of Day", yaxis_title="Month",
                 height=height)
    fig.update_xaxes(dtick=1)
    return fig


def plot_rolling_feature_importance(importances_df, top_n=10, height=500):
    """Heatmap of feature importance over rolling windows."""
    avg_imp = importances_df.mean().nlargest(top_n)
    sub = importances_df[avg_imp.index]

    fig = go.Figure(go.Heatmap(
        z=sub.values,
        x=sub.columns.tolist(),
        y=[str(w) for w in sub.index],
        colorscale="YlOrRd",
        hovertemplate="Feature: %{x}<br>Window: %{y}<br>Importance: %{z:.4f}<extra></extra>",
    ))
    apply_layout(fig,
                 title=f"Rolling Feature Importance — Top {top_n}",
                 xaxis_title="Feature", yaxis_title="Window",
                 height=max(height, len(sub) * 25), width=1000)
    fig.update_xaxes(tickangle=-45)
    return fig
