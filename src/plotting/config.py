"""Shared configuration, color palettes, and feature groupings for EDA plots."""

import plotly.graph_objects as go
import plotly.io as pio

# Color palettes
COLORS = {
    "primary": "#1f77b4",
    "secondary": "#ff7f0e",
    "positive": "#2ca02c",   # generation
    "negative": "#d62728",   # pumping
    "neutral": "#7f7f7f",
    "accent": "#9467bd",
}

COUNTRY_COLORS = {
    "es": "#d62728",
    "fr": "#1f77b4",
    "pt": "#2ca02c",
    "de": "#ff7f0e",
}

CATEGORY_COLORS = {
    "weather": "#e377c2",
    "hydro": "#17becf",
    "wind": "#2ca02c",
    "solar": "#ff7f0e",
    "demand": "#d62728",
    "prices": "#9467bd",
    "cross_border": "#8c564b",
    "target": "#1f77b4",
}

# Qualitative palette for general multi-trace plots
PALETTE = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
]

# Feature groupings (matching dataset_description_2025.md)
FEATURE_GROUPS = {
    "weather": [
        "es_temp_f_d1", "es_temp_f_d2",
        "es_precip_f_d1", "es_precip_f_d2",
        "es_wind_speed_f_d1", "es_wind_speed_f_d2",
    ],
    "hydro": [
        "pt_hydro_ror_f_d1", "pt_hydro_ror_f_d2",
        "es_hydro_ror_f_d1", "es_hydro_ror_f_d2",
        "fr_hydro_ror_f_d1", "fr_hydro_ror_f_d2",
        "es_hydro_res_f_d1", "es_hydro_res_f_d2",
        "fr_hydro_res_f_d1", "fr_hydro_res_f_d2",
        "es_hydro_balance_f_d1", "es_hydro_balance_f_d2",
        "es_hydro_inflow_f_d1", "es_hydro_inflow_f_d2",
    ],
    "wind": [
        "es_wind_n", "es_wind_f_d1", "es_wind_f_d2",
        "pt_wind_f_d1", "pt_wind_f_d2",
        "fr_wind_f_d1", "fr_wind_f_d2",
        "de_wind_f_d1", "de_wind_f_d2",
    ],
    "solar": [
        "es_solar_n", "fr_solar_n",
        "es_solar_f_d1", "es_solar_f_d2",
        "pt_solar_f_d1", "pt_solar_f_d2",
        "fr_solar_f_d1", "fr_solar_f_d2",
        "de_solar_f_d1", "de_solar_f_d2",
    ],
    "demand": [
        "es_demand_f_d1", "es_demand_f_d2",
        "pt_demand_f_d1", "pt_demand_f_d2",
        "fr_demand_f_d1", "fr_demand_f_d2",
        "de_demand_f_d1", "de_demand_f_d2",
    ],
    "prices": [
        "es_gas_market_price_d1", "es_gas_market_price_d2",
        "eua_price",
    ],
    "cross_border": [
        "fr_es_ntc_d1", "fr_es_ntc_d2",
        "es_fr_ntc_d1", "es_fr_ntc_d2",
    ],
}

TARGET = "es_total_ps"

# Default layout template

DEFAULT_LAYOUT = dict(
    template="plotly_white",
    font=dict(family="Inter, Arial, sans-serif", size=12),
    title_font_size=16,
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1,
    ),
    margin=dict(l=60, r=30, t=60, b=50),
    hovermode="x unified",
    width=1000,
    height=500,
)


def apply_layout(fig, title=None, xaxis_title=None, yaxis_title=None, **kwargs):
    """Apply the default layout to a plotly figure with optional overrides."""
    layout = {**DEFAULT_LAYOUT, **kwargs}
    if title:
        layout["title_text"] = title
    if xaxis_title:
        layout["xaxis_title"] = xaxis_title
    if yaxis_title:
        layout["yaxis_title"] = yaxis_title
    fig.update_layout(**layout)
    return fig


def filter_existing_columns(df, columns):
    """Return only columns that exist in the dataframe."""
    return [c for c in columns if c in df.columns]
