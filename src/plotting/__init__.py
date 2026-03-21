"""Plotly-based EDA plotting utilities for the INCC pumped storage project."""

from .config import (
    COLORS, COUNTRY_COLORS, CATEGORY_COLORS, PALETTE,
    FEATURE_GROUPS, TARGET,
    apply_layout, filter_existing_columns,
)
from .timeseries import (
    plot_timeseries,
    plot_target,
    plot_feature_group,
    plot_d1_vs_d2,
    plot_train_test_split,
)
from .distributions import (
    plot_histogram,
    plot_target_distribution,
    plot_boxplots,
    plot_violin,
    plot_missing_values,
)

try:
    from .correlations import (
        plot_correlation_heatmap,
        plot_target_correlations,
        plot_scatter,
        plot_scatter_matrix,
        plot_acf_pacf,
    )
except ImportError as e:
    import warnings
    warnings.warn(
        f"Could not import correlations module (statsmodels may not be installed): {e}. "
        "Install statsmodels to enable plot_target_correlations, plot_scatter, plot_acf_pacf, etc."
    )

from .temporal_patterns import (
    plot_hourly_profile,
    plot_weekly_profile,
    plot_monthly_profile,
    plot_hourly_heatmap,
    plot_monthly_hourly_heatmap,
    plot_rolling_feature_importance,
)
from .target_analysis import (
    plot_regime_analysis,
    plot_feature_vs_target,
    plot_residual_demand_profile,
    plot_unavailability_timeline,
    plot_plant_capacity,
    plot_error_by_regime,
    plot_rolling_rmse,
    plot_conditional_scatter,
)
