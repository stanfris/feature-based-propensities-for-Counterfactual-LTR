from .config import ExperimentConfig
from .data_loader import (
    aggregate_mean_std,
    load_baselines_from_folder,
    load_long_metrics,
    metric_only_baselines,
)

_PLOTTING_EXPORTS = {
    "plot_dm_dr_ips_naiveho_frequency_based",
    "plot_ips_dm_dr_stacked_propensity_grid",
    "plot_grid",
    "plot_policy_models_train_histograms_ps1p0",
    "plot_temperature_analysis",
}

__all__ = [
    "ExperimentConfig",
    "load_long_metrics",
    "aggregate_mean_std",
    "metric_only_baselines",
    "load_baselines_from_folder",
    "plot_grid",
    "plot_dm_dr_ips_naiveho_frequency_based",
    "plot_ips_dm_dr_stacked_propensity_grid",
    "plot_policy_models_train_histograms_ps1p0",
    "plot_temperature_analysis",
]


def __getattr__(name: str):
    if name in _PLOTTING_EXPORTS:
        from . import plotting as _plotting

        return getattr(_plotting, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
