"""
Multi-timeframe forecasting utilities.

This package provides shared utilities for training demand forecasting models
at various granularities (hourly, daily, weekly, monthly, yearly).
"""

from .granularity import Granularity, GRANULARITY_CONFIG
from .metrics import smape, mape, compute_all_metrics
from .io import save_outputs, load_outputs, get_output_path, format_predictions_for_api, list_available_models
from .data import (
    make_synthetic_hourly_demand,
    resample_to_granularity,
    get_data_for_granularity,
    load_real_demand_data,
    get_hourly_data,
)
from .features import build_features, get_available_features, get_feature_columns

__all__ = [
    # Granularity
    "Granularity",
    "GRANULARITY_CONFIG",
    # Metrics
    "smape",
    "mape",
    "compute_all_metrics",
    # IO
    "save_outputs",
    "load_outputs",
    "get_output_path",
    "format_predictions_for_api",
    "list_available_models",
    # Data
    "make_synthetic_hourly_demand",
    "resample_to_granularity",
    "get_data_for_granularity",
    "load_real_demand_data",
    "get_hourly_data",
    # Features
    "build_features",
    "get_available_features",
    "get_feature_columns",
]
