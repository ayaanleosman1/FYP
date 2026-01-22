"""
Forecasting metrics for model evaluation.
"""

import numpy as np
from typing import Union


def smape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-8) -> float:
    """
    Symmetric Mean Absolute Percentage Error.

    SMAPE is a scale-independent metric that treats over and under-predictions
    symmetrically. Values range from 0 (perfect) to 200% (worst).

    Args:
        y_true: Actual values
        y_pred: Predicted values
        eps: Small constant to avoid division by zero

    Returns:
        SMAPE as a percentage (0-200)
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    numerator = np.abs(y_pred - y_true)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2 + eps
    return float(np.mean(numerator / denominator) * 100)


def mape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-8) -> float:
    """
    Mean Absolute Percentage Error.

    Args:
        y_true: Actual values
        y_pred: Predicted values
        eps: Small constant to avoid division by zero

    Returns:
        MAPE as a percentage
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return float(np.mean(np.abs((y_true - y_pred) / (y_true + eps))) * 100)


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Mean Absolute Error.

    Args:
        y_true: Actual values
        y_pred: Predicted values

    Returns:
        MAE in the same units as the target
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Root Mean Squared Error.

    Args:
        y_true: Actual values
        y_pred: Predicted values

    Returns:
        RMSE in the same units as the target
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def compute_all_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Compute all forecast metrics.

    Args:
        y_true: Actual values
        y_pred: Predicted values

    Returns:
        Dictionary with mae, rmse, smape, and mape
    """
    return {
        "mae": round(mae(y_true, y_pred), 2),
        "rmse": round(rmse(y_true, y_pred), 2),
        "smape": round(smape(y_true, y_pred), 2),
        "mape": round(mape(y_true, y_pred), 2),
    }
