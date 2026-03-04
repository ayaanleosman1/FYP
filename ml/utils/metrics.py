import numpy as np


def smape(y_true, y_pred, eps=1e-8):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    numerator = np.abs(y_pred - y_true)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2 + eps
    return float(np.mean(numerator / denominator) * 100)


def mape(y_true, y_pred, eps=1e-8):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return float(np.mean(np.abs((y_true - y_pred) / (y_true + eps))) * 100)


def mae(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def compute_all_metrics(y_true, y_pred):
    return {
        "mae": round(mae(y_true, y_pred), 2),
        "rmse": round(rmse(y_true, y_pred), 2),
        "smape": round(smape(y_true, y_pred), 2),
        "mape": round(mape(y_true, y_pred), 2),
    }
