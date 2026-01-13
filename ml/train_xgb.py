import json
from pathlib import Path
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = BASE_DIR / "outputs"
OUTPUTS_DIR.mkdir(exist_ok=True)

def smape(y_true, y_pred, eps=1e-8):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = (np.abs(y_true) + np.abs(y_pred) + eps) / 2.0
    return float(np.mean(np.abs(y_true - y_pred) / denom) * 100.0)

def make_synthetic_hourly_demand(n_days=90, seed=42):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2026-01-01", periods=n_days * 24, freq="H", tz="UTC")
    df = pd.DataFrame(index=idx)
    hour = df.index.hour
    dow = df.index.dayofweek

    daily = 2500 * np.sin(2 * np.pi * hour / 24) + 8000
    weekend = np.where(dow >= 5, -600, 0)
    noise = rng.normal(0, 250, size=len(df))

    df["demand"] = 30000 + daily + weekend + noise
    return df

def build_features(df):
    out = pd.DataFrame(index=df.index)
    out["hour"] = df.index.hour
    out["dow"] = df.index.dayofweek
    out["month"] = df.index.month
    out["lag_1"] = df["demand"].shift(1)
    out["lag_24"] = df["demand"].shift(24)
    out["roll_24_mean"] = df["demand"].shift(1).rolling(24).mean()
    out["y"] = df["demand"]
    return out.dropna()

def main(horizon=24):
    df = make_synthetic_hourly_demand(n_days=90)
    feat = build_features(df)

    X = feat[["hour", "dow", "month", "lag_1", "lag_24", "roll_24_mean"]]
    y = feat["y"]

    split_point = feat.index.max() - pd.Timedelta(days=7)
    train_mask = feat.index <= split_point
    test_mask = feat.index > split_point

    X_train, y_train = X[train_mask], y[train_mask]
    X_test, y_test = X[test_mask], y[test_mask]

    model = XGBRegressor(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mae = float(mean_absolute_error(y_test, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    smape_val = smape(y_test, y_pred)

    metrics = {
        "model": "xgb",
        "horizon_hours": int(horizon),
        "mae": mae,
        "rmse": rmse,
        "smape": smape_val
    }

    preds = {
        "model": "xgb",
        "horizon_hours": int(horizon),
        "series": [
            {
                "t": t.isoformat().replace("+00:00", "Z"),
                "actual": float(a),
                "predicted": float(p)
            }
            for t, a, p in zip(y_test.index, y_test.values, y_pred)
        ]
    }

    (OUTPUTS_DIR / f"metrics_xgb_{horizon}.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (OUTPUTS_DIR / f"preds_xgb_{horizon}.json").write_text(json.dumps(preds, indent=2), encoding="utf-8")

    print("WROTE:")
    print(OUTPUTS_DIR / f"metrics_xgb_{horizon}.json")
    print(OUTPUTS_DIR / f"preds_xgb_{horizon}.json")

if __name__ == "__main__":
    main(horizon=24)
