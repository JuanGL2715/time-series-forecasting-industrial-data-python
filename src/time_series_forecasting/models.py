from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.arima.model import ARIMA


def evaluate_forecast(actual: list[float], predicted: list[float]) -> dict[str, float]:
    """Compute MAE and RMSE for forecast evaluation."""
    if not actual or not predicted or len(actual) != len(predicted):
        raise ValueError("actual and predicted must be non-empty and have the same length")

    actual_array = np.asarray(actual, dtype=float)
    predicted_array = np.asarray(predicted, dtype=float)
    mae = mean_absolute_error(actual_array, predicted_array)
    rmse = float(np.sqrt(mean_squared_error(actual_array, predicted_array)))

    return {"mae": float(mae), "rmse": rmse}


def fit_arima_model(train_series: pd.Series, order: tuple[int, int, int] = (2, 0, 2)) -> Any:
    """Fit an ARIMA model on the training history."""
    model = ARIMA(
        train_series,
        order=order,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    return model.fit()


def forecast_arima_model(model: Any, horizon: int) -> np.ndarray:
    """Generate a horizon-step ARIMA forecast."""
    return np.asarray(model.forecast(steps=horizon), dtype=float)


def fit_linear_regression_model(X_train: pd.DataFrame, y_train: pd.Series) -> LinearRegression:
    """Fit a linear regression model on engineered lag features."""
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model


def forecast_linear_regression_model(
    model: LinearRegression,
    history: list[float],
    future_dates: list[pd.Timestamp],
    feature_columns: list[str],
) -> list[float]:
    """Create a recursive forecast for future values using lag-based features."""
    predictions: list[float] = []
    history_values = list(history)

    for future_date in future_dates:
        feature_values: dict[str, float | int] = {
            "lag_1": history_values[-1],
            "lag_7": history_values[-7] if len(history_values) >= 7 else history_values[0],
            "lag_14": history_values[-14] if len(history_values) >= 14 else history_values[0],
            "lag_30": history_values[-30] if len(history_values) >= 30 else history_values[0],
            "rolling_mean_3": float(np.mean(history_values[-3:])) if len(history_values) >= 3 else float(history_values[-1]),
            "rolling_mean_7": float(np.mean(history_values[-7:])) if len(history_values) >= 7 else float(history_values[-1]),
            "rolling_mean_30": float(np.mean(history_values[-30:])) if len(history_values) >= 30 else float(history_values[-1]),
            "rolling_std_3": float(np.std(history_values[-3:])) if len(history_values) >= 3 else 0.0,
            "rolling_std_7": float(np.std(history_values[-7:])) if len(history_values) >= 7 else 0.0,
            "rolling_std_30": float(np.std(history_values[-30:])) if len(history_values) >= 30 else 0.0,
            "trend": len(history_values) + 1,
            "day_of_week": future_date.dayofweek,
            "month": future_date.month,
            "is_weekend": int(future_date.dayofweek >= 5),
        }

        feature_frame = pd.DataFrame([feature_values])[feature_columns]
        prediction = float(model.predict(feature_frame)[0])
        predictions.append(prediction)
        history_values.append(prediction)

    return predictions
