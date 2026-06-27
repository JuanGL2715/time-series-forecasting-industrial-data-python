from __future__ import annotations

import numpy as np
import pandas as pd


def prepare_features(df: pd.DataFrame, target_column: str = "value") -> pd.DataFrame:
    """Create lag-based and rolling features for time series forecasting."""
    prepared = df.copy().sort_values("date").reset_index(drop=True)
    prepared[target_column] = prepared[target_column].interpolate(method="linear", limit_direction="both")

    for lag in [1, 7, 14, 30]:
        prepared[f"lag_{lag}"] = prepared[target_column].shift(lag)

    for window in [3, 7, 30]:
        prepared[f"rolling_mean_{window}"] = prepared[target_column].shift(1).rolling(window=window, min_periods=window).mean()
        prepared[f"rolling_std_{window}"] = prepared[target_column].shift(1).rolling(window=window, min_periods=window).std()

    prepared["trend"] = np.arange(1, len(prepared) + 1)
    prepared["day_of_week"] = prepared["date"].dt.dayofweek
    prepared["month"] = prepared["date"].dt.month
    prepared["is_weekend"] = (prepared["day_of_week"] >= 5).astype(int)

    prepared = prepared.dropna().reset_index(drop=True)
    return prepared
