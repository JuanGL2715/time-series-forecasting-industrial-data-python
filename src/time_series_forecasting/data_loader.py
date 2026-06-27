from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def generate_sample_data(output_path: str | Path, rows: int = 400, seed: int = 42) -> pd.DataFrame:
    """Generate a realistic synthetic industrial energy-consumption dataset."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start="2020-01-01", periods=rows, freq="D")

    trend = np.linspace(100, 220, rows)
    seasonality = 12 * np.sin(2 * np.pi * np.arange(rows) / 7) + 6 * np.cos(2 * np.pi * np.arange(rows) / 30)
    noise = rng.normal(0, 5, rows)
    values = trend + seasonality + noise

    df = pd.DataFrame({"date": dates, "value": values})
    df.loc[rng.choice(df.index, size=max(3, rows // 80), replace=False), "value"] = np.nan

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return df


def load_dataset(path: str | Path) -> pd.DataFrame:
    """Load the forecasting dataset from a CSV file."""
    path = Path(path)
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    return df


def save_predictions(path: str | Path, predictions: pd.DataFrame) -> None:
    """Persist prediction outputs to disk."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output_path, index=False)
