from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def plot_time_series(df: pd.DataFrame, target_column: str = "value", save_path: str | Path | None = None) -> None:
    """Plot the observed time series."""
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df["date"], df[target_column], color="#1f77b4", linewidth=1.5)
    ax.set_title("Industrial Energy Consumption Over Time")
    ax.set_xlabel("Date")
    ax.set_ylabel("Value")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=200)
    plt.close(fig)


def plot_actual_vs_predicted(actual: list[float], predicted: list[float], model_name: str, save_path: str | Path | None = None) -> None:
    """Compare actual and predicted values."""
    fig, ax = plt.subplots(figsize=(10, 4))
    dates = list(range(1, len(actual) + 1))
    ax.plot(dates, actual, label="Actual", color="#1f77b4", linewidth=1.5)
    ax.plot(dates, predicted, label=f"{model_name} Forecast", color="#d62728", linestyle="--", linewidth=1.5)
    ax.set_title(f"Actual vs {model_name} Forecast")
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Value")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=200)
    plt.close(fig)


def plot_residuals(actual: list[float], predicted: list[float], save_path: str | Path | None = None) -> None:
    """Plot residuals between actual and predicted values."""
    residuals = [a - p for a, p in zip(actual, predicted)]
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axhline(0, color="black", linewidth=1, linestyle="--")
    ax.plot(residuals, color="#2ca02c", linewidth=1.5)
    ax.set_title("Residual Plot")
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Residual")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=200)
    plt.close(fig)


def plot_trend_analysis(df: pd.DataFrame, save_path: str | Path | None = None) -> None:
    """Visualize moving average trend and original series."""
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df["date"], df["value"], label="Observed", color="#1f77b4", linewidth=1.2)
    rolling_mean = df["value"].rolling(window=7, min_periods=7).mean()
    ax.plot(df["date"], rolling_mean, label="7-Day Rolling Mean", color="#ff7f0e", linewidth=1.5)
    ax.set_title("Trend Analysis")
    ax.set_xlabel("Date")
    ax.set_ylabel("Value")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=200)
    plt.close(fig)
