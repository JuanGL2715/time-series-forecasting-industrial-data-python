from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import pandas as pd

from time_series_forecasting.data_loader import generate_sample_data, load_dataset, save_predictions
from time_series_forecasting.models import (
    evaluate_forecast,
    fit_arima_model,
    fit_linear_regression_model,
    forecast_arima_model,
    forecast_linear_regression_model,
)
from time_series_forecasting.preprocessing import prepare_features
from time_series_forecasting.visualization import (
    plot_actual_vs_predicted,
    plot_residuals,
    plot_time_series,
    plot_trend_analysis,
)


def main() -> None:
    data_path = ROOT / "data" / "industrial_energy.csv"
    output_dir = ROOT / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    if not data_path.exists():
        generate_sample_data(data_path)

    df = load_dataset(data_path)
    prepared_df = prepare_features(df)

    horizon = 30
    train_df = prepared_df.iloc[:-horizon]
    test_df = prepared_df.iloc[-horizon:]

    feature_columns = [column for column in train_df.columns if column not in {"date", "value"}]
    X_train = train_df[feature_columns]
    y_train = train_df["value"]
    X_test = test_df[feature_columns]
    y_test = test_df["value"]

    arima_model = fit_arima_model(train_df["value"], order=(2, 0, 2))
    arima_forecast = forecast_arima_model(arima_model, horizon)

    linear_model = fit_linear_regression_model(X_train, y_train)
    linear_forecast = forecast_linear_regression_model(
        model=linear_model,
        history=list(train_df["value"].tolist()),
        future_dates=list(test_df["date"].tolist()),
        feature_columns=feature_columns,
    )

    arima_metrics = evaluate_forecast(y_test.tolist(), arima_forecast.tolist())
    linear_metrics = evaluate_forecast(y_test.tolist(), linear_forecast)

    plot_time_series(df, save_path=output_dir / "time_series.png")
    plot_trend_analysis(df, save_path=output_dir / "trend_analysis.png")
    plot_actual_vs_predicted(y_test.tolist(), arima_forecast.tolist(), "ARIMA", save_path=output_dir / "arima_vs_actual.png")
    plot_actual_vs_predicted(y_test.tolist(), linear_forecast, "Linear Regression", save_path=output_dir / "linear_regression_vs_actual.png")
    plot_residuals(y_test.tolist(), linear_forecast, save_path=output_dir / "residuals.png")

    predictions_df = pd.DataFrame(
        {
            "date": test_df["date"].tolist(),
            "actual": y_test.tolist(),
            "arima_forecast": arima_forecast.tolist(),
            "linear_regression_forecast": linear_forecast,
        }
    )
    save_predictions(output_dir / "predictions.csv", predictions_df)

    print("Forecasting workflow completed.")
    print("ARIMA metrics:", arima_metrics)
    print("Linear regression metrics:", linear_metrics)


if __name__ == "__main__":
    main()
