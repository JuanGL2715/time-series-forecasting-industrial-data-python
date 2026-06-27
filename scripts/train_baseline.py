from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from time_series_forecasting.data_utils import load_series
from time_series_forecasting.models import evaluate_forecast, forecast_moving_average, forecast_naive


def main() -> None:
    data_path = ROOT / "data" / "sample_series.csv"
    dates, values = load_series(data_path)

    train_values = values[:-3]
    actual_values = values[-3:]

    naive_forecast = forecast_naive(train_values, len(actual_values))
    moving_average_forecast = forecast_moving_average(train_values, len(actual_values), window=3)

    print("Dates:", dates[-3:])
    print("Actual:", actual_values)
    print("Naive forecast:", naive_forecast)
    print("Moving average forecast:", moving_average_forecast)

    naive_metrics = evaluate_forecast(actual_values, naive_forecast)
    moving_average_metrics = evaluate_forecast(actual_values, moving_average_forecast)

    print("Naive metrics:", naive_metrics)
    print("Moving average metrics:", moving_average_metrics)


if __name__ == "__main__":
    main()
