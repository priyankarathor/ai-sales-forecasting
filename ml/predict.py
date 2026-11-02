from pathlib import Path
import math

import joblib
import pandas as pd

from ml.data_loader import load_sales
from ml.artifacts import model_path
from ml.preprocessing import prepare_daily_sales, prepare_model_series

MODEL_PATH = (
    Path(__file__).resolve().parents[1]
    / "artifacts"
    / "sales_model.joblib"
)


def predict_next_day(*, include_details: bool = False) -> dict:
    """Compatibility entry point: predict one observation at the saved cadence."""
    return predict_periods(periods=1, include_details=include_details)


def predict_periods(
    *, periods: int = 4, calendar_month: bool = False,
    include_details: bool = False,
    product_id: str | None = None, region: str | None = None,
    model_type: str = "linear_regression", days: int | None = None,
) -> dict:
    if isinstance(periods, bool) or not isinstance(periods, int) or not 1 <= periods <= 90:
        raise ValueError("Forecast periods must be an integer between 1 and 90")
    if days is not None and (isinstance(days, bool) or not isinstance(days, int) or not 1 <= days <= 90):
        raise ValueError("Forecast days must be an integer between 1 and 90")
    if days is not None and calendar_month:
        raise ValueError("Choose days or calendar_month, not both")
    if (product_id is None) != (region is None):
        raise ValueError("Provide both product_id and region")
    if model_type not in {"linear_regression", "random_forest", "auto"}:
        raise ValueError("Unsupported model type")
    selected_path = MODEL_PATH if product_id is None else model_path(product_id, region, model_type)
    if not selected_path.exists():
        raise FileNotFoundError(
            "Saved model not found. Complete training first."
        )

    artifact = joblib.load(selected_path)
    if product_id is not None and (
        artifact["product_id"] != product_id or artifact["region"] != region
        or artifact.get("requested_model_type", artifact.get("model_type")) != model_type
    ):
        raise ValueError("Saved model parameters do not match the request; retrain first")

    model = artifact["model"]
    product_id = artifact["product_id"]
    region = artifact["region"]

    # Use the product-region combination stored with the model.
    sales = load_sales(
        product_id=product_id,
        region=region,
    )

    daily = prepare_daily_sales(
        sales,
        product_id=product_id,
        region=region,
    ).sort_values("date")

    daily, frequency = prepare_model_series(daily, artifact.get("frequency", "D"))
    period = "weeks" if frequency == "7D" else "days"

    if len(daily) < 7:
        raise ValueError(f"At least seven {period} of history are required")

    recent = daily.tail(7)

    if recent["units_sold"].isna().any():
        raise ValueError(
            f"Resolve missing sales in the latest seven {period} first"
        )

    last_date = daily["date"].iloc[-1]
    trained_through = pd.Timestamp(artifact["trained_through"])

    if last_date < trained_through:
        raise ValueError(
            "Available history ends before the model's training date"
        )

    horizon = 7 if frequency == "7D" else 1
    if days is not None:
        if days % horizon:
            raise ValueError("Weekly models require days to be a multiple of 7; daily values cannot be inferred from weekly observations")
        periods = days // horizon
    month_start = last_date + pd.offsets.MonthBegin(1)
    month_end = month_start + pd.offsets.MonthEnd(0)
    steps = math.ceil((month_end - last_date).days / horizon) if calendar_month else periods
    values = recent["units_sold"].astype(float).tolist()
    forecasts = []
    first_features = None
    for step in range(1, steps + 1):
        next_date = last_date + pd.Timedelta(days=horizon * step)
        # Later periods use earlier predictions, never future actual sales.
        features = pd.DataFrame([{
            "lag_1": values[-1],
            "lag_7": values[-7],
            "rolling_mean_7": sum(values[-7:]) / 7,
            "day_of_week": next_date.dayofweek,
        }])[artifact["feature_columns"]]
        prediction = float(model.predict(features)[0])
        if artifact.get("clip_nonnegative") and math.isfinite(prediction):
            prediction = max(0.0, prediction)
        if not math.isfinite(prediction) or prediction < 0:
            raise RuntimeError("Model returned an invalid sales prediction")
        values.append(prediction)
        if calendar_month and not month_start <= next_date <= month_end:
            continue
        if first_features is None:
            first_features = features
        forecasts.append({
            "date": next_date.date().isoformat(),
            "predicted_units": round(prediction, 2),
        })

    result = {
        "product_id": product_id,
        "region": region,
        "last_sales_date": last_date.date().isoformat(),
        "forecast_date": forecasts[0]["date"],
        "predicted_units": forecasts[0]["predicted_units"],
        "forecast_end_date": forecasts[-1]["date"],
        "forecasts": forecasts,
        "periods": len(forecasts),
        "total_predicted_units": round(sum(row["predicted_units"] for row in forecasts), 2),
        "window": "next_calendar_month" if calendar_month else "next_periods",
        "calendar_month": month_start.strftime("%Y-%m") if calendar_month else None,
        "aggregation": "sum_of_forecast_observations",
        "interpretation": (
            "Weekly dated observations; not assumed to be whole-week totals. "
            "Month grouping uses observation dates, not prorated calendar-month demand."
            if frequency == "7D" else
            "Daily dated observations; totals represent daily demand only if source values are daily totals."
        ),
        "validation_scope": "Saved MAE evaluates one period ahead only; multi-period forecasts are recursive and not separately validated.",
        "model_trained_through": trained_through.date().isoformat(),
        "frequency": frequency,
        "forecast_horizon_days": horizon,
        "requested_days": days,
        "forecast_window_days": int((pd.Timestamp(forecasts[-1]["date"]) - last_date).days),
        "model_type": artifact.get("model_type", "random_forest"),
    }

    if include_details:
        result["history"] = [
            {
                "date": row["date"].date().isoformat(),
                "units_sold": (
                    None if pd.isna(row["units_sold"])
                    else float(row["units_sold"])
                ),
            }
            for _, row in daily.tail(90).iterrows()
        ]
        result["evaluation"] = artifact.get("evaluation", {})
        result["features"] = first_features.iloc[0].to_dict()

    return result


if __name__ == "__main__":
    import json

    result = predict_next_day()
    print(json.dumps(result, indent=2))
