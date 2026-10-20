"""Read the monthly model's exported forecasts for the prediction API."""

import json
from pathlib import Path

import numpy as np
import pandas as pd


FORECAST_DIR = Path(__file__).resolve().parents[1] / "artifacts" / "monthly_demand"


def artifact_directory():
    """Successful API training run choose karein; incomplete run serve nahi hoga."""
    pointer = FORECAST_DIR / "active_run.json"
    if not pointer.exists():
        return FORECAST_DIR
    run = json.loads(pointer.read_text(encoding="utf-8"))["run_id"]
    if len(run) != 32 or any(character not in "0123456789abcdef" for character in run):
        raise ValueError("Invalid monthly run identifier")
    return FORECAST_DIR / "runs" / run


def get_monthly_predictions(product_id=None, country_id=None, month=None):
    """Saved monthly forecast filter karein; request par model dobara train nahi hota."""
    directory = artifact_directory()
    frame = pd.read_csv(directory / "forecast_2026.csv", dtype={"Product_ID": "string", "Country_ID": "string"})
    metadata = json.loads((directory / "run_metadata.json").read_text(encoding="utf-8"))
    columns = ["Month", "Product_ID", "Country_ID", "Predicted_Units", "Lower", "Upper"]
    if not set(columns).issubset(frame.columns) or frame[columns].isna().any().any():
        raise ValueError("Monthly forecast file is incomplete; rerun monthly training")
    dates = pd.to_datetime(frame["Month"], errors="raise")
    if not dates.eq(dates.dt.to_period("M").dt.to_timestamp()).all() or not dates.dt.year.eq(2026).all():
        raise ValueError("Expected month-start forecasts for 2026")
    if frame.duplicated(["Month", "Product_ID", "Country_ID"]).any():
        raise ValueError("Duplicate monthly forecast rows")
    numbers = frame[["Predicted_Units", "Lower", "Upper"]].to_numpy(dtype=float)
    if not np.isfinite(numbers).all() or (numbers < 0).any() or (numbers[:, 1] > numbers[:, 0]).any() or (numbers[:, 0] > numbers[:, 2]).any():
        raise ValueError("Invalid forecast values or prediction ranges")
    if product_id is not None:
        frame = frame.loc[frame.Product_ID == product_id]
    if country_id is not None:
        frame = frame.loc[frame.Country_ID == country_id]
    if month is not None:
        frame = frame.loc[pd.to_datetime(frame.Month).dt.month == month]
    frame = frame.sort_values(["Month", "Product_ID", "Country_ID"])
    return {
        "model": metadata["best_model"], "year": 2026,
        "training_period": metadata["final_training_period"],
        "source": "saved_monthly_forecast", "product_id": product_id, "country_id": country_id,
        "count": len(frame), "total_predicted_units": round(float(frame.Predicted_Units.sum()), 2),
        "range_note": metadata["ranges"],
        "forecasts": frame[columns].to_dict("records"),
    }
