"""React-facing monthly ML inspection, inference and training services."""

import json
import os
from threading import Lock
from uuid import uuid4

import joblib
import numpy as np
import pandas as pd

from ml import monthly_predictions as stored
from ml.monthly_demand_forecast import FEATURES, KEYS, calendar_features, future_price_assumptions


TRAINING_LOCK = Lock()


def load_bundle():
    """Trusted local model load karein; client se model file accept nahi hoti."""
    return joblib.load(stored.artifact_directory() / "best_model.joblib")


def model_details():
    """React ko model settings, data coverage, IDs, metrics aur importance dein."""
    directory = stored.artifact_directory()
    bundle = joblib.load(directory / "best_model.joblib")
    history = bundle["history"]
    tree = bundle.get("model")
    settings, importance = {}, []
    if tree is not None:
        estimator = tree.named_steps["model"]
        params = estimator.get_params()
        settings = {key: params[key] for key in (
            "n_estimators", "max_depth", "learning_rate", "subsample",
            "colsample_bytree", "min_samples_leaf", "random_state",
        ) if key in params}
        names = tree.named_steps["encode"].get_feature_names_out()
        importance = sorted([{"feature": str(name), "importance": float(value)}
                             for name, value in zip(names, estimator.feature_importances_)],
                            key=lambda row: row["importance"], reverse=True)
    else:
        from prophet.serialize import model_from_json
        example = model_from_json(next(iter(bundle["prophet_models_json"].values())))
        settings = {"growth": example.growth, "n_changepoints": example.n_changepoints,
                    "changepoint_prior_scale": example.changepoint_prior_scale,
                    "seasonality_prior_scale": example.seasonality_prior_scale,
                    "annual_fourier_order": 3, "separate_model_per_product_country": True}
    metadata = json.loads((directory / "run_metadata.json").read_text(encoding="utf-8"))
    return {"model": bundle["model_type"], "target": "Units_Sold",
            "training_rows": len(history), "usable_tree_training_rows": len(history) - history.groupby(KEYS).ngroups * 12,
            "history_start": history.Month.min().date().isoformat(),
            "trained_through": bundle["trained_through"],
            "products": sorted(history.Product_ID.unique().tolist()),
            "countries": sorted(history.Country_ID.unique().tolist()),
            "features": bundle["feature_columns"] if tree is not None else ["annual_seasonality", "trend", "Selling_Price", "Discount", "Is_Festival_Month"],
            "parameters": settings, "metrics": bundle["test_metrics"],
            "feature_importance": importance, "assumptions": metadata,
            "evaluation": "2023-2024 train; 2025 fixed-origin recursive selection holdout; winner refit through Dec 2025"}


def sales_history(product_id, country_id):
    """Model ne jo history dekhi thi wahi React charts ke liye return karein."""
    frame = load_bundle()["history"]
    frame = frame.loc[(frame.Product_ID == product_id) & (frame.Country_ID == country_id)].copy()
    if frame.empty:
        raise LookupError("Unknown product/country combination")
    frame["Month"] = frame.Month.dt.strftime("%Y-%m-%d")
    return {"product_id": product_id, "country_id": country_id,
            "count": len(frame), "history": frame.sort_values("Month").to_dict("records")}


def test_predictions(model=None, product_id=None, country_id=None):
    """2025 actual/predicted rows model-comparison graph ke liye dein."""
    directory = stored.artifact_directory()
    name = model or joblib.load(directory / "best_model.joblib")["model_type"]
    if name not in ("Prophet", "XGBoost", "Random Forest"):
        raise ValueError("Invalid model")
    filename = f"test_predictions_{name.lower().replace(' ', '_')}.csv"
    frame = pd.read_csv(directory / filename, dtype={key: "string" for key in KEYS})
    for key, value in (("Product_ID", product_id), ("Country_ID", country_id)):
        if value is not None:
            frame = frame.loc[frame[key] == value]
    if frame.empty:
        raise LookupError("No test predictions for these filters")
    return {"model": name, "count": len(frame), "rows": frame.to_dict("records")}


def live_prediction(request):
    """Saved model se real inference karein; price/discount what-if support karein."""
    bundle = load_bundle()
    history = bundle["history"]
    selected = history.loc[(history.Product_ID == request.product_id) & (history.Country_ID == request.country_id)].sort_values("Month")
    if selected.empty:
        raise LookupError("Unknown product/country combination")
    average = future_price_assumptions(selected).iloc[0]
    price = request.selling_price if request.selling_price is not None else float(average.Selling_Price)
    discount = request.discount if request.discount is not None else float(average.Discount)
    months = pd.date_range("2026-01-01", periods=request.month, freq="MS")
    rows, values = [], selected.Units_Sold.astype(float).tolist()
    if bundle["model_type"] == "Prophet":
        from prophet.serialize import model_from_json
        model = model_from_json(bundle["prophet_models_json"][(request.product_id, request.country_id)])
        future = calendar_features(pd.DataFrame({"Month": months}))
        future["Selling_Price"], future["Discount"] = price, discount
        future = future.rename(columns={"Month": "ds"})
        predicted = model.predict(future[["ds", *model.extra_regressors]])["yhat"].clip(lower=0)
        for index, month in enumerate(months):
            rows.append({"Month": month.date().isoformat(), "Predicted_Units": float(predicted.iloc[index]),
                         "inputs": {key: float(future.iloc[index][key]) for key in model.extra_regressors}})
    else:
        for month in months:
            row = {"Month": month, "Product_ID": request.product_id, "Country_ID": request.country_id,
                   "Selling_Price": price, "Discount": discount,
                   "lag_1": values[-1], "lag_3": values[-3], "lag_12": values[-12],
                   "rolling_mean_3": float(np.mean(values[-3:])), "rolling_mean_6": float(np.mean(values[-6:]))}
            batch = calendar_features(pd.DataFrame([row]))
            raw_value = float(bundle["model"].predict(batch[FEATURES])[0])
            if not np.isfinite(raw_value):
                raise ValueError("Non-finite model output")
            value = max(0, raw_value)
            values.append(value)
            rows.append({"Month": month.date().isoformat(), "Predicted_Units": value,
                         "inputs": batch[FEATURES].to_dict("records")[0]})
    if not all(np.isfinite(row["Predicted_Units"]) for row in rows):
        raise ValueError("Non-finite model output")
    return {"model": bundle["model_type"], "source": "live_model_inference",
            "product_id": request.product_id, "country_id": request.country_id,
            "requested_month": request.month, "prediction": rows[-1], "forecast_path": rows,
            "assumptions": {"selling_price": price, "discount": discount,
                            "scope": "Fixed for every month from January through requested month; later lags use predictions",
                            "default": "Last 12 observed months average unless overridden",
                            "what_if_note": "Scenario prediction, not a causal estimate of price impact"}}


def train_monthly():
    """React se compare/refit karein; successful run hi active model banega."""
    from ml.monthly_demand_forecast import main

    if not TRAINING_LOCK.acquire(blocking=False):
        raise BlockingIOError("Monthly training already running")
    try:
        run_id = uuid4().hex
        directory = stored.FORECAST_DIR / "runs" / run_id
        code = main(["--from-mongo-sales", "--skip-mongo", "--output-dir", str(directory)])
        if code != 0:
            raise RuntimeError("Monthly training failed; inspect server logs. Previous model remains active.")
        pointer = stored.FORECAST_DIR / f"active_run_{run_id}.tmp"
        pointer.write_text(json.dumps({"run_id": run_id}), encoding="utf-8")
        os.replace(pointer, stored.FORECAST_DIR / "active_run.json")
        return {"status": "completed", "run_id": run_id, "details": model_details()}
    finally:
        TRAINING_LOCK.release()
