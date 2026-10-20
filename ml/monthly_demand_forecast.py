# %% 1. Imports aur configuration
"""Run: python -m ml.monthly_demand_forecast --input ml_dataset.csv.

2025 is a fixed-origin, recursive 12-month model-selection holdout.
After selection, refit the winner on 2023-2025 for the 2026 forecast.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from database.monthly_connection import get_monthly_mongo_uri


KEYS = ["Product_ID", "Country_ID"]
PRICE_FEATURES = ["Selling_Price", "Discount"]
NUMERIC_FEATURES = [
    "Selling_Price", "Discount", "lag_1", "lag_3", "lag_12",
    "rolling_mean_3", "rolling_mean_6", "Month_Number", "Quarter", "Year",
    "Is_Festival_Month",
]
FEATURES = KEYS + NUMERIC_FEATURES
OUTPUT_COLUMNS = ["Month", *KEYS, "Predicted_Units", "Lower", "Upper"]
SEED = 42


# %% 2. Data cleaning aur strict monthly panel validation
def load_and_clean(path):
    """CSV clean karein; unknown sales ko zero ya future data se fill na karein."""
    from ml.mongo_sales_analysis import CSV_COLUMNS

    frame = pd.read_csv(path, dtype={key: "string" for key in KEYS})
    missing = set(CSV_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    return clean_sales_frame(frame)


def load_mongo_sales(uri=None, database="sales_forecasting"):
    """Monthly sales collection se demand data read karein; RM costs required nahi."""
    from pymongo import MongoClient

    projection = {"_id": 0, "Month": 1, "Product_ID": 1, "Country_ID": 1,
                  "Units_Sold": 1, "Selling_Price": 1, "Discount_%": 1}
    with MongoClient(uri or get_monthly_mongo_uri(), serverSelectionTimeoutMS=8000) as client:
        records = list(client[database]["sales"].find({}, projection))
    frame = pd.DataFrame(records).rename(columns={"Discount_%": "Discount"})
    print(f"Step 2: read {len(frame)} rows directly from {database}.sales")
    return clean_sales_frame(frame)


def clean_sales_frame(frame):
    """Demand ke required fields validate karein; profit data fabricate na karein."""
    frame = frame.copy()
    required = {"Month", *KEYS, "Units_Sold", *PRICE_FEATURES}
    if required - set(frame.columns):
        raise ValueError(f"Missing sales columns: {sorted(required - set(frame.columns))}")
    frame["Month"] = pd.to_datetime(frame["Month"], utc=True, errors="raise").dt.tz_localize(None)
    if frame["Month"].isna().any() or not frame["Month"].eq(frame["Month"].dt.to_period("M").dt.to_timestamp()).all():
        raise ValueError("Month must be the first day of each month at midnight UTC")
    for key in KEYS:
        frame[key] = frame[key].astype("string").str.strip()
        if frame[key].isna().any() or frame[key].eq("").any():
            raise ValueError(f"Missing/blank {key}")
    for column in ["Units_Sold", "Selling_Price", "Discount", "RM_Cost", "Shipping", "Profit_per_unit"]:
        if column not in frame:
            continue
        frame[column] = pd.to_numeric(frame[column], errors="raise")
        if not np.isfinite(frame[column].to_numpy(dtype=float)).all():
            raise ValueError(f"Missing/infinite numeric data in {column}")
    nonnegative = [column for column in ["Units_Sold", "Selling_Price", "RM_Cost", "Shipping"] if column in frame]
    if (frame[nonnegative] < 0).any().any():
        raise ValueError("Units, price, RM cost and shipping must be nonnegative")
    if not frame["Discount"].between(0, 100).all():
        raise ValueError("Discount must be a percentage between 0 and 100")
    if frame.duplicated(["Month", *KEYS]).any():
        raise ValueError("Duplicate month/product/country: resolve upstream; prices cannot be summed")
    expected = pd.date_range("2023-01-01", "2025-12-01", freq="MS")
    if frame["Product_ID"].nunique() != 5 or frame["Country_ID"].nunique() != 10:
        raise ValueError("Expected 5 products and 10 countries")
    groups = list(frame.groupby(KEYS, sort=True))
    if len(groups) != 50:
        raise ValueError("Expected all 50 product-country combinations")
    for key, group in groups:
        if not pd.DatetimeIndex(group["Month"].sort_values()).equals(expected):
            raise ValueError(f"{key}: expected exactly 36 months, Jan 2023-Dec 2025")
    frame = calendar_features(frame)
    print(f"Step 2: cleaned {len(frame)} rows; 50 complete series, 36 months each")
    return frame.sort_values([*KEYS, "Month"]).reset_index(drop=True)


# %% 3. Calendar, grouped lags aur rolling features
def calendar_features(frame):
    """Calendar features date se dobara banayein; supplied flags par depend na karein."""
    result = frame.copy()
    result["Month_Number"] = result["Month"].dt.month
    result["Quarter"] = result["Month"].dt.quarter
    result["Year"] = result["Month"].dt.year
    result["Is_Festival_Month"] = result["Month_Number"].isin([10, 11]).astype(int)
    return result


def engineer_features(history):
    """Har series mein sirf previous months se lags aur rolling means banayein."""
    frame = calendar_features(history.sort_values([*KEYS, "Month"]))
    grouped = frame.groupby(KEYS, sort=False)["Units_Sold"]
    for lag in (1, 3, 12):
        frame[f"lag_{lag}"] = grouped.shift(lag)
    for window in (3, 6):
        frame[f"rolling_mean_{window}"] = grouped.transform(
            lambda values: values.shift(1).rolling(window, min_periods=window).mean()
        )
    # 12-month lag ke kaaran pehle 12 months training rows nahi banenge.
    return frame.dropna(subset=NUMERIC_FEATURES).reset_index(drop=True)


def future_price_assumptions(history):
    """Forecast origin ke last 12 observed months ka per-series price/discount average lein."""
    # 2025 backtest: averages sirf 2024 se; 2026 forecast: averages sirf 2025 se.
    # Saare 12 future months mein ye fixed averages assume hain; future actuals nahi.
    return history.sort_values("Month").groupby(KEYS).tail(12).groupby(KEYS)[PRICE_FEATURES].mean()


# %% 4. Global XGBoost aur Random Forest training
def fit_tree_model(name, history):
    """Train-only one-hot encoding ke saath ek global demand model fit karein."""
    if name == "XGBoost":
        from xgboost import XGBRegressor
        estimator = XGBRegressor(
            n_estimators=250, max_depth=3, learning_rate=0.04,
            subsample=0.9, colsample_bytree=0.9, objective="reg:squarederror",
            random_state=SEED, n_jobs=2,
        )
    elif name == "Random Forest":
        estimator = RandomForestRegressor(
            n_estimators=250, max_depth=10, min_samples_leaf=3,
            random_state=SEED, n_jobs=2,
        )
    else:
        raise ValueError(f"Unknown tree model: {name}")
    features = engineer_features(history)
    if features.empty:
        raise ValueError("Not enough training history for lag_12")
    encoder = ColumnTransformer([
        ("ids", OneHotEncoder(handle_unknown="error", sparse_output=False), KEYS),
        ("numeric", "passthrough", NUMERIC_FEATURES),
    ])
    model = Pipeline([("encode", encoder), ("model", estimator)])
    model.fit(features[FEATURES], features["Units_Sold"])
    print(f"Step 4: {name} fitted on {len(features)} rows; IDs encoded from training only")
    return model


def recursive_tree_forecast(model, history, months):
    """Poora future horizon predict karein; next lag mein prediction hi append karein."""
    assumptions = future_price_assumptions(history)
    series = {key: group.sort_values("Month")["Units_Sold"].astype(float).tolist()
              for key, group in history.groupby(KEYS, sort=True)}
    output = []
    for month in months:
        rows = []
        for key, values in series.items():
            rows.append({
                "Month": month, **dict(zip(KEYS, key)),
                **assumptions.loc[key].to_dict(),
                "lag_1": values[-1], "lag_3": values[-3], "lag_12": values[-12],
                "rolling_mean_3": np.mean(values[-3:]), "rolling_mean_6": np.mean(values[-6:]),
            })
        batch = calendar_features(pd.DataFrame(rows))
        predictions = np.maximum(0, model.predict(batch[FEATURES]))
        if not np.isfinite(predictions).all():
            raise ValueError("Model generated non-finite predictions")
        for row, prediction in zip(rows, predictions):
            key = tuple(row[column] for column in KEYS)
            series[key].append(float(prediction))
            output.append({"Month": month, **dict(zip(KEYS, key)), "Predicted_Units": float(prediction)})
    return pd.DataFrame(output)


# %% 5. Prophet: har product-country ka alag model
def prophet_forecast(history, months, model_store=None):
    """Har series par Prophet fit karein; sirf forecast-origin information use karein."""
    from prophet import Prophet

    assumptions = future_price_assumptions(history)
    outputs = []
    for number, (key, group) in enumerate(history.groupby(KEYS, sort=True), 1):
        training = group.sort_values("Month").rename(columns={"Month": "ds", "Units_Sold": "y"})
        # 24 train observations ke liye low-order yearly seasonality rakhein.
        model = Prophet(yearly_seasonality=False, weekly_seasonality=False,
                        daily_seasonality=False, uncertainty_samples=0)
        model.add_seasonality(name="annual", period=365.25, fourier_order=3)
        # Constant regressors omit karein; Prophet ko lags ki zarurat nahi hai.
        regressors = [col for col in [*PRICE_FEATURES, "Is_Festival_Month"] if training[col].nunique() > 1]
        for column in regressors:
            model.add_regressor(column)
        # Short monthly series par default Newton ki jagah LBFGS se fit karein.
        model.fit(training[["ds", "y", *regressors]], algorithm="LBFGS")
        if model_store is not None:
            from prophet.serialize import model_to_json
            model_store[key] = model_to_json(model)
        future = calendar_features(pd.DataFrame({"Month": months})).rename(columns={"Month": "ds"})
        for column in PRICE_FEATURES:
            future[column] = assumptions.loc[key, column]
        predicted = model.predict(future[["ds", *regressors]])["yhat"].clip(lower=0).to_numpy()
        if not np.isfinite(predicted).all():
            raise ValueError(f"Prophet produced invalid predictions for {key}")
        outputs.append(pd.DataFrame({"Month": months, **dict(zip(KEYS, key)), "Predicted_Units": predicted}))
        if number % 10 == 0:
            print(f"Step 5: Prophet fitted {number}/50 series")
    return pd.concat(outputs, ignore_index=True)


# %% 6. Chronological comparison: 2023-24 train, full 2025 test
def score_predictions(actual, forecast):
    """Same 600 holdout rows par MAPE, RMSE aur MAE calculate karein."""
    merged = actual[["Month", *KEYS, "Units_Sold"]].merge(
        forecast, on=["Month", *KEYS], how="outer", validate="one_to_one", indicator=True,
    )
    if not merged["_merge"].eq("both").all():
        raise ValueError("Forecast and actual keys do not match")
    merged = merged.drop(columns="_merge")
    actual_values = merged["Units_Sold"].to_numpy(dtype=float)
    predicted = merged["Predicted_Units"].to_numpy(dtype=float)
    nonzero = actual_values != 0
    # Actual=0 par MAPE undefined hai; un rows ko MAPE se exclude aur count report karein.
    mape = float(np.mean(np.abs((actual_values[nonzero] - predicted[nonzero]) / actual_values[nonzero])) * 100) if nonzero.any() else np.nan
    metrics = {"MAPE_Pct": mape, "RMSE": float(np.sqrt(mean_squared_error(actual_values, predicted))),
               "MAE": float(mean_absolute_error(actual_values, predicted)),
               "MAPE_Excluded_Zero_Rows": int((~nonzero).sum())}
    return metrics, merged


def compare_models(history):
    """Teen models compare karein; pehle se fixed rule lowest RMSE, then MAE hai."""
    train = history.loc[history["Month"] < "2025-01-01"].copy()
    test = history.loc[history["Month"] >= "2025-01-01"].copy()
    months = pd.date_range("2025-01-01", periods=12, freq="MS")
    print(f"Step 6: time split = {len(train)} train rows / {len(test)} test rows; no random split")
    print("2025 predictions use 2024 average price/discount and recursive lags; no 2025 actual features")
    comparisons, predictions, fitted = [], {}, {}
    for name in ("Prophet", "XGBoost", "Random Forest"):
        if name == "Prophet":
            forecast = prophet_forecast(train, months)
        else:
            fitted[name] = fit_tree_model(name, train)
            forecast = recursive_tree_forecast(fitted[name], train, months)
        metrics, predictions[name] = score_predictions(test, forecast)
        comparisons.append({"Model": name, **metrics})
        print(f"Step 6: {name} evaluated on all {len(test)} holdout rows")
    table = pd.DataFrame(comparisons).sort_values(["RMSE", "MAE", "Model"]).reset_index(drop=True)
    best = table.iloc[0]["Model"]
    print("\nModel comparison (MAPE in %, RMSE/MAE in units):\n" + table.to_string(index=False))
    print(f"Best model: {best}; lowest 12-month holdout RMSE, with MAE as tie-breaker.")
    print("RMSE large demand errors ko zyada weight deta hai. 2025 model selection ke liye use hua;")
    print("ye metrics independent final-test accuracy ya 2026 accuracy ki guarantee nahi hain.")
    return best, table, predictions, fitted


# %% 7. Winner refit aur approximate prediction ranges
def add_prediction_ranges(forecast, backtest):
    """2025 ke per-horizon errors se approximate 90% empirical ranges banayein."""
    # Har horizon ke 50 cross-series absolute errors ka 90th percentile use hota hai.
    # Ye descriptive ranges hain: formal coverage guarantee nahi; model selection
    # aur calibration dono 2025 par hain. Future price uncertainty bhi include nahi.
    errors = backtest.assign(
        Horizon=backtest["Month"].dt.month,
        Absolute_Error=(backtest["Units_Sold"] - backtest["Predicted_Units"]).abs(),
    )
    widths = errors.groupby("Horizon")["Absolute_Error"].quantile(0.9)
    result = forecast.copy()
    width = result["Month"].dt.month.map(widths).to_numpy()
    result["Lower"] = np.maximum(0, result["Predicted_Units"] - width)
    result["Upper"] = result["Predicted_Units"] + width
    return result[OUTPUT_COLUMNS].sort_values(["Month", *KEYS]).reset_index(drop=True)


# %% 8. Matplotlib charts
def save_charts(history, predictions, forecast, xgboost_model, directory):
    """Test comparison, product forecasts aur XGBoost importances ki PNGs save karein."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"figure.dpi": 140, "axes.grid": True, "grid.alpha": 0.2})
    actual = history.loc[history["Month"].dt.year == 2025].groupby("Month")["Units_Sold"].sum()
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(actual.index, actual.values, label="Actual", color="black", linewidth=2.5)
    for name, frame in predictions.items():
        totals = frame.groupby("Month")["Predicted_Units"].sum()
        ax.plot(totals.index, totals.values, marker="o", label=name)
    ax.set(title="2025 test: total demand across 50 combinations", ylabel="Units")
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(directory / "actual_vs_predicted_2025.png")
    plt.close(fig)

    fig, axes = plt.subplots(5, 1, figsize=(12, 19))
    for ax, (product, frame) in zip(axes, forecast.groupby("Product_ID", sort=True)):
        # Har country ki separate line; correlated intervals ko sum nahi karte.
        for country, group in frame.groupby("Country_ID", sort=True):
            group = group.sort_values("Month")
            ax.plot(group["Month"], group["Predicted_Units"], label=str(country))
        ax.set(title=f"Product {product}: 2026 demand by country", ylabel="Units")
        ax.legend(ncol=5, fontsize=8)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(directory / "product_forecasts_2026.png")
    plt.close(fig)

    names = xgboost_model.named_steps["encode"].get_feature_names_out()
    importance = pd.Series(xgboost_model.named_steps["model"].feature_importances_, index=names).sort_values()
    fig, ax = plt.subplots(figsize=(10, 9))
    importance.plot.barh(ax=ax, color="#207f98")
    ax.set(title="XGBoost feature importance (2023-2024 training)", xlabel="Normalized gain importance")
    fig.tight_layout()
    fig.savefig(directory / "xgboost_feature_importance.png")
    plt.close(fig)
    importance.rename("Importance").to_csv(directory / "xgboost_feature_importance.csv")
    print(f"Step 8: saved 3 matplotlib charts and feature importance CSV in {directory}")


# %% 9. MongoDB persistence
def save_to_mongodb(forecast, uri=None, database="sales_forecasting"):
    """Forecast rows upsert karein; repeat run par duplicate forecasts na banayein."""
    from pymongo import ASCENDING, MongoClient, UpdateOne

    records = forecast.copy()
    records["Month"] = records["Month"].map(lambda month: month.to_pydatetime())
    documents = records.to_dict("records")
    # CSV IDs string ke roop mein preserve hote hain (leading zeros bhi).
    with MongoClient(uri or get_monthly_mongo_uri(), serverSelectionTimeoutMS=5000) as client:
        client.admin.command("ping")
        collection = client[database]["forecast_2026"]
        collection.create_index([(column, ASCENDING) for column in ["Month", *KEYS]], unique=True)
        operations = [UpdateOne({key: row[key] for key in ["Month", *KEYS]}, {"$set": row}, upsert=True)
                      for row in documents]
        result = collection.bulk_write(operations, ordered=False)
        print(f"Step 9: MongoDB forecast_2026: {result.upserted_count} inserted, {result.matched_count} updated/matched")


# %% 10. End-to-end execution
def main(argv=None):
    """CSV se comparison, 2026 forecast, charts aur MongoDB export execute karein."""
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--input", type=Path, default=Path("ml_dataset.csv"))
    source.add_argument("--from-mongo-sales", action="store_true", help="Train directly on monthly sales without RM costs")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/monthly_demand"))
    parser.add_argument("--mongo-uri", default=None)
    parser.add_argument("--database", default="sales_forecasting")
    parser.add_argument("--skip-mongo", action="store_true", help="Generate local artifacts only")
    args = parser.parse_args(argv)
    try:
        if not args.from_mongo_sales and not args.input.is_file():
            raise FileNotFoundError(f"Input CSV not found: {args.input}. Run ml.mongo_sales_analysis first.")
        # Required libraries pehle verify karein; aadha comparison quietly skip nahi hoga.
        import prophet  # noqa: F401
        import xgboost  # noqa: F401
        import matplotlib  # noqa: F401

        print("Step 1: dependencies ready; random seed = 42")
        np.random.seed(SEED)
        history = load_mongo_sales(args.mongo_uri, args.database) if args.from_mongo_sales else load_and_clean(args.input)
        print(f"Step 3: {len(engineer_features(history))} usable lag-feature rows; Profit_per_unit excluded from predictors")
        best, table, predictions, fitted = compare_models(history)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        table.to_csv(args.output_dir / "model_comparison.csv", index=False)
        for name, frame in predictions.items():
            frame.to_csv(args.output_dir / f"test_predictions_{name.lower().replace(' ', '_')}.csv", index=False)
        months = pd.date_range("2026-01-01", periods=12, freq="MS")
        model_store = {}
        if best == "Prophet":
            raw = prophet_forecast(history, months, model_store=model_store)
        else:
            model = fit_tree_model(best, history)
            raw = recursive_tree_forecast(model, history, months)
        import joblib
        # Monthly model ko alag save karein; existing weekly model ka path alag hai.
        joblib.dump({
            "model_type": best, "model": model if best != "Prophet" else None,
            "prophet_models_json": model_store,
            "history": history[["Month", *KEYS, "Units_Sold", *PRICE_FEATURES]],
            "feature_columns": FEATURES, "trained_through": "2025-12-01",
            "test_metrics": table.to_dict("records"),
        }, args.output_dir / "best_model.joblib")
        forecast = add_prediction_ranges(raw, predictions[best])
        output = args.output_dir / "forecast_2026.csv"
        forecast.to_csv(output, index=False, date_format="%Y-%m-%d")
        metadata = {
            "best_model": best, "selection_rule": "2025 fixed-origin RMSE, MAE tie-breaker",
            "data_source": f"{args.database}.sales" if args.from_mongo_sales else str(args.input),
            "final_training_period": "2023-01 through 2025-12",
            "future_price_discount": "Per-series mean of last 12 observed months, held constant",
            "ranges": "Approximate 90% absolute-error quantiles by horizon, pooled across 2025 series; no coverage guarantee",
            "selection_and_calibration_share_holdout": True,
            "predictors_excluded": ["RM_Cost", "Shipping", "Profit_per_unit"],
        }
        (args.output_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        print(f"Step 7: winner refitted through Dec 2025; {len(forecast)} forecasts saved to {output}")
        print("Lower/Upper: approximate 90% empirical ranges; calibration shares the model-selection holdout")
        save_charts(history, predictions, forecast, fitted["XGBoost"], args.output_dir)
        if args.skip_mongo:
            print("Step 9: MongoDB skipped by --skip-mongo")
        else:
            save_to_mongodb(forecast, args.mongo_uri, args.database)
        print("Step 10: demand forecasting completed")
        return 0
    except Exception as exc:
        # Mongo fail hone par already saved CSV/charts disk par available rahenge.
        print(f"Forecasting failed ({type(exc).__name__}): {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
