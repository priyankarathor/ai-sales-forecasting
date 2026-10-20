from pathlib import Path

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from ml.baseline import SevenPeriodBaseline
from ml.artifacts import model_path as isolated_model_path, save_artifact

from ml.data_loader import load_sales
from ml.preprocessing import prepare_daily_sales, prepare_model_series
from ml.features import (
    create_features,
    FEATURE_COLUMNS,
    TARGET_COLUMN,
)


def train_model(product_id: str, region: str, *, model_type: str = "linear_regression",
                frequency: str = "auto", test_fraction: float = 0.2):
    if model_type not in {"linear_regression", "random_forest", "auto"}:
        raise ValueError("Unsupported model type")
    if frequency not in {"auto", "D", "7D"}:
        raise ValueError("Unsupported frequency")
    if not 0.1 <= test_fraction <= 0.4:
        raise ValueError("test_fraction must be between 0.1 and 0.4")
    requested_model_type = model_type
    sales = load_sales(
        product_id=product_id,
        region=region,
    )

    daily = prepare_daily_sales(
        sales,
        product_id=product_id,
        region=region,
    )

    daily, frequency = prepare_model_series(daily, None if frequency == "auto" else frequency)
    period = "weeks" if frequency == "7D" else "days"
    print(f"Observation frequency: {period} (recorded units, no missing dates filled)")
    data = create_features(daily, frequency=frequency)

    # Starter guard, not a guarantee of sufficient history.
    if len(data) < 35:
        raise ValueError(
            f"Only {len(data)} usable {period} found. "
            f"Provide at least 42 consecutive observed {period}."
        )

    # Keep a chronological holdout, with at least seven observations.
    test_days = max(7, int(len(data) * test_fraction))

    train = data.iloc[:-test_days]
    test = data.iloc[-test_days:]

    X_train = train[FEATURE_COLUMNS]
    y_train = train[TARGET_COLUMN]

    X_test = test[FEATURE_COLUMNS]
    y_test = test[TARGET_COLUMN]

    forest = RandomForestRegressor(
        n_estimators=200,
        max_depth=8,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    )

    candidates = {
        "linear_regression": make_pipeline(StandardScaler(), LinearRegression()),
        "random_forest": forest,
    }
    evaluated = {}
    for name, candidate in candidates.items():
        if model_type != "auto" and name != model_type:
            continue
        candidate.fit(X_train, y_train)
        estimates = np.maximum(0, candidate.predict(X_test))
        if not np.isfinite(estimates).all():
            raise ValueError("Training produced non-finite predictions")
        evaluated[name] = (float(mean_absolute_error(y_test, estimates)), candidate, estimates)
    model_type = min(evaluated, key=lambda name: evaluated[name][0])
    model_mae, model, predictions = evaluated[model_type]

    # Baseline: predict the same sales as seven observation periods earlier.
    baseline_predictions = test["lag_7"]

    model_mae = mean_absolute_error(y_test, predictions)
    baseline_mae = mean_absolute_error(
        y_test,
        baseline_predictions,
    )

    print(f"Training {period}:", len(train))
    print(f"Evaluation {period}:", len(test))
    print(f"Model MAE: {model_mae:.2f} units")
    print(f"Baseline MAE: {baseline_mae:.2f} units")

    comparison = test[["date", TARGET_COLUMN]].copy()
    comparison["predicted_units"] = predictions
    comparison["baseline_units"] = baseline_predictions

    print("\nLatest predictions:")
    print(comparison.tail(10).to_string(index=False))

    if requested_model_type == "auto" and model_mae >= baseline_mae:
        print(
            "\nCandidates did not beat the baseline. "
            "Saving the better seven-observation baseline for forecasting."
        )
        model = SevenPeriodBaseline()
        model_type = "seven_period_baseline"
        model_mae = baseline_mae
        comparison["predicted_units"] = baseline_predictions

    # Evaluation is complete. Refit using all available history.
    model.fit(
        data[FEATURE_COLUMNS],
        data[TARGET_COLUMN],
    )

    artifact = {
        "model": model,
        "feature_columns": FEATURE_COLUMNS,
        "target_column": TARGET_COLUMN,
        "product_id": product_id,
        "region": region,
        "frequency": frequency,
        "model_type": model_type,
        "requested_model_type": requested_model_type,
        "clip_nonnegative": True,
        "test_fraction": test_fraction,
        "data_summary": {"source": "sales", "source_rows": len(sales),
                         "observations": len(daily), "usable_observations": len(data),
                         "first_sales_date": daily["date"].min().date().isoformat()},
        "trained_through": daily["date"].max().isoformat(),
        "evaluation": {
            "train_days": len(train),
            "train_observations": len(train),
            "test_observations": test_days,
            "model_mae": float(model_mae),
            "candidate_mae": {name: result[0] for name, result in evaluated.items()},
            "random_forest_mae": evaluated.get("random_forest", (None,))[0],
            "linear_regression_mae": evaluated.get("linear_regression", (None,))[0],
            "baseline_mae": float(baseline_mae),
            "test_days": test_days,
            "forecast_horizon_days": 7 if frequency == "7D" else 1,
            "period_unit": period,
            "rmse": float(np.sqrt(np.mean((comparison["predicted_units"] - y_test) ** 2))),
            "wape_pct": (float(np.abs(comparison["predicted_units"] - y_test).sum() / y_test.sum() * 100)
                         if y_test.sum() > 0 else None),
            "scope": "Chronological holdout, one observation ahead using actual prior sales; not recursive horizon accuracy.",
            "comparison": [
                {
                    "date": row["date"].date().isoformat(),
                    "actual_units": float(row[TARGET_COLUMN]),
                    "predicted_units": float(row["predicted_units"]),
                    "baseline_units": float(row["baseline_units"]),
                }
                for _, row in comparison.iterrows()
            ],
        },
    }

    artifacts_dir = (
        Path(__file__).resolve().parents[1] / "artifacts"
    )
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    # Keep the latest model for legacy GET routes and a separate model for POST requests.
    model_path = artifacts_dir / "sales_model.joblib"
    save_artifact(artifact, model_path)
    save_artifact(artifact, isolated_model_path(product_id, region, requested_model_type))

    print(f"\nModel saved: {model_path}")
    return {key: value for key, value in artifact.items() if key != "model"}


if __name__ == "__main__":
    sales = load_sales()

    combinations = (
        sales[["product_id", "region"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )

    # Show numbered options.
    print(combinations.to_string())

    choice = int(input("\nEnter row number to train: "))

    if choice < 0 or choice >= len(combinations):
        raise ValueError("Invalid row number")

    selected = combinations.iloc[choice]

    print(
        f"\nTraining: {selected['product_id']} "
        f"| {selected['region']}"
    )

    train_model(
        product_id=selected["product_id"],
        region=selected["region"],
    )
