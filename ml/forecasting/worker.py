"""CLI/background-worker entry point; never imported by existing HTTP routes."""
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
from uuid import uuid4

import joblib
import pandas as pd

from .cleaning import clean_sales
from .config import TenantConfig
from .training import FEATURES, forecast, select_model


def pull(cfg):
    """Read a complete CSV/Mongo snapshot. No write-capable source methods are used."""
    if cfg.source["type"] == "csv":
        # Strings preserve leading zeros, missing strings and original numeric representations.
        return pd.read_csv(cfg.source["path"], dtype=str, keep_default_na=False).to_dict("records")
    from bson import json_util
    from pymongo import MongoClient
    uri = os.environ.get(cfg.source.get("uri_env", "MONTHLY_MONGO_URL"))
    if not uri:
        raise ValueError("Set the source uri_env environment variable")
    with MongoClient(uri, serverSelectionTimeoutMS=8000) as client:
        source = client[cfg.source["database"]][cfg.source["collection"]]
        # Optional source_filter must select the tenant when a source collection is shared.
        records = list(source.find(cfg.source.get("filter", {})))
    return json.loads(json_util.dumps(records))


def normalize_raw(records, cfg):
    """Decode Mongo extended date/decimal wrappers only in the cleaning input copy."""
    result = []
    for original in records:
        row = original.copy()
        for source in cfg.mapping.values():
            value = row.get(source)
            if isinstance(value, dict) and "$date" in value:
                value = value["$date"]
                if isinstance(value, dict) and "$numberLong" in value:
                    value = pd.to_datetime(int(value["$numberLong"]), unit="ms", utc=True).isoformat()
            elif isinstance(value, dict) and "$numberDecimal" in value:
                value = value["$numberDecimal"]
            row[source] = value
        result.append(row)
    return result


def clean_load(cfg, store, load_id):
    records = store.read_raw(cfg.tenant_id, load_id)
    try:
        clean, quality = clean_sales(normalize_raw(records, cfg), cfg)
        # Strict first release: reject malformed rows instead of silently lowering period totals.
        if quality["rows_rejected"] or quality["gaps"]:
            store.quality_failure(cfg.tenant_id, load_id, quality)
            raise ValueError("Data quality gate failed; inspect sales_loads.quality; raw data preserved")
        store.save_clean(cfg.tenant_id, load_id, clean, quality)
    except ValueError:
        # Preserve an existing detailed quality report when one was already written.
        if "quality" not in locals():
            store.quality_failure(cfg.tenant_id, load_id, {"error": "Cleaning failed; check schema, duplicates and returns policy"})
        raise
    return quality


def run(cfg, store, models_dir=Path("backend/models"), *, load_id=None):
    if load_id is None:
        load_id = uuid4().hex
        store.save_raw(cfg.tenant_id, load_id, pull(cfg))
    elif not store.read_raw(cfg.tenant_id, load_id):
        raise ValueError("No preserved raw snapshot for this tenant/load_id")
    quality = clean_load(cfg, store, load_id)
    # Train only from the saved, tenant-scoped clean table.
    history = store.read_sales(cfg.tenant_id, load_id)
    run_id = "v_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "_" + uuid4().hex
    result = {}
    try:
        model, result = select_model(history, cfg)
        result.update({"config": asdict(cfg), "load_id": load_id, "quality": quality,
                       "data_cutoff": history.sale_date.max().date().isoformat(),
                       "features": FEATURES, "random_state": cfg.random_state})
        if model is None:
            store.save_run(cfg.tenant_id, load_id, run_id, result)
            return {"run_id": run_id, "status": "rejected", **result}
        target_dates = pd.date_range(history.sale_date.max(), periods=cfg.horizon + 1, freq=cfg.frequency)[1:]
        prediction = forecast(model, history, target_dates, cfg)
        directory = Path(models_dir) / cfg.tenant_id / run_id
        directory.mkdir(parents=True, exist_ok=False)
        import sklearn
        import numpy
        result["library_versions"] = {"sklearn": sklearn.__version__, "pandas": pd.__version__,
                                      "numpy": numpy.__version__, "joblib": joblib.__version__}
        if result["model_type"] == "xgboost":
            import xgboost
            result["library_versions"]["xgboost"] = xgboost.__version__
        result["model_parameters"] = {k: (str(v) if isinstance(v, float) and not math.isfinite(v) else v)
                                      for k, v in model.named_steps["model"].get_params().items()
                                      if v is None or isinstance(v, (str, int, float, bool))}
        result["model_path"] = str(directory / "model.joblib")
        joblib.dump({"tenant_id": cfg.tenant_id, "model_version": run_id,
                     "model": model, "metadata": result}, directory / "model.joblib")
        (directory / "metadata.json").write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
        prediction.to_csv(directory / "forecasts.csv", index=False)
        # Only this transaction makes the model/version visible as promoted.
        store.save_run(cfg.tenant_id, load_id, run_id, result, prediction)
        return {"run_id": run_id, "status": "promoted", "forecast_rows": len(prediction), **result}
    except Exception as exc:
        result.update({"status": "failed", "passed": False,
                       "error_type": type(exc).__name__, "load_id": load_id})
        store.save_run(cfg.tenant_id, load_id, run_id, result)
        raise


def load_model(path, tenant_id):
    """Only trusted local model files may be loaded; enforce tenant ownership."""
    bundle = joblib.load(path)
    if bundle["tenant_id"] != tenant_id:
        raise ValueError("Model belongs to another tenant")
    return bundle


def demo(path):
    import numpy as np
    rng = np.random.default_rng(42)
    rows = []
    for product in ("P1", "P2"):
        for region in ("C1", "C2"):
            for i, date in enumerate(pd.date_range("2021-01-01", periods=60, freq="MS")):
                units = 150 + 2*i + 40*np.sin(2*np.pi*i/12) + rng.normal(0, 3)
                rows.append({"Month": date.date().isoformat(), "Product_ID": product,
                             "Country_ID": region, "Units_Sold": round(max(0, units))})
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["init", "check", "demo", "run", "run-upload", "clean", "accuracy"])
    parser.add_argument("--tenant-id", help="Tenant owning the uploaded CSV")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--load-id", help="Re-clean/retrain an existing preserved load")
    parser.add_argument("--output", type=Path, default=Path("data/forecast_demo.csv"))
    parser.add_argument("--models-dir", type=Path, default=Path("backend/models"))
    args = parser.parse_args()
    if args.command == "demo":
        demo(args.output)
        print(f"Demo snapshot saved: {args.output}")
        return
    from database.forecast_store import ForecastStore
    store = ForecastStore()
    try:
        if args.command == "check":
            from sqlalchemy import text
            with store.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            print("Forecast database connection successful")
            return
        if args.command == "init":
            store.create_schema()
            print("Forecasting schema initialized")
            return
        if args.command == "run-upload":
            if not args.tenant_id or not args.load_id:
                parser.error("run-upload requires --tenant-id and --load-id")
            saved = store.read_upload_config(args.tenant_id, args.load_id)
            if saved is None:
                parser.error("No uploaded CSV configuration for this tenant/load_id")
            cfg = TenantConfig(**saved)
            result = run(cfg, store, args.models_dir, load_id=args.load_id)
            print(json.dumps(result, indent=2, default=str, allow_nan=False))
            return
        if args.config is None:
            parser.error("--config is required")
        cfg = TenantConfig.load(args.config)
        if args.command == "accuracy":
            result = store.accuracy(cfg.tenant_id)
        elif args.command == "clean":
            if not args.load_id:
                parser.error("clean requires --load-id")
            result = clean_load(cfg, store, args.load_id)
        else:
            result = run(cfg, store, args.models_dir, load_id=args.load_id)
        encoded = json.dumps(result, indent=2, default=str, allow_nan=False)
        if args.command == "accuracy":
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.with_suffix(".json").write_text(encoded, encoding="utf-8")
        print(encoded)
    finally:
        store.engine.dispose()


if __name__ == "__main__":
    main()
