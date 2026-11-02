"""Offline tests for cleaning, temporal evaluation, promotion and tenant storage."""
from dataclasses import replace
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from ml.forecasting.cleaning import clean_sales
from ml.forecasting.config import TenantConfig
from ml.forecasting.training import FEATURES, features, forecast, select_model
from ml.forecasting.worker import demo, load_model, run


def config(**changes):
    base = TenantConfig("demo", {"type": "csv", "path": "unused.csv"},
                        {"date": "Month", "product": "Product_ID", "region": "Country_ID", "quantity": "Units_Sold"},
                        candidate_models=("random_forest",))
    return replace(base, **changes)


def records(periods=48, units=None):
    values = units if units is not None else np.arange(periods) + 100
    return [{"Month": date.date().isoformat(), "Product_ID": "001", "Country_ID": "IN", "Units_Sold": str(value)}
            for date, value in zip(pd.date_range("2022-01-01", periods=periods, freq="MS"), values)]


class CleaningTests(unittest.TestCase):
    def test_repeatable_gaps_invalid_and_outliers(self):
        source = records()
        source.pop(4)
        source[0]["Units_Sold"] = "bad"
        source[-1]["Units_Sold"] = "99999"
        original = json.dumps(source)
        first, quality = clean_sales(source, config())
        second, _ = clean_sales(source, config())
        pd.testing.assert_frame_equal(first, second)
        self.assertEqual(json.dumps(source), original)
        self.assertEqual(quality["rows_rejected"], 1)
        self.assertTrue(quality["gaps"])
        self.assertTrue(quality["outliers"])
        self.assertEqual(first.product_id.iloc[0], "001")
        self.assertNotIn(pd.Timestamp("2022-05-01"), set(first.sale_date))

    def test_duplicates_and_returns_are_explicit(self):
        source = records(1)
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            clean_sales(source + source, config())
        mapping = {**config().mapping, "record_id": "id"}
        cfg = config(mapping=mapping, source_grain="transaction", returns_policy="subtract")
        source = [{**source[0], "id": "one"}, {**source[0], "id": "two", "Units_Sold": "-10"}]
        clean, quality = clean_sales(source + source[:1], cfg)
        self.assertEqual(clean.units_sold.iloc[0], 90)
        self.assertEqual(quality["duplicates_removed"], 1)
        # Two distinct sales with identical business values are both retained.
        clean, _ = clean_sales([{**source[0], "id": "one"}, {**source[0], "id": "two"}], cfg)
        self.assertEqual(clean.units_sold.iloc[0], 200)
        with self.assertRaisesRegex(ValueError, "Negative sales"):
            clean_sales(source, replace(cfg, returns_policy="reject"))
        clean, quality = clean_sales(source, replace(cfg, returns_policy="exclude"))
        self.assertEqual(clean.units_sold.iloc[0], 100)
        self.assertEqual(quality["returns_excluded"], 1)

    def test_config_rejects_path_traversal_and_credentials(self):
        with self.assertRaises(ValueError):
            config(tenant_id="../other")
        with self.assertRaises(ValueError):
            config(source={"type": "mongo", "uri": "credential"})


class TrainingTests(unittest.TestCase):
    def test_features_never_use_current_or_future_target(self):
        clean, _ = clean_sales(records(), config())
        original = features(clean, config())
        cutoff = pd.Timestamp("2024-01-01")
        changed = clean.copy()
        changed.loc[changed.sale_date >= cutoff, "units_sold"] = 99999
        altered = features(changed, config())
        pd.testing.assert_frame_equal(original.loc[original.sale_date <= cutoff, FEATURES],
                                      altered.loc[altered.sale_date <= cutoff, FEATURES])

    def test_recursive_prediction_uses_earlier_prediction(self):
        clean, _ = clean_sales(records(), config())
        class Model:
            def predict(self, frame):
                return frame.lag_1.to_numpy() + 1
        targets = pd.date_range("2026-01-01", periods=3, freq="MS")
        result = forecast(Model(), clean, targets, config())
        np.testing.assert_array_equal(result.predicted_units, [148, 149, 150])

    def test_baseline_tie_is_rejected_and_splits_are_chronological(self):
        clean, _ = clean_sales(records(units=np.full(48, 100)), config())
        model, result = select_model(clean, config())
        self.assertIsNone(model)
        self.assertFalse(result["passed"])
        self.assertEqual(result["baseline"]["wape"], 0)
        for fold in result["backtests"]["random_forest"]:
            self.assertLess(fold["train_end"], fold["test_start"])
            self.assertLess(fold["test_end"], result["test_start"])

    def test_zero_actuals_and_gaps_fail_honestly(self):
        clean, _ = clean_sales(records(units=np.zeros(48)), config())
        with self.assertRaisesRegex(ValueError, "WAPE"):
            select_model(clean, config())
        with self.assertRaisesRegex(ValueError, "incomplete panel"):
            select_model(clean.drop(index=20), config())

    def test_missing_optional_xgboost(self):
        from ml.forecasting.training import make_model
        import builtins
        real_import = builtins.__import__
        def missing(name, *args, **kwargs):
            if name == "xgboost":
                raise ImportError("not installed")
            return real_import(name, *args, **kwargs)
        with patch("builtins.__import__", side_effect=missing):
            self.assertIsNone(make_model("xgboost", config()))


@unittest.skipUnless(importlib.util.find_spec("sqlalchemy"), "Install forecast-worker extra")
class WorkerStorageTests(unittest.TestCase):
    def setUp(self):
        from database.forecast_store import ForecastStore
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.store = ForecastStore("sqlite:///:memory:")
        self.store.create_schema()

    def tearDown(self):
        self.store.engine.dispose()
        self.temp.cleanup()

    def test_failed_gate_logs_run_without_forecasts_or_model(self):
        from database.forecast_store import forecasts, model_runs
        from sqlalchemy import select
        with patch("ml.forecasting.worker.pull", return_value=records(units=np.full(48, 100))):
            result = run(config(), self.store, self.root)
        self.assertEqual(result["status"], "rejected")
        self.assertFalse(list(self.root.rglob("*.joblib")))
        with self.store.engine.connect() as conn:
            self.assertEqual(len(conn.execute(select(model_runs)).all()), 1)
            self.assertEqual(conn.execute(select(forecasts)).all(), [])

    def test_unknown_load_cannot_pull_or_access_another_tenant(self):
        self.store.save_raw("other", "private_load", records())
        with patch("ml.forecasting.worker.pull") as pull:
            with self.assertRaisesRegex(ValueError, "No preserved"):
                run(config(), self.store, self.root, load_id="private_load")
            pull.assert_not_called()

    def test_training_error_is_recorded(self):
        from database.forecast_store import model_runs
        from sqlalchemy import select
        with patch("ml.forecasting.worker.pull", return_value=records(12)):
            with self.assertRaisesRegex(ValueError, "at least"):
                run(config(), self.store, self.root)
        with self.store.engine.connect() as conn:
            row = conn.execute(select(model_runs)).mappings().one()
            self.assertEqual(row["status"], "failed")
            self.assertEqual(row["metadata"]["error_type"], "ValueError")

    def test_tenant_isolation_idempotency_and_regional_accuracy(self):
        source = records()
        for tenant in ("one", "two"):
            self.store.save_raw(tenant, "load", source)
            clean, quality = clean_sales(source, config())
            for _ in range(2):
                self.store.save_clean(tenant, "load", clean, quality)
        self.assertEqual(len(self.store.read_sales("one", "load")), 48)
        self.assertEqual(self.store.read_raw("two", "load"), source)
        # Same product/date, different region must not multiply accuracy rows.
        clean = pd.concat([clean, clean.assign(region="US", units_sold=9999)], ignore_index=True)
        self.store.save_clean("one", "load", clean, quality)
        prediction = pd.DataFrame([{"sale_date": pd.Timestamp("2022-01-01"), "product_id": "001",
                                    "region": "IN", "predicted_units": 110.0}])
        for version in ("v1", "v2"):
            self.store.save_run("one", "load", version, {"passed": True}, prediction)
        accuracy = self.store.accuracy("one")
        self.assertEqual(len(accuracy["rows"]), 2)
        self.assertEqual(accuracy["versions"][0]["mae"], 10)
        self.assertEqual(self.store.accuracy("two")["rows"], [])

    def test_quality_failure_preserves_raw_and_previous_clean_snapshot(self):
        from ml.forecasting.worker import clean_load
        source = records()
        self.store.save_raw("demo", "good", source)
        clean_load(config(), self.store, "good")
        broken = records()
        broken[10]["Units_Sold"] = "bad"
        self.store.save_raw("demo", "bad", broken)
        with self.assertRaisesRegex(ValueError, "quality gate"):
            clean_load(config(), self.store, "bad")
        self.assertEqual(self.store.read_raw("demo", "bad"), broken)
        self.assertEqual(len(self.store.read_sales("demo", "good")), 48)

    def test_successful_worker_saves_versioned_model_and_forecasts(self):
        path = self.root / "demo.csv"
        demo(path)
        cfg = config(source={"type": "csv", "path": str(path)},
                     candidate_models=("random_forest", "hist_gb", "xgboost"))
        result = run(cfg, self.store, self.root / "models")
        self.assertTrue(result["passed"])
        model_path = Path(result["model_path"])
        self.assertTrue(model_path.exists())
        bundle = load_model(model_path, "demo")
        self.assertEqual(bundle["model_version"], result["run_id"])
        with self.assertRaisesRegex(ValueError, "another tenant"):
            load_model(model_path, "other")
        self.assertEqual(result["forecast_rows"], 12)
        # Predictions saved with the actual fitted preprocessing pipeline.
        self.assertIn("pre", bundle["model"].named_steps)


if __name__ == "__main__":
    unittest.main()
