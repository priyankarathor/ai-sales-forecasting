import asyncio
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import joblib
import pandas as pd
from sklearn.dummy import DummyRegressor

from app.main import app
from ml.features import FEATURE_COLUMNS, create_features
from ml.predict import predict_next_day, predict_periods
from ml.train import train_model
from ml.preprocessing import prepare_daily_sales, prepare_model_series


class PredictionTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.model_path = Path(self.directory.name) / "sales_model.joblib"
        self.sales = pd.DataFrame({
            "sale_date": pd.date_range("2026-09-01", periods=14),
            "product_id": "P001",
            "region": "Delhi",
            "quantity_sold": range(1, 15),
        })
        model = DummyRegressor(strategy="constant", constant=12.345)
        model.fit(pd.DataFrame([[1, 1, 1, 1]], columns=FEATURE_COLUMNS), [1])
        self.artifact = {
            "model": model,
            "product_id": "P001",
            "region": "Delhi",
            "feature_columns": FEATURE_COLUMNS,
            "trained_through": "2026-09-14T00:00:00",
        }
        joblib.dump(self.artifact, self.model_path)
        self.path_patch = patch("ml.predict.MODEL_PATH", self.model_path)
        self.path_patch.start()
        self.addCleanup(self.path_patch.stop)
        self.loader_patch = patch("ml.predict.load_sales", return_value=self.sales)
        self.loader = self.loader_patch.start()
        self.addCleanup(self.loader_patch.stop)

    def request_prediction(self, path="/predictions/next-day", *, html=False, body=None):
        messages = []
        endpoint, _, query = path.partition("?")
        payload = b"" if body is None else json.dumps(body).encode()

        async def request():
            async def receive():
                return {"type": "http.request", "body": payload, "more_body": False}

            async def send(message):
                messages.append(message)

            await app({
                "type": "http", "asgi": {"version": "3.0"},
                "http_version": "1.1", "method": "GET" if body is None else "POST", "scheme": "http",
                "path": endpoint, "query_string": query.encode(),
                "headers": [] if body is None else [(b"content-type", b"application/json")], "server": ("testserver", 80),
                "client": ("127.0.0.1", 1234), "root_path": "",
            }, receive, send)

        asyncio.run(request())
        status = next(m["status"] for m in messages if m["type"] == "http.response.start")
        body = b"".join(m.get("body", b"") for m in messages if m["type"] == "http.response.body")
        return status, body.decode() if html else json.loads(body)

    def test_dashboard_page_is_served(self):
        status, body = self.request_prediction("/dashboard", html=True)
        self.assertEqual(status, 200)
        self.assertIn("/predictions/dashboard-data", body)

    def test_dashboard_data_has_history_features_and_evaluation(self):
        self.artifact["evaluation"] = {
            "model_mae": 2.0, "baseline_mae": 3.0,
            "comparison": [{"date": "2026-09-14", "actual_units": 14.0,
                            "predicted_units": 12.0, "baseline_units": 7.0}],
        }
        joblib.dump(self.artifact, self.model_path)
        status, result = self.request_prediction("/predictions/dashboard-data")
        self.assertEqual(status, 200)
        self.assertEqual(len(result["history"]), 14)
        self.assertEqual(result["history"][-1]["units_sold"], 14)
        self.assertEqual(result["features"]["lag_1"], 14)
        self.assertEqual(result["evaluation"], self.artifact["evaluation"])

    def test_dashboard_supports_old_artifacts_and_history_gaps(self):
        self.loader.return_value = self.sales.drop(index=2)
        status, result = self.request_prediction("/predictions/dashboard-data")
        self.assertEqual(status, 200)
        self.assertIsNone(result["history"][2]["units_sold"])
        self.assertEqual(result["evaluation"], {})

    def test_dashboard_missing_model_returns_503(self):
        self.model_path.unlink()
        status, result = self.request_prediction("/predictions/dashboard-data")
        self.assertEqual(status, 503)
        self.assertIn("training", result["detail"])

    def test_training_saves_held_out_comparison_for_dashboard(self):
        sales = pd.DataFrame({
            "sale_date": pd.date_range("2026-07-01", periods=60),
            "product_id": "P001", "region": "Delhi",
            "quantity_sold": range(1, 61),
        })
        model = Mock()
        model.predict.side_effect = lambda features: features["lag_1"].to_numpy() + 1
        fake_train_path = Path(self.directory.name) / "ml" / "train.py"
        with patch("ml.train.load_sales", return_value=sales), patch(
            "ml.train.RandomForestRegressor", return_value=model
        ), patch("ml.train.__file__", str(fake_train_path)), patch(
            "ml.train.save_artifact"
        ) as save, patch("builtins.print"):
            train_model("P001", "Delhi", model_type="random_forest")
        artifact = save.call_args.args[0]
        evaluation = artifact["evaluation"]
        self.assertEqual(evaluation["train_days"], 43)
        self.assertEqual(evaluation["test_days"], 10)
        self.assertEqual(len(evaluation["comparison"]), 10)
        self.assertEqual(evaluation["comparison"][-1]["actual_units"], 60)
        self.assertEqual(evaluation["comparison"][-1]["date"], "2026-08-29")
        self.assertEqual(model.fit.call_count, 2)
        json.dumps(evaluation, allow_nan=False)

    def test_weekly_training_saves_baseline_and_predicts_next_observation(self):
        sales = pd.DataFrame({
            "sale_date": pd.date_range("2025-01-06", periods=50, freq="7D"),
            "product_id": "P001", "region": "Delhi", "quantity_sold": 4,
        })
        fake_train_path = Path(self.directory.name) / "ml" / "train.py"
        with patch("ml.train.load_sales", return_value=sales), patch(
            "ml.train.__file__", str(fake_train_path)
        ), patch("ml.train.isolated_model_path", return_value=Path(self.directory.name) / "weekly.joblib"
        ), patch("builtins.print"):
            train_model("P001", "Delhi", model_type="auto")
        model_path = Path(self.directory.name) / "artifacts" / "sales_model.joblib"
        artifact = joblib.load(model_path)
        self.assertEqual(artifact["frequency"], "7D")
        self.assertEqual(artifact["model_type"], "seven_period_baseline")
        self.assertEqual(artifact["evaluation"]["model_mae"], 0)
        self.loader.return_value = sales
        with patch("ml.predict.MODEL_PATH", model_path):
            status, result = self.request_prediction("/predictions/next-period")
            self.assertEqual(status, 200)
            self.assertEqual(result["forecast_horizon_days"], 7)
            self.assertEqual(result["predicted_units"], 4)
            self.assertEqual(result["forecast_date"], "2025-12-22")
            status, result = self.request_prediction("/predictions/next-day")
            self.assertEqual(status, 422)
            status, result = self.request_prediction("/predictions/dashboard-data")
            self.assertEqual(status, 200)
            self.assertEqual(len(result["history"]), 50)
            self.assertEqual(result["features"]["rolling_mean_7"], 4)

    def test_weekly_inference_preserves_missing_weeks(self):
        sales = self.sales.copy()
        sales["sale_date"] = pd.date_range("2026-01-05", periods=14, freq="7D")
        sales = sales.drop(index=10)
        daily = prepare_daily_sales(sales, "P001", "Delhi")
        weekly, frequency = prepare_model_series(daily)
        self.assertEqual(frequency, "7D")
        self.assertEqual(len(weekly), 14)
        self.assertTrue(pd.isna(weekly.iloc[10]["units_sold"]))
        with self.assertRaisesRegex(ValueError, "Missing sales"):
            create_features(weekly, frequency=frequency)

    def test_weekly_model_rejects_off_schedule_observations(self):
        daily = prepare_daily_sales(self.sales, "P001", "Delhi")
        with self.assertRaisesRegex(ValueError, "no longer match"):
            prepare_model_series(daily, "7D")

    def test_recursive_forecast_feeds_predictions_into_future_features(self):
        model = Mock()
        model.predict.side_effect = lambda frame: [frame.iloc[0]["lag_1"] + 1]
        self.artifact["model"] = model
        with patch("ml.predict.joblib.load", return_value=self.artifact):
            result = predict_periods(periods=9)
        self.assertEqual([r["predicted_units"] for r in result["forecasts"]], list(range(15, 24)))
        self.assertEqual(result["total_predicted_units"], sum(range(15, 24)))
        last_input = model.predict.call_args_list[-1].args[0].iloc[0]
        self.assertEqual(last_input["lag_1"], 22)
        self.assertEqual(last_input["lag_7"], 16)
        self.assertEqual(last_input["rolling_mean_7"], 19)
        self.assertEqual(result["forecast_end_date"], "2026-09-23")

    def test_weekly_future_forecast_dates_and_totals(self):
        self.sales["sale_date"] = pd.date_range("2026-06-01", periods=14, freq="7D")
        self.artifact.update(frequency="7D", trained_through="2026-08-31")
        joblib.dump(self.artifact, self.model_path)
        status, result = self.request_prediction("/predictions/future?periods=4")
        self.assertEqual(status, 200)
        self.assertEqual([row["date"] for row in result["forecasts"]], [
            "2026-09-07", "2026-09-14", "2026-09-21", "2026-09-28",
        ])
        self.assertEqual(result["total_predicted_units"], 49.4)
        self.assertEqual(result["forecast_horizon_days"], 7)

    def test_next_calendar_month_rolls_forward_through_intervening_dates(self):
        model = Mock()
        model.predict.side_effect = lambda frame: [frame.iloc[0]["lag_1"] + 1]
        self.artifact["model"] = model
        with patch("ml.predict.joblib.load", return_value=self.artifact):
            result = predict_periods(calendar_month=True)
        self.assertEqual(result["calendar_month"], "2026-10")
        self.assertEqual(result["periods"], 31)
        self.assertEqual(result["forecast_date"], "2026-10-01")
        self.assertEqual(result["forecast_end_date"], "2026-10-31")
        self.assertEqual(result["predicted_units"], 31)
        self.assertEqual(model.predict.call_count, 47)

    def test_calendar_month_handles_leap_year(self):
        self.sales["sale_date"] = pd.date_range("2024-01-18", periods=14)
        self.artifact["trained_through"] = "2024-01-31"
        joblib.dump(self.artifact, self.model_path)
        result = predict_periods(calendar_month=True)
        self.assertEqual(result["periods"], 29)
        self.assertEqual(result["forecast_end_date"], "2024-02-29")

    def test_weekly_month_uses_observation_date_buckets(self):
        self.sales["sale_date"] = pd.date_range("2026-06-01", periods=14, freq="7D")
        self.artifact.update(frequency="7D", trained_through="2026-08-31")
        joblib.dump(self.artifact, self.model_path)
        result = predict_periods(calendar_month=True)
        self.assertEqual(result["calendar_month"], "2026-09")
        self.assertEqual(result["periods"], 4)
        self.assertIn("not prorated", result["interpretation"])
        self.assertEqual(result["forecast_end_date"], "2026-09-28")

    def test_future_period_limits_are_validated(self):
        for count in (0, 91):
            status, result = self.request_prediction(f"/predictions/future?periods={count}")
            self.assertEqual(status, 422)
        for count in (0, 91, True, 2.5):
            with self.assertRaises(ValueError):
                predict_periods(periods=count)

    def production_body(self, **changes):
        return {
            "product_id": "P001", "region": "Delhi",
            "periods": 4, "units_are_period_totals": True,
            "stock_as_of": "2026-09-14", "current_stock": 10,
            "safety_stock": 5, "incoming_stock": 4, **changes,
        }

    def test_production_plan_rounds_up_and_reports_capacity_shortfall(self):
        status, result = self.request_prediction(
            "/predictions/production-plan", body=self.production_body(available_capacity=30),
        )
        self.assertEqual(status, 200)
        self.assertEqual(result["forecast"]["total_predicted_units"], 49.4)
        self.assertEqual(result["required_production_units"], 41)
        self.assertEqual(result["capacity_shortfall_units"], 11)

    def test_production_does_not_go_negative_and_capacity_is_optional(self):
        status, result = self.request_prediction(
            "/predictions/production-plan", body=self.production_body(current_stock=100),
        )
        self.assertEqual(status, 200)
        self.assertEqual(result["required_production_units"], 0)
        self.assertIsNone(result["capacity_shortfall_units"])

    def test_production_requires_matching_inventory_date_and_confirmed_totals(self):
        for changes in (
            {"stock_as_of": "2026-09-15"}, {"units_are_period_totals": False},
            {"current_stock": -1}, {"safety_stock": -1}, {"incoming_stock": -1},
            {"available_capacity": -1}, {"calendar_month": True},
            {"product_id": "different"}, {"region": "different"},
        ):
            with self.subTest(changes=changes):
                status, result = self.request_prediction(
                    "/predictions/production-plan", body=self.production_body(**changes),
                )
                self.assertEqual(status, 422)

    def test_future_missing_model_returns_503(self):
        self.model_path.unlink()
        status, result = self.request_prediction("/predictions/future?periods=4")
        self.assertEqual(status, 503)

    def test_prediction_from_saved_model(self):
        result = predict_next_day()
        self.assertEqual(result["forecast_date"], "2026-09-15")
        self.assertEqual(result["last_sales_date"], "2026-09-14")
        self.assertEqual(result["predicted_units"], 12.35)
        self.assertEqual(result["product_id"], "P001")
        self.loader.assert_called_once_with(product_id="P001", region="Delhi")

    def test_post_forecast_uses_selected_model_and_requested_days(self):
        self.artifact.update(model_type="linear_regression")
        joblib.dump(self.artifact, self.model_path)
        with patch("ml.predict.model_path", return_value=self.model_path) as selected:
            status, result = self.request_prediction("/predictions/future", body={
                "product_id": "P001", "region": "Delhi", "days": 9,
                "include_details": True,
            })
        self.assertEqual(status, 200)
        selected.assert_called_once_with("P001", "Delhi", "linear_regression")
        self.assertEqual(result["periods"], 9)
        self.assertEqual(result["forecast_window_days"], 9)
        self.assertEqual(len(result["history"]), 14)

    def test_post_weekly_days_and_invalid_parameters(self):
        self.sales["sale_date"] = pd.date_range("2026-06-01", periods=14, freq="7D")
        self.artifact.update(frequency="7D", trained_through="2026-08-31", model_type="linear_regression")
        joblib.dump(self.artifact, self.model_path)
        body = {"product_id": "P001", "region": "Delhi", "days": 28}
        with patch("ml.predict.model_path", return_value=self.model_path):
            status, result = self.request_prediction("/predictions/future", body=body)
            self.assertEqual(status, 200)
            self.assertEqual(result["periods"], 4)
            for changes in ({"days": 30}, {"days": True}, {"days": 0}, {"days": 91},
                            {"days": 2.5}, {"days": "7"}, {"region": " "},
                            {"model_type": "unknown"}, {"discount": 10}, {"region": "Other"}):
                with self.subTest(changes=changes):
                    status, _ = self.request_prediction("/predictions/future", body={**body, **changes})
                    self.assertEqual(status, 422)

    def test_post_missing_selected_model_never_falls_back_to_another_product(self):
        with patch("ml.predict.model_path", return_value=Path(self.directory.name) / "missing.joblib"):
            status, _ = self.request_prediction("/predictions/future", body={
                "product_id": "Other", "region": "Delhi", "days": 7,
            })
        self.assertEqual(status, 503)
        self.loader.assert_not_called()

    def test_linear_regression_pipeline_trains_saves_and_forecasts_trend(self):
        sales = pd.DataFrame({
            "sale_date": pd.date_range("2026-01-01", periods=70),
            "product_id": "P001", "region": "Delhi", "quantity_sold": range(10, 80),
        })
        isolated = Path(self.directory.name) / "isolated.joblib"
        with patch("ml.train.load_sales", return_value=sales), patch(
            "ml.train.__file__", str(Path(self.directory.name) / "ml" / "train.py")
        ), patch("ml.train.isolated_model_path", return_value=isolated), patch("builtins.print"):
            result = train_model("P001", "Delhi")
        self.assertEqual(result["model_type"], "linear_regression")
        self.assertLess(result["evaluation"]["model_mae"], 1e-8)
        self.assertLess(result["evaluation"]["rmse"], 1e-8)
        artifact = joblib.load(isolated)
        self.assertIn("standardscaler", artifact["model"].named_steps)
        self.loader.return_value = sales
        with patch("ml.predict.model_path", return_value=isolated):
            forecast = predict_periods(product_id="P001", region="Delhi", days=7)
        self.assertEqual([row["predicted_units"] for row in forecast["forecasts"]], list(range(80, 87)))
        json.dumps(result, allow_nan=False)

    def test_training_post_validates_and_passes_parameters(self):
        with patch("routes.predictions.train_model", return_value={"model_type": "linear_regression"}) as train:
            status, _ = self.request_prediction("/predictions/train", body={
                "product_id": "101", "region": "South", "frequency": "7D", "test_fraction": 0.3,
            })
            self.assertEqual(status, 200)
            self.assertEqual(train.call_args.kwargs["test_fraction"], 0.3)
            self.assertEqual(train.call_args.kwargs["model_type"], "linear_regression")
        status, _ = self.request_prediction("/predictions/train", body={
            "product_id": "101", "region": "South", "test_fraction": 0.9,
        })
        self.assertEqual(status, 422)

    def test_prediction_features_match_training(self):
        model = Mock()
        model.predict.return_value = [12]
        self.artifact["model"] = model
        daily = pd.DataFrame({
            "date": pd.date_range("2026-09-01", periods=15),
            "units_sold": range(1, 16),
        })
        expected = create_features(daily).tail(1)[FEATURE_COLUMNS].reset_index(drop=True)
        with patch("ml.predict.joblib.load", return_value=self.artifact):
            predict_next_day()
        pd.testing.assert_frame_equal(
            model.predict.call_args.args[0], expected, check_dtype=False,
        )

    def test_endpoint_is_registered_and_returns_prediction(self):
        self.assertIn("/predictions/next-day", app.openapi()["paths"])
        status, result = self.request_prediction()
        self.assertEqual(status, 200)
        self.assertEqual(result["forecast_date"], "2026-09-15")

    def test_missing_model_returns_503(self):
        self.model_path.unlink()
        status, result = self.request_prediction()
        self.assertEqual(status, 503)
        self.assertIn("training", result["detail"])
        self.loader.assert_not_called()

    def test_missing_recent_day_returns_422(self):
        self.loader.return_value = self.sales.drop(index=10)
        status, result = self.request_prediction()
        self.assertEqual(status, 422)
        self.assertIn("missing sales", result["detail"])

    def test_too_little_history(self):
        self.loader.return_value = self.sales.tail(6)
        with self.assertRaisesRegex(ValueError, "seven days"):
            predict_next_day()

    def test_history_ending_before_training_is_rejected(self):
        self.loader.return_value = self.sales.iloc[:-1]
        with self.assertRaisesRegex(ValueError, "training date"):
            predict_next_day()

    def test_invalid_model_output_returns_500_without_internal_details(self):
        for output in (float("nan"), float("inf"), -1):
            with self.subTest(output=output):
                model = Mock()
                model.predict.return_value = [output]
                self.artifact["model"] = model
                with patch("ml.predict.joblib.load", return_value=self.artifact), patch("routes.predictions.logger"):
                    status, result = self.request_prediction()
                self.assertEqual(status, 500)
                self.assertEqual(result["detail"], "Prediction failed. Check server logs.")


if __name__ == "__main__":
    unittest.main()
