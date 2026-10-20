"""React ML API, live recursive inference, CORS and training activation tests."""

import asyncio
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from app.main import app
from ml import monthly_predictions as stored
from routes.monthly_ml import LivePredictionRequest
from services import monthly_ml
import test_predictions
from test_monthly_demand_forecast import CapturingModel, make_panel


class MonthlyMLTests(unittest.TestCase):
    """Check API contracts and inference without remote database writes."""

    def test_live_inference_recursion_and_what_if(self):
        """February's lag uses predicted January and overrides reach model inputs."""
        model = CapturingModel()
        bundle = {"model_type": "XGBoost", "model": model, "history": make_panel()}
        with patch.object(monthly_ml, "load_bundle", return_value=bundle):
            result = monthly_ml.live_prediction(LivePredictionRequest(
                product_id="P0", country_id="C0", month=2, selling_price=222, discount=7,
            ))
        self.assertEqual(result["source"], "live_model_inference")
        self.assertEqual(result["forecast_path"][0]["Predicted_Units"], 36)
        self.assertEqual(result["prediction"]["Predicted_Units"], 37)
        self.assertEqual(result["prediction"]["inputs"]["lag_1"], 36)
        self.assertEqual(result["prediction"]["inputs"]["Selling_Price"], 222)
        self.assertEqual(result["prediction"]["inputs"]["Discount"], 7)

    def test_api_validation_and_unknown_ids(self):
        """Bad scenario parameters return 422; unseen combinations return 404."""
        helper = test_predictions.PredictionTests()
        for payload in (
            {"product_id": "P1", "country_id": "C1", "month": 13},
            {"product_id": "P1", "country_id": "C1", "discount": 101},
            {"product_id": "P1", "country_id": "C1", "selling_price": -1},
        ):
            self.assertEqual(helper.request_prediction("/predictions/monthly/predict", body=payload)[0], 422)
        with patch.object(monthly_ml, "load_bundle", return_value={"history": make_panel()}):
            self.assertEqual(helper.request_prediction("/predictions/monthly/predict", body={"product_id": "UNKNOWN", "country_id": "C1"})[0], 404)

    def test_training_lock_and_failure_preserve_active_run(self):
        """Failed/duplicate training cannot replace active forecasts."""
        with TemporaryDirectory() as directory, patch.object(stored, "FORECAST_DIR", Path(directory)):
            pointer = Path(directory) / "active_run.json"
            pointer.write_text(json.dumps({"run_id": "a" * 32}))
            with patch("ml.monthly_demand_forecast.main", return_value=1):
                with self.assertRaises(RuntimeError):
                    monthly_ml.train_monthly()
            self.assertEqual(json.loads(pointer.read_text())["run_id"], "a" * 32)
            monthly_ml.TRAINING_LOCK.acquire()
            try:
                with self.assertRaises(BlockingIOError):
                    monthly_ml.train_monthly()
            finally:
                monthly_ml.TRAINING_LOCK.release()

    def test_training_success_activates_new_run(self):
        """Completed run gets a new active pointer and a usable API response."""
        with TemporaryDirectory() as directory, patch.object(stored, "FORECAST_DIR", Path(directory)), patch("ml.monthly_demand_forecast.main", return_value=0), patch.object(monthly_ml, "model_details", return_value={"model": "XGBoost"}):
            response = monthly_ml.train_monthly()
            self.assertEqual(response["status"], "completed")
            self.assertEqual(stored.artifact_directory(), Path(directory) / "runs" / response["run_id"])

    def test_cors_preflight(self):
        """Vite React's JSON POST preflight is accepted with the correct origin."""
        messages = []

        async def request():
            """Issue an OPTIONS request directly against the app."""
            async def receive():
                """Provide an empty HTTP body."""
                return {"type": "http.request", "body": b"", "more_body": False}

            async def send(message):
                """Collect ASGI response messages."""
                messages.append(message)

            await app({"type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1",
                       "method": "OPTIONS", "scheme": "http", "path": "/predictions/monthly/predict",
                       "query_string": b"", "root_path": "", "headers": [
                           (b"origin", b"http://localhost:5173"),
                           (b"access-control-request-method", b"POST"),
                           (b"access-control-request-headers", b"content-type")],
                       "server": ("testserver", 80), "client": ("127.0.0.1", 1234)}, receive, send)

        asyncio.run(request())
        start = next(item for item in messages if item["type"] == "http.response.start")
        self.assertEqual(start["status"], 200)
        self.assertEqual(dict(start["headers"])[b"access-control-allow-origin"], b"http://localhost:5173")

    def test_missing_runtime_dependency_is_actionable(self):
        """Missing XGBoost must identify the dependency instead of a generic failure."""
        with patch.object(monthly_ml, "model_details", side_effect=ModuleNotFoundError("missing xgboost", name="xgboost")), patch("routes.monthly_ml.logger.exception"):
            status, body = test_predictions.PredictionTests().request_prediction("/predictions/monthly/details")
        self.assertEqual(status, 503)
        self.assertEqual(body["detail"]["code"], "monthly_dependency_missing")
        self.assertEqual(body["detail"]["dependency"], "xgboost")

    def test_vercel_training_is_explicitly_unsupported(self):
        """Serverless inference must not attempt persistent training writes."""
        with patch.dict("os.environ", {"VERCEL": "1"}), patch.object(monthly_ml, "train_monthly") as train:
            status, body = test_predictions.PredictionTests().request_prediction("/predictions/monthly/train", body={})
        self.assertEqual(status, 501)
        train.assert_not_called()
