"""Read-only monthly forecast API checks using temporary artifacts."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import pandas as pd

import test_predictions


class MonthlyPredictionAPITests(unittest.TestCase):
    """Exercise monthly filters and missing/corrupt artifact responses."""

    def setUp(self):
        """Use isolated forecast files rather than the real trained model outputs."""
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.frame = pd.DataFrame([
            {"Month": "2026-01-01", "Product_ID": "P1", "Country_ID": "C1", "Predicted_Units": 200.5, "Lower": 180, "Upper": 220},
            {"Month": "2026-02-01", "Product_ID": "P1", "Country_ID": "C1", "Predicted_Units": 210.5, "Lower": 190, "Upper": 230},
            {"Month": "2026-01-01", "Product_ID": "P2", "Country_ID": "C2", "Predicted_Units": 100, "Lower": 80, "Upper": 120},
        ])
        self.frame.to_csv(self.root / "forecast_2026.csv", index=False)
        (self.root / "run_metadata.json").write_text(json.dumps({"best_model": "XGBoost", "final_training_period": "2023-2025", "ranges": "Approximate ranges"}))
        patcher = patch("ml.monthly_predictions.FORECAST_DIR", self.root)
        patcher.start()
        self.addCleanup(patcher.stop)

    def request(self, query=""):
        """Call the registered FastAPI endpoint through the existing ASGI helper."""
        return test_predictions.PredictionTests().request_prediction("/predictions/monthly" + query)

    def test_filters_and_totals(self):
        """Product/country and month filters return only matching forecast rows."""
        status, body = self.request("?product_id=P1&country_id=C1")
        self.assertEqual(status, 200)
        self.assertEqual(body["count"], 2)
        self.assertEqual(body["total_predicted_units"], 411)
        self.assertEqual(body["model"], "XGBoost")
        status, body = self.request("?product_id=P1&country_id=C1&month=1")
        self.assertEqual(status, 200)
        self.assertEqual(body["count"], 1)
        self.assertEqual(body["forecasts"][0]["Predicted_Units"], 200.5)

    def test_unknown_filter_and_invalid_month(self):
        """Unknown IDs return 404 and invalid calendar months return 422."""
        self.assertEqual(self.request("?product_id=UNKNOWN")[0], 404)
        self.assertEqual(self.request("?month=13")[0], 422)

    def test_missing_and_corrupt_artifacts(self):
        """Missing files and non-finite numbers must not return misleading predictions."""
        self.frame["Upper"] = self.frame["Upper"].astype(float)
        self.frame.loc[0, "Upper"] = float("inf")
        self.frame.to_csv(self.root / "forecast_2026.csv", index=False)
        with patch("routes.predictions.logger.exception"):
            self.assertEqual(self.request()[0], 503)
        (self.root / "forecast_2026.csv").unlink()
        self.assertEqual(self.request()[0], 503)
