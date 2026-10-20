"""Deterministic checks for chronology, grouped features, metrics and forecast ranges."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import pandas as pd

from ml import monthly_demand_forecast as demand


def make_panel():
    """Build a complete 1,800-row panel without external services."""
    rows = []
    for product in range(5):
        for country in range(10):
            for index, month in enumerate(pd.date_range("2023-01-01", periods=36, freq="MS")):
                rows.append({
                    "Month": month, "Product_ID": f"P{product}", "Country_ID": f"C{country}",
                    "Units_Sold": float(100 * product + 10 * country + index),
                    "Selling_Price": float(100 + index), "Discount": 10.0,
                    "RM_Cost": 5.0, "Shipping": 2.0, "Profit_per_unit": 10.0,
                    "Month_Number": month.month, "Is_Festival_Month": int(month.month in (10, 11)),
                })
    return pd.DataFrame(rows)


class CapturingModel:
    """Predict prior demand plus one and capture recursive inputs."""

    def __init__(self):
        """Store inputs for chronology assertions."""
        self.inputs = []

    def predict(self, features):
        """Use only the lag_1 feature."""
        self.inputs.append(features.copy())
        return features["lag_1"].to_numpy() + 1


class MonthlyDemandTests(unittest.TestCase):
    """Check failure cases and forecasting semantics with known values."""

    def test_lags_and_rolling_are_series_local_and_past_only(self):
        """Jan 2024 must use only 2023 observations for each combination."""
        frame = demand.engineer_features(make_panel())
        row = frame.query("Product_ID == 'P1' and Country_ID == 'C2'").iloc[0]
        self.assertEqual(row["Month"], pd.Timestamp("2024-01-01"))
        self.assertEqual(row["lag_1"], 131)
        self.assertEqual(row["lag_3"], 129)
        self.assertEqual(row["lag_12"], 120)
        self.assertEqual(row["rolling_mean_3"], 130)
        self.assertEqual(row["rolling_mean_6"], 128.5)

    def test_recursive_predictions_and_origin_price_assumptions(self):
        """February uses predicted January; prices use only origin history."""
        history = make_panel().query("Month < '2025-01-01'")
        model = CapturingModel()
        forecast = demand.recursive_tree_forecast(model, history, pd.date_range("2025-01-01", periods=12, freq="MS"))
        self.assertEqual(len(forecast), 600)
        np.testing.assert_allclose(model.inputs[1]["lag_1"], model.inputs[0]["lag_1"] + 1)
        np.testing.assert_allclose(model.inputs[0]["Selling_Price"], 117.5)
        self.assertTrue(model.inputs[-1]["Is_Festival_Month"].eq(0).all())
        self.assertTrue(model.inputs[9]["Is_Festival_Month"].eq(1).all())

    def test_target_change_cannot_affect_earlier_features(self):
        """Changing future targets leaves all earlier engineered rows identical."""
        original = make_panel()
        changed = original.copy()
        changed.loc[changed["Month"] >= "2025-01-01", "Units_Sold"] = 999999
        a = demand.engineer_features(original).query("Month < '2025-01-01'")
        b = demand.engineer_features(changed).query("Month < '2025-01-01'")
        pd.testing.assert_frame_equal(a, b)

    def test_cleaning_rejects_duplicates_and_missing_months(self):
        """Incomplete history must fail rather than fabricate zero demand."""
        panel = make_panel()
        with TemporaryDirectory() as directory:
            path = Path(directory) / "panel.csv"
            panel.to_csv(path, index=False)
            self.assertEqual(len(demand.load_and_clean(path)), 1800)
            panel.iloc[1:].to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "36 months"):
                demand.load_and_clean(path)
            pd.concat([panel, panel.iloc[:1]]).to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                demand.load_and_clean(path)

    def test_demand_training_does_not_require_cost_data(self):
        """Sales alone produce the same features without inventing missing costs."""
        panel = make_panel()
        sales = panel[["Month", *demand.KEYS, "Units_Sold", *demand.PRICE_FEATURES]]
        cleaned = demand.clean_sales_frame(sales)
        self.assertNotIn("Profit_per_unit", cleaned)
        self.assertNotIn("RM_Cost", cleaned)
        pd.testing.assert_frame_equal(
            demand.engineer_features(cleaned)[demand.FEATURES],
            demand.engineer_features(panel)[demand.FEATURES], check_dtype=False,
        )

    def test_demand_training_does_not_require_cost_data(self):
        """Sales alone produce the same features without inventing missing costs."""
        panel = make_panel()
        sales = panel[["Month", *demand.KEYS, "Units_Sold", *demand.PRICE_FEATURES]]
        cleaned = demand.clean_sales_frame(sales)
        self.assertNotIn("Profit_per_unit", cleaned)
        self.assertNotIn("RM_Cost", cleaned)
        pd.testing.assert_frame_equal(
            demand.engineer_features(cleaned)[demand.FEATURES],
            demand.engineer_features(panel)[demand.FEATURES], check_dtype=False,
        )

    def test_zero_sales_metrics_and_ranges(self):
        """MAPE excludes zeros explicitly; MAE/RMSE retain them."""
        actual = make_panel().iloc[:2].copy()
        actual["Units_Sold"] = [0.0, 10.0]
        forecast = actual[["Month", *demand.KEYS]].assign(Predicted_Units=[2.0, 8.0])
        metrics, backtest = demand.score_predictions(actual, forecast)
        self.assertEqual(metrics["MAPE_Pct"], 20)
        self.assertEqual(metrics["MAPE_Excluded_Zero_Rows"], 1)
        self.assertEqual(metrics["RMSE"], 2)
        self.assertEqual(metrics["MAE"], 2)
        future = forecast.copy()
        future["Month"] += pd.DateOffset(years=3)
        output = demand.add_prediction_ranges(future, backtest)
        self.assertTrue(output["Lower"].le(output["Predicted_Units"]).all())
        self.assertTrue(output["Upper"].ge(output["Predicted_Units"]).all())
        self.assertEqual(list(output.columns), demand.OUTPUT_COLUMNS)


if __name__ == "__main__":
    unittest.main()
