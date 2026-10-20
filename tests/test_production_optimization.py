"""Known-optimum LP, cost conversion, FX and sensitivity tests."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from bson import BSON
import numpy as np
import pandas as pd

from ml import production_optimization as planning


def fixture_panel():
    """Return a small LP with independently verifiable resource tradeoffs."""
    return pd.DataFrame({
        "Product_ID": ["P1", "P1", "P2", "P2"],
        "Country_ID": ["C1", "C2", "C1", "C2"],
        "Annual_Demand": [1000.0] * 4,
        "Profit_per_unit": [10.0, -1.0, 8.0, -2.0],
        "Baseline_2025_Units": [10.0] * 4,
        "Annual_Capacity": [1200.0] * 4,
        "Labour_hours_per_unit": [1.0, 1.0, 2.0, 2.0],
        "RM_Cost_INR": [2.0, 2.0, 1.0, 1.0],
        "USD_EUR_RM_INR": [2.0, 2.0, 1.0, 1.0],
    })


class ProductionOptimizationTests(unittest.TestCase):
    """Validate optimal allocations, economic units and import-cost assumptions."""

    def test_labour_binding_and_shadow_sign(self):
        """A labour hour is worth INR 10 when P1 has the highest hourly margin."""
        allocation, sensitivity, summary = planning.solve_scenario(fixture_panel(), "Base", labour_hours=100, rm_budget=1000)
        self.assertAlmostEqual(summary["Total_Profit_INR"], 1000)
        np.testing.assert_allclose(allocation.Recommended_Units, [100, 0, 0, 0])
        labour = sensitivity.set_index("Constraint").loc["Labour"]
        self.assertAlmostEqual(labour.Shadow_Price, 10)
        self.assertAlmostEqual(labour.Resolved_1pct_Gain_INR, 10)
        _, _, relaxed = planning.solve_scenario(fixture_panel(), "Base", labour_hours=101, rm_budget=1000)
        self.assertAlmostEqual(relaxed["Total_Profit_INR"] - summary["Total_Profit_INR"], labour.Shadow_Price)

    def test_rm_budget_and_fx_change_both_cost_and_profit(self):
        """With budget binding, P2 wins per rupee and USD inflation reduces profit."""
        base, sensitivity, summary = planning.solve_scenario(fixture_panel(), "Base", labour_hours=10000, rm_budget=100)
        fx, _, stressed = planning.solve_scenario(fixture_panel(), "USD_EUR_plus_5pct", labour_hours=10000, rm_budget=100)
        self.assertAlmostEqual(summary["Total_Profit_INR"], 800)
        self.assertAlmostEqual(base.iloc[2].Recommended_Units, 100)
        self.assertAlmostEqual(sensitivity.set_index("Constraint").loc["RM_Budget", "Shadow_Price"], 8)
        self.assertAlmostEqual(fx.iloc[2].Scenario_RM_Cost, 1.05)
        self.assertAlmostEqual(fx.iloc[2].Scenario_Profit_per_unit, 7.95)
        self.assertAlmostEqual(stressed["Total_Profit_INR"], 100 / 1.05 * 7.95)
        self.assertLess(stressed["Total_Profit_INR"], summary["Total_Profit_INR"])

    def test_demand_reduction_and_capacity(self):
        """Demand and product capacities constrain allocations separately."""
        panel = fixture_panel()
        panel.loc[panel.Product_ID == "P1", "Annual_Capacity"] = 40
        panel.loc[2, "Annual_Demand"] = 10
        allocation, _, _ = planning.solve_scenario(panel, "Demand_minus_10pct", labour_hours=1000, rm_budget=1000)
        np.testing.assert_allclose(allocation.Recommended_Units, [40, 0, 9, 0])

    def test_break_even_does_not_claim_growth(self):
        """Zero/negative margins must not invent a profitable expansion."""
        panel = fixture_panel()
        panel["Profit_per_unit"] = [0, -1, 0, -2]
        allocation, _, summary = planning.solve_scenario(panel, "Base")
        self.assertEqual(summary["Total_Profit_INR"], 0)
        _, _, recommendation = planning.summarize_plan(allocation)
        self.assertIn("support nahi", recommendation)

    def test_bom_currency_conversion_and_missing_price(self):
        """RM currency comes from raw_materials; only USD/EUR cost is shocked."""
        panel = fixture_panel().drop(columns=["Annual_Capacity", "Labour_hours_per_unit", "RM_Cost_INR", "USD_EUR_RM_INR"])
        tables = {
            "products": pd.DataFrame({"Product_ID": ["P1", "P2"], "Capacity_per_month": [100, 200], "Labour_hours_per_unit": [1, 2]}),
            "bom": pd.DataFrame({"Product_ID": ["P1", "P1", "P2"], "RM_ID": ["A", "B", "B"], "Qty_per_unit": [2, 3, 1]}),
            "raw_materials": pd.DataFrame({"RM_ID": ["A", "B"], "Currency": ["USD", "INR"]}),
            "rm_price": pd.DataFrame({"Month": ["2025-12-01"] * 2, "RM_ID": ["A", "B"], "Price_per_unit": [10, 5]}),
            "currency_rates": pd.DataFrame({"Month": ["2025-12-01"], "Currency": ["USD"], "Rate_to_INR": [80]}),
            "shipping_duty": panel[planning.KEYS].assign(Shipping_per_unit=2, **{"Import_Duty_%": 5}),
        }
        result = planning.prepare_costs(panel, tables)
        self.assertEqual(result.iloc[0].RM_Cost_INR, 1615)
        self.assertEqual(result.iloc[0].USD_EUR_RM_INR, 1600)
        self.assertEqual(result.iloc[2].RM_Cost_INR, 5)
        self.assertEqual(result.iloc[2].USD_EUR_RM_INR, 0)
        tables["rm_price"] = tables["rm_price"].iloc[1:]
        with self.assertRaisesRegex(ValueError, "Missing RM"):
            planning.prepare_costs(panel, tables)

    def test_csv_panel_and_scenario_chart(self):
        """Complete input panel aggregates annual demand and 2025 baseline correctly."""
        from test_monthly_demand_forecast import make_panel

        history = make_panel()
        forecast = history.loc[history.Month.dt.year == 2025, ["Month", *planning.KEYS]].copy()
        forecast["Month"] += pd.DateOffset(years=1)
        forecast["Predicted_Units"] = 10.0
        with TemporaryDirectory() as directory:
            root = Path(directory)
            history.to_csv(root / "history.csv", index=False)
            forecast.to_csv(root / "forecast.csv", index=False)
            result = planning.load_csv_inputs(root / "forecast.csv", root / "history.csv")
            self.assertEqual(len(result), 50)
            self.assertTrue(result.Annual_Demand.eq(120).all())
            self.assertTrue(result.Profit_per_unit.eq(10).all())
            comparison = pd.DataFrame({"Scenario": planning.SCENARIOS, "Total_Profit_INR": [100, 80, 90]})
            planning.plot_scenarios(comparison, root / "chart.png")
            self.assertGreater((root / "chart.png").stat().st_size, 1000)

    def test_complete_workflow_and_mongo_document_encoding(self):
        """Run all scenarios for 50 combinations with mocked external MongoDB I/O."""
        from test_monthly_demand_forecast import make_panel

        history = make_panel()
        forecast = history.loc[history.Month.dt.year == 2025, ["Month", *planning.KEYS]].copy()
        forecast["Month"] += pd.DateOffset(years=1)
        forecast["Predicted_Units"] = 100.0
        tables = {
            "products": pd.DataFrame({"Product_ID": [f"P{i}" for i in range(5)], "Capacity_per_month": [100] * 5, "Labour_hours_per_unit": [1] * 5}),
            "bom": pd.DataFrame({"Product_ID": [f"P{i}" for i in range(5)], "RM_ID": ["A"] * 5, "Qty_per_unit": [1] * 5}),
            "raw_materials": pd.DataFrame({"RM_ID": ["A"], "Currency": ["USD"]}),
            "rm_price": pd.DataFrame({"Month": ["2025-12-01"], "RM_ID": ["A"], "Price_per_unit": [1]}),
            "currency_rates": pd.DataFrame({"Month": ["2025-12-01"], "Currency": ["USD"], "Rate_to_INR": [80]}),
            "shipping_duty": history[planning.KEYS].drop_duplicates().assign(Shipping_per_unit=2, **{"Import_Duty_%": 5}),
        }
        with TemporaryDirectory() as directory, patch.object(planning, "load_mongo_inputs", return_value=tables), patch("pymongo.MongoClient") as factory, patch("builtins.print"):
            root = Path(directory)
            history.to_csv(root / "history.csv", index=False)
            forecast.to_csv(root / "forecast.csv", index=False)
            status = planning.main(["--forecast", str(root / "forecast.csv"), "--dataset", str(root / "history.csv"), "--output-dir", str(root / "output")])
            self.assertEqual(status, 0)
            comparison = pd.read_csv(root / "output/scenario_comparison.csv")
            self.assertEqual(len(comparison), 3)
            self.assertGreater((root / "output/scenario_profit_comparison.png").stat().st_size, 1000)
            collection = factory.return_value.__enter__.return_value.__getitem__.return_value.__getitem__.return_value
            self.assertEqual(collection.update_one.call_count, 3)
            for call in collection.update_one.call_args_list:
                document = call.args[1]["$set"]
                self.assertEqual(len(document["Allocation"]), 50)
                self.assertEqual(len(document["Products"]), 5)
                self.assertEqual(len(document["Countries"]), 10)
                BSON.encode(document)


if __name__ == "__main__":
    unittest.main()
