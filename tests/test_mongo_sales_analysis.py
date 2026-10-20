"""Offline checks for CSV compatibility and connection cleanup."""

from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from bson import BSON
from bson.decimal128 import Decimal128
import pandas as pd
from pymongo.errors import ConnectionFailure

from ml import mongo_sales_analysis as reports


class MongoSalesAnalysisTests(unittest.TestCase):
    """Verify exported values and failure handling without a MongoDB server."""

    def test_csv_round_trip(self):
        """BSON decimal values and dates survive export with the required schema."""
        row = dict(zip(reports.CSV_COLUMNS, [
            datetime(2025, 10, 1, tzinfo=timezone.utc), "P1", "C1", 10,
            Decimal128("100.5"), 10, Decimal128("20.25"), 5,
            Decimal128("50.155"), 10, 1,
        ]))
        with TemporaryDirectory() as directory:
            path = Path(directory) / "dataset.csv"
            reports.export_ml_dataset([row], path)
            frame = pd.read_csv(path)
        self.assertEqual(list(frame.columns), reports.CSV_COLUMNS)
        self.assertEqual(frame.iloc[0]["Month"], "2025-10-01")
        self.assertEqual(frame.iloc[0]["RM_Cost"], 20.25)
        self.assertAlmostEqual(frame.iloc[0]["Profit_per_unit"], 50.155)
        self.assertEqual(frame.iloc[0]["Is_Festival_Month"], 1)

    def test_empty_export_has_headers(self):
        """Empty rows still produce a readable CSV with all required columns."""
        with TemporaryDirectory() as directory:
            path = Path(directory) / "empty.csv"
            reports.export_ml_dataset([], path)
            frame = pd.read_csv(path)
        self.assertTrue(frame.empty)
        self.assertEqual(list(frame.columns), reports.CSV_COLUMNS)

    def test_connection_failure_closes_client(self):
        """A failed ping returns a failure code and releases the client."""
        with patch.object(reports, "MongoClient") as factory, patch("builtins.print"):
            client = factory.return_value
            client.admin.command.side_effect = ConnectionFailure("unreachable")
            self.assertEqual(reports.main(), 1)
            client.close.assert_called_once_with()

    def test_pipeline_bson_encoding(self):
        """Every builder returns a pipeline the PyMongo BSON encoder can accept."""
        for builder in (
            reports.product_totals_pipeline, reports.monthly_sales_pipeline,
            reports.yearly_growth_pipeline, reports.top_countries_pipeline,
            reports.profit_pipeline,
        ):
            with self.subTest(pipeline=builder.__name__):
                BSON.encode({"pipeline": builder()})


if __name__ == "__main__":
    unittest.main()
