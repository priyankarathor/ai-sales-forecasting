import unittest
from unittest.mock import Mock, patch

import pandas as pd
from fastapi import HTTPException
from pymongo.errors import ServerSelectionTimeoutError

from ml.data_loader import load_sales, SALES_COLUMNS
from ml.quality import sales_quality
from routes.forecast import get_all_summary, get_dataset, get_sales_quality


class ForecastDataTests(unittest.TestCase):
    def sales(self):
        return pd.DataFrame({
            "sale_id": [f"S{i}" for i in range(50)],
            "sale_date": pd.date_range("2025-01-06", periods=50, freq="7D"),
            "product_id": "101", "region": "North", "quantity_sold": 5,
        })

    def test_loader_filters_database_and_projects_only_sales_columns(self):
        collection = Mock()
        collection.find.return_value = [self.sales().iloc[0].to_dict()]
        with patch("ml.data_loader.sales_collection", collection):
            result = load_sales("101", "North")
        self.assertEqual(list(result.columns), SALES_COLUMNS)
        self.assertEqual(collection.find.call_args.args[0], {"product_id": "101", "region": "North"})
        self.assertEqual(collection.find.call_args.args[1]["_id"], 0)

    def test_quality_identifies_complete_missing_and_duplicate_histories(self):
        sales = self.sales()
        with patch("ml.quality.load_sales", return_value=sales):
            result = sales_quality()
        self.assertEqual(result["ready_series"], 1)
        self.assertEqual(result["series"][0]["frequency"], "7D")
        with patch("ml.quality.load_sales", return_value=sales.drop(index=10)):
            result = sales_quality()
        self.assertEqual(result["ready_series"], 0)
        self.assertEqual(result["series"][0]["missing_observations"], 1)
        with patch("ml.quality.load_sales", return_value=pd.concat([sales, sales.iloc[[0]]])):
            result = sales_quality()
        self.assertEqual(result["ready_series"], 0)
        self.assertEqual(result["series"][0]["duplicate_sale_ids"], 1)

    def test_dataset_pagination_and_summary_fetch_all_configured_collections(self):
        database = Mock()
        database.__getitem__ = Mock(return_value=Mock())
        collection = database.__getitem__.return_value
        collection.count_documents.return_value = 2
        collection.find_one.return_value = {"sale_id": "S1"}
        collection.find.return_value.sort.return_value.skip.return_value.limit.return_value = [{"sale_id": "S2"}]
        with patch("routes.forecast.db", database):
            result = get_dataset("sales", skip=1, limit=1)
            self.assertEqual(result["data"], [{"sale_id": "S2"}])
            self.assertEqual(result["total"], 2)
            collection.find.return_value.sort.return_value.skip.assert_called_with(1)
            self.assertEqual(set(get_all_summary()), {"sales", "products", "regions", "users", "forecasts"})

    def test_quality_database_errors_return_503(self):
        with patch("routes.forecast.sales_quality", side_effect=ServerSelectionTimeoutError("private connection")):
            with self.assertRaises(HTTPException) as caught:
                get_sales_quality()
        self.assertEqual(caught.exception.status_code, 503)
        self.assertNotIn("private", caught.exception.detail)


if __name__ == "__main__":
    unittest.main()
