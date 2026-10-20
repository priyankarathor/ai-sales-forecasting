"""MongoDB GET API pagination, BSON encoding and filter tests."""

from datetime import datetime, timezone
import unittest
from unittest.mock import patch

from bson import Decimal128, ObjectId
from pymongo.errors import ConnectionFailure

from routes import monthly_read
import test_predictions


class MonthlyReadTests(unittest.TestCase):
    """Check real GET route contracts without using a remote database."""

    def request(self, path):
        """Reuse the ASGI request helper against the registered app."""
        return test_predictions.PredictionTests().request_prediction(path)

    def test_pagination_bson_and_filters(self):
        """React receives JSON-safe IDs/dates and accurate filtered pagination."""
        with patch.object(monthly_read, "MongoClient") as factory:
            target = factory.return_value.__enter__.return_value.__getitem__.return_value.__getitem__.return_value
            target.count_documents.return_value = 3
            target.find.return_value.sort.return_value.skip.return_value.limit.return_value = [
                {"_id": ObjectId("0123456789abcdef01234567"), "Month": datetime(2025, 1, 1, tzinfo=timezone.utc), "Price_per_unit": Decimal128("9.194")}
            ]
            status, result = self.request("/monthly-data/rm_price?rm_id=R1&month=2025-01-01&skip=1&limit=1")
            self.assertEqual(status, 200)
            self.assertEqual(result["database"], "sales_forecasting")
            self.assertEqual(result["next_skip"], 2)
            self.assertTrue(result["has_more"])
            self.assertEqual(result["data"][0]["_id"], "0123456789abcdef01234567")
            self.assertEqual(result["data"][0]["Price_per_unit"], "9.194")
            self.assertEqual(result["data"][0]["Month"], "2025-01-01T00:00:00+00:00")
            target.find.assert_called_once_with({"RM_ID": "R1", "Month": datetime(2025, 1, 1, tzinfo=timezone.utc)})
            factory.return_value.__exit__.assert_called_once()

    def test_bad_inputs_prevent_reads(self):
        """Invalid collections, page sizes and dates cannot query MongoDB."""
        with patch.object(monthly_read, "MongoClient") as factory:
            for path in ("/monthly-data/unknown", "/monthly-data/sales?limit=1001", "/monthly-data/sales?skip=-1", "/monthly-data/sales?month=2025-01-02"):
                self.assertEqual(self.request(path)[0], 422)
            factory.assert_not_called()

    def test_all_summary_and_empty_page(self):
        """All supported collections are present, including empty collections."""
        with patch.object(monthly_read, "MongoClient") as factory:
            target = factory.return_value.__enter__.return_value.__getitem__.return_value.__getitem__.return_value
            target.count_documents.return_value = 0
            target.find_one.return_value = None
            target.find.return_value.sort.return_value.skip.return_value.limit.return_value = []
            status, result = self.request("/monthly-data/all")
            self.assertEqual(status, 200)
            self.assertEqual(set(result["collections"]), set(monthly_read.READ_COLLECTIONS))
            self.assertFalse(result["collections"]["sales"]["has_more"])
            status, result = self.request("/monthly-data/summary")
            self.assertEqual(status, 200)
            self.assertEqual(len(result["collections"]), 11)
            self.assertTrue(all(row["total"] == 0 for row in result["collections"]))

    def test_database_error(self):
        """Database failures return 503 with no connection credentials."""
        with patch.object(monthly_read, "MongoClient", side_effect=ConnectionFailure("secret")):
            status, result = self.request("/monthly-data/products")
            self.assertEqual(status, 503)
            self.assertNotIn("secret", result["detail"])
