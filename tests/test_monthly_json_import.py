"""Monthly JSON endpoint tests; MongoDB writes are mocked."""

import asyncio
from datetime import datetime, timezone
from io import BytesIO
import json
import unittest
from unittest.mock import patch

from fastapi import UploadFile
from pymongo.errors import ConnectionFailure

from app.main import app
from routes import monthly_data
import test_product_upload


SALE = {"Month": {"$date": "2023-01-01T00:00:00Z"}, "Product_ID": "P1",
        "Country_ID": "C1", "Units_Sold": 196, "Selling_Price": 214, "Discount_%": 5}


def post_body(path, payload):
    """Make a raw JSON ASGI request without an external HTTP client."""
    body = json.dumps(payload).encode()
    messages = []

    async def request():
        """Collect the HTTP response from the real FastAPI router."""
        async def receive():
            """Supply the request body."""
            return {"type": "http.request", "body": body, "more_body": False}

        async def send(message):
            """Collect response messages."""
            messages.append(message)

        await app({"type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1",
                   "method": "POST", "scheme": "http", "path": path, "query_string": b"",
                   "headers": [(b"content-type", b"application/json")],
                   "server": ("testserver", 80), "client": ("127.0.0.1", 1234), "root_path": ""}, receive, send)

    asyncio.run(request())
    status = next(message["status"] for message in messages if message["type"] == "http.response.start")
    content = b"".join(message.get("body", b"") for message in messages if message["type"] == "http.response.body")
    return status, json.loads(content)


class MonthlyJSONTests(unittest.TestCase):
    """Verify schema validation, BSON dates, registration and repeatable upserts."""

    def test_raw_json_converts_extended_date(self):
        """A posted $date becomes a BSON-compatible datetime before persistence."""
        with patch.object(monthly_data, "persist_records", return_value={"inserted": 1}) as persist:
            status, body = post_body("/monthly-data/sales", [SALE])
        self.assertEqual(status, 200, body)
        self.assertEqual(persist.call_args.args[0], "sales")
        self.assertEqual(persist.call_args.args[1][0]["Month"], datetime(2023, 1, 1, tzinfo=timezone.utc))

    def test_file_upload_matches_raw_json(self):
        """Real multipart endpoint accepts UTF-8 BOM and MongoDB date wrappers."""
        file = UploadFile(filename="sales.json", file=BytesIO(b"\xef\xbb\xbf" + json.dumps([SALE]).encode()))
        with patch.object(monthly_data, "persist_records", return_value={"received": 1}) as persist:
            status, body = test_product_upload.ProductUploadTests().post_form({"file": file}, path="/monthly-data/sales/upload-json")
        self.assertEqual(status, 200, body)
        self.assertIsInstance(persist.call_args.args[1][0]["Month"], datetime)

    def test_invalid_row_prevents_all_writes(self):
        """Invalid second row must stop the complete request before persistence."""
        for invalid in ({**SALE, "Units_Sold": -1}, {**SALE, "Discount_%": 101},
                        {**SALE, "Units_Sold": "196"}, {**SALE, "Units_Sold": True},
                        {**SALE, "Month": "2023-01-02"}, {**SALE, "$set": {}},
                        {key: value for key, value in SALE.items() if key != "Country_ID"}):
            with self.subTest(invalid=invalid), patch.object(monthly_data, "persist_records") as persist:
                status, _ = post_body("/monthly-data/sales", [SALE, invalid])
                self.assertEqual(status, 422)
                persist.assert_not_called()

    def test_empty_duplicate_and_unknown_collection(self):
        """Unknown destinations and duplicate keys cannot write."""
        with patch.object(monthly_data, "persist_records") as persist:
            for payload in ([], [SALE, SALE]):
                self.assertEqual(post_body("/monthly-data/sales", payload)[0], 422)
            self.assertEqual(post_body("/monthly-data/arbitrary", [SALE])[0], 422)
            persist.assert_not_called()

    def test_all_collection_schemas_accept_required_fields(self):
        """All nine documented collection names have valid schemas."""
        for name in monthly_data.KEYS:
            row = {field: "Sample" for field in monthly_data.TEXT_FIELDS[name]}
            row.update({field: 1 for field in monthly_data.NUMBER_FIELDS[name]})
            if "Month" in monthly_data.KEYS[name]:
                row["Month"] = "2025-12-01"
            with self.subTest(collection=name), patch.object(monthly_data, "persist_records", return_value={"received": 1}):
                self.assertEqual(post_body(f"/monthly-data/{name}", [row])[0], 200)

    def test_upsert_targets_correct_database_and_closes_client(self):
        """Same natural key updates one document in the monthly database."""
        records = monthly_data.validate_records("sales", [{**SALE, "Month": "2023-01-01"}])
        with patch.object(monthly_data, "get_monthly_mongo_uri", return_value="mongodb://configured-host:27017"), patch.object(monthly_data, "MongoClient") as factory:
            client = factory.return_value.__enter__.return_value
            collection = client.__getitem__.return_value.__getitem__.return_value
            collection.bulk_write.return_value.upserted_count = 1
            collection.bulk_write.return_value.matched_count = 0
            collection.bulk_write.return_value.modified_count = 0
            result = monthly_data.persist_records("sales", records)
            factory.assert_called_once_with("mongodb://configured-host:27017", serverSelectionTimeoutMS=5000)
            client.__getitem__.assert_called_with("sales_forecasting")
            operation = collection.bulk_write.call_args.args[0][0]
            self.assertEqual(operation._filter, {field: records[0][field] for field in monthly_data.KEYS["sales"]})
            self.assertTrue(operation._upsert)
            factory.return_value.__exit__.assert_called_once()
        self.assertEqual(result["inserted"], 1)

    def test_db_failure_returns_503(self):
        """Connection failures produce a usable error without exposing the URI."""
        with patch.object(monthly_data, "persist_records", side_effect=ConnectionFailure("secret URI")):
            status, response = post_body("/monthly-data/sales", [SALE])
        self.assertEqual(status, 503)
        self.assertNotIn("secret", response["detail"])


if __name__ == "__main__":
    unittest.main()
