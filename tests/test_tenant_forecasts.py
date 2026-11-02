"""SQL API tests use temporary SQLite; no local/customer databases are touched."""
import asyncio
from io import BytesIO
import json
from pathlib import Path
import tempfile
import unittest

from fastapi import FastAPI
import pandas as pd

from database.forecast_store import ForecastStore
from ml.forecasting.config import TenantConfig
from ml.forecasting.worker import demo, run
from routes.tenant_forecasts import get_store, router


class TenantAPITests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.store = ForecastStore(f"sqlite:///{(self.root / 'test.sqlite').as_posix()}")
        self.store.create_schema()
        self.app = FastAPI()
        self.app.include_router(router)
        self.app.dependency_overrides[get_store] = lambda: self.store

    def tearDown(self):
        self.store.engine.dispose()
        self.temp.cleanup()

    def request(self, method, path, body=b"", content_type="application/json", key=None):
        messages = []
        headers = [(b"content-type", content_type.encode())]
        if key:
            headers.append((b"x-tenant-key", key.encode()))

        async def perform():
            async def receive():
                return {"type": "http.request", "body": body, "more_body": False}

            async def send(message):
                messages.append(message)

            await self.app({"type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1",
                            "method": method, "scheme": "http", "path": path, "query_string": b"",
                            "headers": headers, "server": ("testserver", 80), "client": ("127.0.0.1", 1),
                            "root_path": ""}, receive, send)
        asyncio.run(perform())
        status = next(m["status"] for m in messages if m["type"] == "http.response.start")
        return status, json.loads(b"".join(m.get("body", b"") for m in messages if m["type"] == "http.response.body"))

    def register(self, tenant="acme"):
        status, body = self.request("POST", "/sql/tenants", json.dumps(
            {"tenant_id": tenant, "company_name": "Acme Ltd"}).encode())
        self.assertEqual(status, 201, body)
        return body["api_key"]

    def upload(self, key, csv, options=None):
        options = options or {"mapping": {"date": "Month", "product": "Product_ID",
            "region": "Country_ID", "quantity": "Units_Sold"}, "candidate_models": ["random_forest"]}
        body = (b'--test\r\nContent-Disposition: form-data; name="options"\r\n\r\n' +
                json.dumps(options).encode() +
                b'\r\n--test\r\nContent-Disposition: form-data; name="file"; filename="sales.csv"\r\n'
                b'Content-Type: text/csv\r\n\r\n' + csv + b'\r\n--test--\r\n')
        return self.request("POST", "/sql/tenants/acme/upload-csv", body,
                            "multipart/form-data; boundary=test", key)

    def test_registration_authentication_and_duplicate_conflict(self):
        key = self.register()
        self.assertTrue(self.store.authenticate("acme", key))
        self.assertFalse(self.store.authenticate("other", key))
        status, _ = self.request("POST", "/sql/tenants", b'{"tenant_id":"acme","company_name":"Other"}')
        self.assertEqual(status, 409)
        for supplied in (None, "wrong"):
            self.assertEqual(self.request("GET", "/sql/tenants/acme/loads", key=supplied)[0], 401)
        other_key = self.register("other")
        self.assertEqual(self.request("GET", "/sql/tenants/acme/predictions", key=other_key)[0], 401)
        self.assertEqual(self.request("GET", "/sql/tenants/acme/loads", key=key), (200, []))

    def test_csv_worker_and_prediction_readback(self):
        key = self.register()
        path = self.root / "input.csv"
        demo(path)
        frame = pd.read_csv(path, dtype=str)
        frame["extra_details"] = "00123"
        status, uploaded = self.upload(key, frame.to_csv(index=False).encode())
        self.assertEqual(status, 201, uploaded)
        load_id = uploaded["load_id"]
        raw = self.store.read_raw("acme", load_id)
        self.assertEqual(raw[0]["extra_details"], "00123")
        saved = self.store.read_upload_config("acme", load_id)
        self.assertIsNone(self.store.read_upload_config("other", load_id))
        result = run(TenantConfig(**saved), self.store, self.root / "models", load_id=load_id)
        self.assertEqual(result["status"], "promoted")
        status, predictions = self.request("GET", "/sql/tenants/acme/predictions", key=key)
        self.assertEqual(status, 200)
        self.assertEqual(len(predictions), 12)
        self.assertEqual({row["run_id"] for row in predictions}, {result["run_id"]})
        other_key = self.register("other")
        self.assertEqual(self.request("GET", "/sql/tenants/other/predictions", key=other_key), (200, []))
        status, _ = self.request("GET", f"/sql/tenants/other/loads/{load_id}/raw", key=other_key)
        self.assertEqual(status, 404)

    def test_failed_quality_keeps_raw_and_previous_clean_sales(self):
        key = self.register()
        csv = b'Month,Product_ID,Country_ID,Units_Sold,details\n2024-01-01,001,IN,10,original\n'
        status, good = self.upload(key, csv)
        self.assertEqual(status, 201, good)
        status, failed = self.upload(key, csv.replace(b",10,", b",bad,"))
        self.assertEqual(status, 422, failed)
        self.assertTrue(failed["detail"]["raw_data_preserved"])
        self.assertEqual(len(self.store.read_sales("acme", good["load_id"])), 1)
        self.assertEqual(self.store.read_raw("acme", failed["detail"]["load_id"])[0]["Units_Sold"], "bad")
        self.assertEqual(self.store.list_loads("acme")[0]["status"], "failed")

    def test_invalid_configuration_and_csv_do_not_create_loads(self):
        key = self.register()
        for csv, options in ((b"", None), (b"a\n1\n", {"frequency": "MS", "horizon": 30}),
                             (b"a\n1\n", {"mapping": {"date": "same", "quantity": "same"}})):
            self.assertEqual(self.upload(key, csv, options)[0], 422)
        self.assertEqual(self.store.list_loads("acme"), [])


if __name__ == "__main__":
    unittest.main()
