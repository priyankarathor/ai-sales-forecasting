from datetime import datetime
import asyncio
from io import BytesIO
import json
import unittest
from unittest.mock import patch
from fastapi import HTTPException, UploadFile
import pandas as pd
from routes.product import REQUIRED_COLUMNS, upload_products
from app.main import app

class ProductUploadTests(unittest.TestCase):
    def post_form(self, fields, path="/products/upload-excel"):
        boundary = "product-upload-test"
        parts = []
        for name, value in fields.items():
            header = f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"'
            if isinstance(value, UploadFile):
                header += f'; filename="{value.filename}"\r\nContent-Type: application/octet-stream'
                payload = value.file.read()
            else:
                payload = value.encode()
            parts.append(header.encode() + b"\r\n\r\n" + payload + b"\r\n")
        body = b"".join(parts) + f"--{boundary}--\r\n".encode()
        messages = []

        async def request():
            async def receive():
                return {"type": "http.request", "body": body, "more_body": False}

            async def send(message):
                messages.append(message)

            await app({
                "type": "http", "asgi": {"version": "3.0"},
                "http_version": "1.1", "method": "POST", "scheme": "http",
                "path": path, "query_string": b"",
                "headers": [
                    (b"content-type", f"multipart/form-data; boundary={boundary}".encode()),
                    (b"content-length", str(len(body)).encode()),
                ],
                "server": ("testserver", 80), "client": ("127.0.0.1", 1234),
                "root_path": "",
            }, receive, send)

        asyncio.run(request())
        status = next(m["status"] for m in messages if m["type"] == "http.response.start")
        payload = b"".join(m.get("body", b"") for m in messages if m["type"] == "http.response.body")
        return status, json.loads(payload)

    def make_upload(self, row):
        content = BytesIO()
        pd.DataFrame([row]).to_excel(content, index=False)
        content.seek(0)
        return UploadFile(filename="products.xlsx", file=content)

    def test_excel_numbers_dates_and_blank_related_products(self):
        row = dict.fromkeys(REQUIRED_COLUMNS, "sample")
        row.update(
            product_id=123,
            cost_price_inr=100.5,
            mrp_inr=150,
            opening_stock=10,
            launch_date=datetime(2026, 10, 9),
            substitute_products=None,
            related_products=None,
        )
        row[" product_name "] = row.pop("product_name")
        with patch("routes.product.insert_products", return_value=1) as insert:
            response = upload_products(self.make_upload(row))

        self.assertEqual(response["total_products"], 1)
        product = insert.call_args.args[0][0]
        self.assertEqual(product["product_id"], "123")
        self.assertEqual(product["cost_price_inr"], "100.5")
        self.assertEqual(product["launch_date"], "2026-10-09T00:00:00")
        self.assertEqual(product["substitute_products"], "")
        self.assertEqual(product["related_products"], "")

    def test_supplied_product_columns_are_saved(self):
        columns = [
            "product_id", "product_category", "brand", "product_name",
            "size_model", "product_type", "cost_price_inr", "mrp_inr",
            "launch_date", "substitute_products", "related_products", "supplier",
            "opening_stock", "reorder_level", "unit", "status",
        ]
        row = dict.fromkeys(columns, "sample")
        with patch("routes.product.insert_products", return_value=1) as insert:
            status, response = self.post_form({"product": self.make_upload(row)})
        self.assertEqual(status, 200, response)
        self.assertEqual(set(insert.call_args.args[0][0]), set(columns))

    def test_missing_product_category_is_reported(self):
        row = dict.fromkeys(REQUIRED_COLUMNS, "sample")
        row.pop("product_category")
        with patch("routes.product.insert_products") as insert:
            status, response = self.post_form({"product": self.make_upload(row)})
        self.assertEqual(status, 400)
        self.assertIn("product_category", response["detail"])
        insert.assert_not_called()

    def test_invalid_required_value_does_not_insert(self):
        row = dict.fromkeys(REQUIRED_COLUMNS, "sample")
        row["product_id"] = None
        with patch("routes.product.insert_products") as insert:
            with self.assertRaises(HTTPException) as raised:
                upload_products(self.make_upload(row))

        self.assertEqual(raised.exception.status_code, 422)
        self.assertEqual(raised.exception.detail["excel_row"], 2)
        insert.assert_not_called()

    def test_product_form_field_is_accepted(self):
        row = dict.fromkeys(REQUIRED_COLUMNS, "sample")
        with patch("routes.product.insert_products", return_value=1) as insert:
            response = upload_products(product=self.make_upload(row))

        self.assertEqual(response["total_products"], 1)
        insert.assert_called_once()

    def test_missing_file_does_not_insert(self):
        with patch("routes.product.insert_products") as insert:
            with self.assertRaises(HTTPException) as raised:
                upload_products()

        self.assertEqual(raised.exception.status_code, 422)
        self.assertIn("'product'", raised.exception.detail)
        insert.assert_not_called()

    def test_two_files_are_rejected(self):
        row = dict.fromkeys(REQUIRED_COLUMNS, "sample")
        with patch("routes.product.insert_products") as insert:
            with self.assertRaises(HTTPException) as raised:
                upload_products(
                    file=self.make_upload(row), product=self.make_upload(row)
                )

        self.assertEqual(raised.exception.status_code, 400)
        insert.assert_not_called()

    def test_http_filename_text_returns_actionable_error(self):
        with patch("routes.product.insert_products") as insert:
            status, response = self.post_form({"file": "products.xlsx"})
        self.assertEqual(status, 422)
        self.assertIn("field type to File", response["detail"])
        insert.assert_not_called()

    def test_http_upload_works_with_extra_filename_text(self):
        for upload_field, text_field in (("product", "file"), ("file", "product")):
            with self.subTest(upload_field=upload_field):
                row = dict.fromkeys(REQUIRED_COLUMNS, "sample")
                with patch("routes.product.insert_products", return_value=1) as insert:
                    status, response = self.post_form({
                        upload_field: self.make_upload(row), text_field: "products.xlsx",
                    })
                self.assertEqual(status, 200, response)
                self.assertEqual(response["total_products"], 1)
                insert.assert_called_once()


if __name__ == "__main__":
    unittest.main()
