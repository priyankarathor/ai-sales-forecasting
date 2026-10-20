from datetime import datetime
from io import BytesIO
import unittest
from unittest.mock import patch

from fastapi import UploadFile
import pandas as pd

from app.main import app
from database.connection import sales_collection
from models.sales import insert_sales
import test_product_upload


class SalesUploadTests(unittest.TestCase):
    def make_row(self):
        return {
            "sale_id": "S001",
            "sale_date": datetime(2026, 10, 13),
            "product_id": "P001",
            "region": "Delhi",
            "quantity_sold": 2,
            "mrp_inr": 100.5,
            "discount_pct": 0,
            "net_revenue_inr": 201,
        }

    def post_rows(self, rows, field="sales"):
        content = BytesIO()
        pd.DataFrame(rows).to_excel(content, index=False)
        content.seek(0)
        return test_product_upload.ProductUploadTests().post_form(
            {field: UploadFile(filename="sales.xlsx", file=content)},
            path="/sales/upload-excel",
        )

    def test_route_is_registered(self):
        self.assertIn("/sales/upload-excel", app.openapi()["paths"])
        self.assertEqual(sales_collection.name, "sales")

    def test_both_file_fields_accept_sales_workbook(self):
        for field in ("sales", "file"):
            with self.subTest(field=field):
                with patch("routes.sales.insert_sales", return_value=1) as insert:
                    status, response = self.post_rows([self.make_row()], field=field)
                self.assertEqual(status, 200, response)
                self.assertEqual(response["total_sales"], 1)
                sale = insert.call_args.args[0][0]
                self.assertEqual(sale["sale_date"], "2026-10-13T00:00:00")
                self.assertEqual(sale["quantity_sold"], "2")
                self.assertEqual(sale["mrp_inr"], "100.5")
                self.assertEqual(sale["discount_pct"], "0")

    def test_missing_column_does_not_insert(self):
        row = self.make_row()
        row.pop("product_id")
        with patch("routes.sales.insert_sales") as insert:
            status, response = self.post_rows([row])
        self.assertEqual(status, 400)
        self.assertIn("product_id", response["detail"])
        insert.assert_not_called()

    def test_invalid_second_row_does_not_insert_any_rows(self):
        invalid = self.make_row()
        invalid["sale_id"] = None
        with patch("routes.sales.insert_sales") as insert:
            status, response = self.post_rows([self.make_row(), invalid])
        self.assertEqual(status, 422)
        self.assertEqual(response["detail"]["excel_row"], 3)
        insert.assert_not_called()

    def test_model_inserts_into_sales_collection(self):
        records = [self.make_row()]
        with patch("models.sales.sales_collection") as collection:
            collection.insert_many.return_value.inserted_ids = ["test-id"]
            self.assertEqual(insert_sales(records), 1)
        collection.insert_many.assert_called_once_with(records)


if __name__ == "__main__":
    unittest.main()
