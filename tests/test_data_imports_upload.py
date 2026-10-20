from datetime import datetime
from io import BytesIO
import unittest
from unittest.mock import patch

from fastapi import UploadFile
import pandas as pd

from app.main import app
import test_product_upload


class DataImportsUploadTests(unittest.TestCase):
    def post_rows(self, dataset, rows, field):
        content = BytesIO()
        pd.DataFrame(rows).to_excel(content, index=False)
        content.seek(0)
        return test_product_upload.ProductUploadTests().post_form(
            {field: UploadFile(filename=f"{dataset}.xlsx", file=content)},
            path=f"/data-imports/{dataset}/upload-excel",
        )

    def test_route_is_registered(self):
        self.assertIn("/data-imports/{dataset}/upload-excel", app.openapi()["paths"])

    def test_products_import_uses_product_schema_and_model(self):
        row = {
            "product_id": "P001",
            "product_category": "Phone",
            "brand": "Acme",
            "product_name": "Model A",
            "size_model": "128GB",
            "product_type": "Retail",
            "cost_price_inr": 100,
            "mrp_inr": 150,
            "launch_date": datetime(2026, 10, 9),
            "substitute_products": None,
            "related_products": None,
            "supplier": "Supplier",
            "opening_stock": 10,
            "reorder_level": 5,
            "unit": "pcs",
            "status": "active",
        }
        with patch("models.product.insert_products", return_value=1) as insert:
            status, response = self.post_rows("products", [row], "product")

        self.assertEqual(status, 200, response)
        self.assertEqual(response["total_products"], 1)
        product = insert.call_args.args[0][0]
        self.assertEqual(product["launch_date"], "2026-10-09T00:00:00")
        self.assertEqual(product["substitute_products"], "")

    def test_sales_import_accepts_file_field(self):
        row = {
            "sale_id": "S001",
            "sale_date": datetime(2026, 10, 13),
            "product_id": "P001",
            "region": "Delhi",
            "quantity_sold": 2,
            "mrp_inr": 100.5,
            "discount_pct": 0,
            "net_revenue_inr": 201,
        }
        with patch("models.sales.insert_sales", return_value=1) as insert:
            status, response = self.post_rows("sales", [row], "file")

        self.assertEqual(status, 200, response)
        self.assertEqual(response["total_sales"], 1)
        self.assertEqual(insert.call_args.args[0][0]["quantity_sold"], "2")

    def test_region_sales_import_reports_missing_columns(self):
        row = {
            "region_sales_id": "RS001",
            "sales_month": datetime(2026, 10, 1),
            "region_id": "R001",
            "region": "North",
            "hub_city": "Delhi",
            "units_sold": 10,
            "target_units": 20,
            "avg_discount_pct": 0,
            "net_revenue_inr": 1000.5,
        }
        with patch("models.region_sales.insert_region_sales") as insert:
            status, response = self.post_rows("region-sales", [row], "region_sales")

        self.assertEqual(status, 400)
        self.assertIn("product_id", response["detail"])
        insert.assert_not_called()

    def sales_forecasting_row(self):
        return {
            "sale_id": "S001",
            "sale_date": datetime(2026, 10, 14),
            "product_id": "P001",
            "region": "North",
            "quantity_sold": 2,
            "mrp_inr": 100,
            "discount_pct": 10,
            "net_revenue_inr": 180,
        }

    def test_sales_forecasting_import_uses_its_model(self):
        for field in ("file", "sales_forecasting"):
            with self.subTest(field=field):
                with patch("models.sales_forecasting.insert_sales_forecasting", return_value=1) as insert:
                    with patch("models.sales.insert_sales") as sales_insert:
                        status, response = self.post_rows(
                            "sales_forecasting", [self.sales_forecasting_row()], field
                        )
                self.assertEqual(status, 200, response)
                self.assertEqual(response["total_sales"], 1)
                insert.assert_called_once()
                sales_insert.assert_not_called()
                record = insert.call_args.args[0][0]
                self.assertEqual(set(record), set(self.sales_forecasting_row()))
                self.assertEqual(record["sale_date"], "2026-10-14T00:00:00")
                self.assertEqual(record["quantity_sold"], "2")

    def test_sales_forecasting_missing_column_does_not_insert(self):
        row = self.sales_forecasting_row()
        row.pop("net_revenue_inr")
        with patch("models.sales_forecasting.insert_sales_forecasting") as insert:
            status, response = self.post_rows("sales_forecasting", [row], "file")
        self.assertEqual(status, 400)
        self.assertIn("net_revenue_inr", response["detail"])
        insert.assert_not_called()

    def test_sales_forecasting_invalid_row_does_not_insert(self):
        row = self.sales_forecasting_row()
        row["sale_id"] = None
        with patch("models.sales_forecasting.insert_sales_forecasting") as insert:
            status, response = self.post_rows("sales_forecasting", [row], "file")
        self.assertEqual(status, 422)
        self.assertEqual(response["detail"]["excel_row"], 2)
        insert.assert_not_called()

    def test_unknown_dataset_is_rejected(self):
        status, response = test_product_upload.ProductUploadTests().post_form(
            {},
            path="/data-imports/unknown/upload-excel",
        )
        self.assertEqual(status, 404)
        self.assertIn("Unknown data import", response["detail"])

    def user_salesforce_row(self):
        return {
            "user_id": "U001",
            "email_id": "user@example.com",
            "full_name": "Sales User",
            "phone": 9876543210,
            "role": "Sales Representative",
            "department": "Sales",
            "region": "North",
            "status": "active",
            "created_at": datetime(2026, 10, 14),
            "last_login_at": None,
        }

    def test_user_salesforce_import_normalizes_excel_values(self):
        for field in ("file", "user_salesforce"):
            with self.subTest(field=field):
                with patch("models.user_salesforce.insert_user_salesforce", return_value=1) as insert:
                    status, response = self.post_rows(
                        "user_salesforce", [self.user_salesforce_row()], field
                    )
                self.assertEqual(status, 200, response)
                self.assertEqual(response["total_users"], 1)
                user = insert.call_args.args[0][0]
                self.assertEqual(set(user), set(self.user_salesforce_row()))
                self.assertEqual(user["phone"], "9876543210")
                self.assertEqual(user["created_at"], "2026-10-14T00:00:00")
                self.assertIsNone(user["last_login_at"])

    def test_user_salesforce_missing_column_does_not_insert(self):
        row = self.user_salesforce_row()
        row.pop("email_id")
        with patch("models.user_salesforce.insert_user_salesforce") as insert:
            status, response = self.post_rows("user_salesforce", [row], "file")
        self.assertEqual(status, 400)
        self.assertIn("email_id", response["detail"])
        insert.assert_not_called()

    def test_user_salesforce_invalid_row_does_not_insert(self):
        row = self.user_salesforce_row()
        row["user_id"] = None
        with patch("models.user_salesforce.insert_user_salesforce") as insert:
            status, response = self.post_rows("user_salesforce", [row], "file")
        self.assertEqual(status, 422)
        self.assertEqual(response["detail"]["excel_row"], 2)
        insert.assert_not_called()


if __name__ == "__main__":
    unittest.main()
