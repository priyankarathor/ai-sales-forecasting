from datetime import datetime
from io import BytesIO
import unittest
from unittest.mock import patch

from fastapi import UploadFile
import pandas as pd

import test_product_upload
from models.region_sales import insert_region_sales


class RegionSalesUploadTests(unittest.TestCase):
    def make_row(self):
        return {
            "region_sales_id": "RS001",
            "sales_month": datetime(2026, 10, 1),
            "region_id": "R001",
            "region": "North",
            "hub_city": "Delhi",
            "product_id": "P001",
            "units_sold": 10,
            "target_units": 20,
            "avg_discount_pct": 0,
            "net_revenue_inr": 1000.5,
        }

    def post_row(self, row, field="region_sales"):
        content = BytesIO()
        pd.DataFrame([row]).to_excel(content, index=False)
        content.seek(0)
        return test_product_upload.ProductUploadTests().post_form(
            {field: UploadFile(filename="region_sales.xlsx", file=content)},
            path="/region_sales/upload-excel",
        )

    def test_registered_endpoint_accepts_both_upload_fields(self):
        for field in ("file", "region_sales"):
            with self.subTest(field=field):
                with patch("routes.region_sales.insert_region_sales", return_value=1) as insert:
                    status, response = self.post_row(self.make_row(), field)
                self.assertEqual(status, 200, response)
                self.assertEqual(response["total_region_sales"], 1)
                record = insert.call_args.args[0][0]
                self.assertEqual(record["units_sold"], "10")
                self.assertEqual(record["sales_month"], "2026-10-01T00:00:00")
                self.assertEqual(record["net_revenue_inr"], "1000.5")

    def test_missing_column_does_not_insert(self):
        row = self.make_row()
        row.pop("product_id")
        with patch("routes.region_sales.insert_region_sales") as insert:
            status, response = self.post_row(row)
        self.assertEqual(status, 400)
        self.assertIn("product_id", response["detail"])
        insert.assert_not_called()

    def test_blank_required_id_reports_excel_row(self):
        row = self.make_row()
        row["region_sales_id"] = None
        with patch("routes.region_sales.insert_region_sales") as insert:
            status, response = self.post_row(row)
        self.assertEqual(status, 422)
        self.assertEqual(response["detail"]["excel_row"], 2)
        insert.assert_not_called()

    def test_model_uses_region_sales_collection(self):
        records = [self.make_row()]
        with patch("models.region_sales.region_sales_collection") as collection:
            collection.insert_many.return_value.inserted_ids = ["test-id"]
            self.assertEqual(insert_region_sales(records), 1)
        collection.insert_many.assert_called_once_with(records)


if __name__ == "__main__":
    unittest.main()
