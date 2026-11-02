import pandas as pd
from database.connection import sales_collection

SALES_COLUMNS = [
    "sale_id",
    "sale_date",
    "product_id",
    "region",
    "quantity_sold",
    "mrp_inr",
    "discount_pct", 
    "net_revenue_inr",
]

def load_sales(
    product_id: str | None = None,
    region: str | None = None,
) -> pd.DataFrame:
    query = {}

    if product_id is not None:
        query["product_id"] = product_id

    if region is not None:
        query["region"] = region

    projection = {
        "_id": 0,
        **{column: 1 for column in SALES_COLUMNS},
    }

    records = list(
        sales_collection.find(query, projection)
    )

    if not records:
        raise ValueError("No sales records found for these filters")

    return pd.DataFrame(records, columns=SALES_COLUMNS)


if __name__ == "__main__":
    sales = load_sales()

    print("Total records:", len(sales))
    print(sales.head())
    print("\nColumn types:")
    print(sales.dtypes)