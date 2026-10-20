from database.connection import sales_forecasting_collection


def insert_sales_forecasting(sales: list[dict]) -> int:
    if not sales:
        return 0

    result = sales_forecasting_collection.insert_many(sales)
    return len(result.inserted_ids)
