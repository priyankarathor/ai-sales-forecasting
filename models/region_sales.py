from database.connection import region_sales_collection

def insert_region_sales(sales: list[dict]) -> int:
    if not sales:
        return 0

    result = region_sales_collection.insert_many(sales)

    return len(result.inserted_ids)