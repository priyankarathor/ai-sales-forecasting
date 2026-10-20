from database.connection import sales_collection


def insert_sales(sales: list[dict]) -> int:
    if not sales:
        return 0

    result = sales_collection.insert_many(sales)

    return len(result.inserted_ids)