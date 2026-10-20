from database.connection import products_collection


def insert_products(products: list[dict]) -> int:
    if not products:
        return 0

    result = products_collection.insert_many(products)

    return len(result.inserted_ids)