from database.connection import user_salesforce_collection


def insert_user_salesforce(users: list[dict]) -> int:
    if not users:
        return 0

    result = user_salesforce_collection.insert_many(users)
    return len(result.inserted_ids)
