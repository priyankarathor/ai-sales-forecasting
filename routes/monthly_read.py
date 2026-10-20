"""Read-only monthly MongoDB APIs for React tables and dropdowns."""

from datetime import date, datetime, timezone
from typing import Literal

from bson import Decimal128, ObjectId
from fastapi import APIRouter, HTTPException, Query
from fastapi.encoders import jsonable_encoder
from pymongo import MongoClient
from pymongo.errors import PyMongoError

from database.monthly_connection import get_monthly_mongo_uri
from routes.monthly_data import DATABASE, KEYS


router = APIRouter(prefix="/monthly-data", tags=["Monthly Data GET"])
ReadCollection = Literal[
    "products", "countries", "sales", "raw_materials", "bom", "rm_price",
    "currency_rates", "shipping_duty", "fixed_costs", "forecast_2026", "production_plan",
]
READ_COLLECTIONS = [*KEYS, "forecast_2026", "production_plan"]


def serialize(value):
    """ObjectId string, date ISO aur Decimal128 exact decimal string mein dein."""
    return jsonable_encoder(value, custom_encoder={ObjectId: str, Decimal128: lambda number: str(number.to_decimal())})


def read_page(database, collection, skip, limit, filters):
    """Stable _id order mein ek page dein; count filtered records ka hai."""
    target = database[collection]
    total = target.count_documents(filters)
    records = list(target.find(filters).sort("_id", 1).skip(skip).limit(limit))
    has_more = bool(records) and skip + len(records) < total
    return {"collection": collection, "total": total, "returned": len(records),
            "skip": skip, "limit": limit, "has_more": has_more,
            "next_skip": skip + len(records) if has_more else None, "data": records}


def execute_read(operation):
    """Har request ke baad client close karein; connection secrets return na hon."""
    try:
        with MongoClient(get_monthly_mongo_uri(), serverSelectionTimeoutMS=5000, tz_aware=True) as client:
            return serialize(operation(client[DATABASE]))
    except PyMongoError as exc:
        raise HTTPException(status_code=503, detail="Monthly MongoDB data is unavailable. Check the configured connection.") from exc


@router.get("/summary")
def summary():
    """React ke collection menu ke liye counts aur sample field names dein."""
    def operation(database):
        """Sirf supported project collections inspect karein."""
        collections = []
        for name in READ_COLLECTIONS:
            sample = database[name].find_one({}, {"_id": 0})
            collections.append({"collection": name, "total": database[name].count_documents({}),
                                "fields": list(sample) if sample else []})
        return {"database": DATABASE, "collections": collections}
    return execute_read(operation)


@router.get("/all")
def all_data(skip: int = Query(default=0, ge=0), limit: int = Query(default=100, ge=1, le=1000)):
    """Ek response mein har supported collection ka page dein, unlimited dump nahi."""
    return execute_read(lambda database: {"database": DATABASE, "collections": {
        name: read_page(database, name, skip, limit, {}) for name in READ_COLLECTIONS
    }})


@router.get("/{collection}")
def collection_data(
    collection: ReadCollection,
    skip: int = Query(default=0, ge=0),
    limit: int = Query(default=100, ge=1, le=1000),
    product_id: str | None = Query(default=None, min_length=1),
    country_id: str | None = Query(default=None, min_length=1),
    rm_id: str | None = Query(default=None, min_length=1),
    currency: str | None = Query(default=None, min_length=1),
    month: date | None = None,
):
    """Collection rows dein; IDs/currency exact-match aur month YYYY-MM-01 format hai."""
    filters = {}
    for field, value in (("Product_ID", product_id), ("Country_ID", country_id), ("RM_ID", rm_id), ("Currency", currency)):
        if value is not None:
            filters[field] = value
    if month is not None:
        if month.day != 1:
            raise HTTPException(status_code=422, detail="month must be the first day, for example 2025-01-01")
        filters["Month"] = datetime(month.year, month.month, 1, tzinfo=timezone.utc)
    return execute_read(lambda database: {"database": DATABASE, **read_page(database, collection, skip, limit, filters)})
