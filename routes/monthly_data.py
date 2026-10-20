"""JSON imports for the capitalized monthly sales_forecasting schema."""

from datetime import datetime, timezone
import json
import math
from typing import Annotated, Literal

from bson import json_util
from fastapi import APIRouter, Body, File, HTTPException, UploadFile
from pymongo import MongoClient, UpdateOne
from pymongo.errors import PyMongoError
from starlette.concurrency import run_in_threadpool
from database.monthly_connection import get_monthly_mongo_uri


router = APIRouter(prefix="/monthly-data", tags=["Monthly JSON Imports"])
CollectionName = Literal[
    "products", "countries", "sales", "raw_materials", "bom", "rm_price",
    "currency_rates", "shipping_duty", "fixed_costs",
]
DATABASE = "sales_forecasting"

# Har collection ke natural keys: same file dobara upload karne par update hoga.
KEYS = {
    "products": ["Product_ID"], "countries": ["Country_ID"],
    "sales": ["Month", "Product_ID", "Country_ID"],
    "raw_materials": ["RM_ID"], "bom": ["Product_ID", "RM_ID"],
    "rm_price": ["Month", "RM_ID"], "currency_rates": ["Month", "Currency"],
    "shipping_duty": ["Product_ID", "Country_ID"], "fixed_costs": ["Month"],
}
TEXT_FIELDS = {
    "products": ["Product_ID", "Name", "Category"],
    "countries": ["Country_ID", "Country", "Currency", "Region"],
    "sales": ["Product_ID", "Country_ID"],
    "raw_materials": ["RM_ID", "Name", "Source_Country", "Currency", "Unit"],
    "bom": ["Product_ID", "RM_ID"], "rm_price": ["RM_ID"],
    "currency_rates": ["Currency"], "shipping_duty": ["Product_ID", "Country_ID"],
    "fixed_costs": [],
}
NUMBER_FIELDS = {
    "products": ["Capacity_per_month", "Labour_hours_per_unit", "Labour_Rate_per_hour"],
    "countries": [], "sales": ["Units_Sold", "Selling_Price", "Discount_%"],
    "raw_materials": [], "bom": ["Qty_per_unit"], "rm_price": ["Price_per_unit"],
    "currency_rates": ["Rate_to_INR"], "shipping_duty": ["Shipping_per_unit", "Import_Duty_%"],
    "fixed_costs": ["Factory_Rent", "Salary", "Electricity", "Other"],
}


def validate_records(collection, records):
    """Poori file validate karein; invalid row par DB write shuru na karein."""
    if not isinstance(records, list) or not records:
        raise ValueError("Send a non-empty JSON array of documents")
    required = set(TEXT_FIELDS[collection] + NUMBER_FIELDS[collection])
    if "Month" in KEYS[collection]:
        required.add("Month")
    result, seen = [], set()
    for position, document in enumerate(records, 1):
        if not isinstance(document, dict):
            raise ValueError(f"Row {position}: expected a JSON object")
        missing = required - document.keys()
        extra = document.keys() - required - {"_id"}
        if missing or extra:
            raise ValueError(f"Row {position}: missing fields {sorted(missing)}, unknown fields {sorted(extra)}")
        # Exported _id ignore karein: upsert natural keys se hota hai.
        row = {field: document[field] for field in required}
        for field in TEXT_FIELDS[collection]:
            value = row[field]
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"Row {position}: {field} must be a non-empty string")
            row[field] = value.strip()
        for field in NUMBER_FIELDS[collection]:
            value = row[field]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError(f"Row {position}: {field} must be a finite nonnegative JSON number")
            if field in ("Discount_%", "Import_Duty_%") and value > 100:
                raise ValueError(f"Row {position}: {field} must be between 0 and 100")
            if field == "Rate_to_INR" and value == 0:
                raise ValueError(f"Row {position}: Rate_to_INR must be positive")
        if "Month" in required:
            month = row["Month"]
            if isinstance(month, str):
                month = datetime.fromisoformat(month.replace("Z", "+00:00"))
            if not isinstance(month, datetime):
                raise ValueError(f"Row {position}: Month must be an ISO date or MongoDB $date")
            month = month.replace(tzinfo=timezone.utc) if month.tzinfo is None else month.astimezone(timezone.utc)
            if month.day != 1 or any((month.hour, month.minute, month.second, month.microsecond)):
                raise ValueError(f"Row {position}: Month must be first-of-month midnight UTC")
            row["Month"] = month
        key = tuple(row[field] for field in KEYS[collection])
        if key in seen:
            raise ValueError(f"Row {position}: duplicate natural key in this upload")
        seen.add(key)
        result.append(row)
    return result


def persist_records(collection, records):
    """Monthly database mein natural-key upsert karein aur connection close karein."""
    # Existing daily/weekly API ka sales_forcasting database alag hai.
    uri = get_monthly_mongo_uri()
    with MongoClient(uri, serverSelectionTimeoutMS=5000) as client:
        client.admin.command("ping")
        target = client[DATABASE][collection]
        target.create_index([(field, 1) for field in KEYS[collection]], unique=True)
        operations = [UpdateOne({field: row[field] for field in KEYS[collection]}, {"$set": row}, upsert=True) for row in records]
        result = target.bulk_write(operations, ordered=True)
    return {"database": DATABASE, "collection": collection, "received": len(records),
            "inserted": result.upserted_count, "matched": result.matched_count,
            "modified": result.modified_count}


def import_json(collection, content):
    """Extended JSON $date ko BSON datetime mein decode karke import karein."""
    try:
        decoded = json_util.loads(content)
        records = validate_records(collection, decoded)
    except (ValueError, TypeError, OverflowError, KeyError) as exc:
        raise HTTPException(status_code=422, detail=f"Invalid monthly JSON: {exc}") from exc
    try:
        return persist_records(collection, records)
    except PyMongoError as exc:
        # DB error mein URI/credentials expose na karein. Bulk writes transaction nahi
        # hain; DB-level failure par partial write possible hai, same file retry safe hai.
        raise HTTPException(status_code=503, detail="MongoDB import failed. Check connectivity and existing duplicate keys; retrying the same file is safe.") from exc


@router.post("/{collection}/upload-json")
async def upload_json(collection: CollectionName, file: Annotated[UploadFile, File(description="JSON array file; MongoDB Extended JSON $date supported")]):
    """JSON file ko multipart form-data ke file field se upload karein."""
    if not file.filename or not file.filename.lower().endswith(".json"):
        raise HTTPException(status_code=400, detail="Upload a .json file using the 'file' field")
    try:
        content = (await file.read()).decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise HTTPException(status_code=422, detail="JSON file must use UTF-8 encoding") from exc
    finally:
        await file.close()
    return await run_in_threadpool(import_json, collection, content)


@router.post("/{collection}")
def post_json(collection: CollectionName, records: Annotated[list[dict], Body(description="Non-empty array of monthly documents; Month supports ISO string or {$date: ISO string}")]):
    """Postman Body > raw > JSON se documents ka array import karein."""
    return import_json(collection, json.dumps(records))
