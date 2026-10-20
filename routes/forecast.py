from fastapi import APIRouter, HTTPException, Query
from database.connection import db
from pymongo.errors import PyMongoError
from ml.quality import sales_quality

router = APIRouter(
    prefix="/forecast",
    tags=["Forecast Data"],
)

COLLECTIONS = {
    "sales": "sales",
    "products": "product",
    "regions": "region_sales_collection",
    "users": "user_salesforce",
    "forecasts": "sales_forecasting",
}


@router.get("/sales-quality")
def get_sales_quality(product_id: str | None = None, region: str | None = None):
    try:
        return sales_quality(product_id, region)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except PyMongoError as exc:
        raise HTTPException(status_code=503, detail="Sales database is unavailable.") from exc


@router.get("/all-summary")
def get_all_summary():
    summary = {}

    for dataset, collection_name in COLLECTIONS.items():
        collection = db[collection_name]
        sample = collection.find_one({}, {"_id": 0})

        summary[dataset] = {
            "total": collection.count_documents({}),
            "columns": list(sample.keys()) if sample else [],
        }

    return summary


@router.get("/{dataset}")
def get_dataset(
    dataset: str,
    skip: int = Query(default=0, ge=0),
    limit: int = Query(default=100, ge=1, le=1000),
):
    collection_name = COLLECTIONS.get(dataset)

    if collection_name is None:
        raise HTTPException(
            status_code=404,
            detail="Unknown dataset",
        )

    collection = db[collection_name]

    cursor = (
        collection.find({}, {"_id": 0})
        .sort("_id", 1)
        .skip(skip)
        .limit(limit)
    )

    records = list(cursor)

    return {
        "dataset": dataset,
        "total": collection.count_documents({}),
        "returned": len(records),
        "skip": skip,
        "limit": limit,
        "data": records,
    }
