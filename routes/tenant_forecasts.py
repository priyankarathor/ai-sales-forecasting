"""Tenant registration and SQL snapshot APIs; model training stays in the worker."""
from dataclasses import asdict
from io import BytesIO
import logging
from typing import Annotated, Literal
from uuid import uuid4

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile
from fastapi.security import APIKeyHeader
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from ml.forecasting.config import TenantConfig

router = APIRouter(prefix="/sql", tags=["PostgreSQL forecasting"])
api_key = APIKeyHeader(name="X-Tenant-Key", auto_error=False)
logger = logging.getLogger(__name__)


class CompanyRegistration(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    tenant_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")
    company_name: str = Field(min_length=1, max_length=200)


class CSVOptions(BaseModel):
    model_config = ConfigDict(extra="forbid")
    mapping: dict[str, str] = Field(default_factory=lambda: {
        "date": "sale_date", "product": "product_id", "region": "region", "quantity": "quantity_sold"})
    frequency: Literal["D", "MS"] = "MS"
    horizon: int = Field(default=3, ge=1, le=30)
    backtest_folds: int = Field(default=2, ge=2, le=5)
    candidate_models: list[Literal["random_forest", "hist_gb", "xgboost"]] = Field(
        default_factory=lambda: ["random_forest", "hist_gb", "xgboost"], min_length=1)
    returns_policy: Literal["reject", "exclude", "subtract"] = "reject"
    source_grain: Literal["period", "transaction"] = "period"

    @field_validator("mapping")
    @classmethod
    def validate_mapping(cls, value):
        if set(value) - {"date", "product", "region", "quantity", "record_id"}:
            raise ValueError("Unknown mapping fields")
        if any(not column.strip() for column in value.values()) or len(set(value.values())) != len(value):
            raise ValueError("Mapping requires distinct, nonblank source columns")
        return value

    def worker_config(self, tenant_id):
        return TenantConfig(tenant_id=tenant_id, source={"type": "csv"}, **self.model_dump())


def get_store():
    store = None
    try:
        from database.forecast_store import ForecastStore
        from sqlalchemy.exc import SQLAlchemyError
        store = ForecastStore()
    except (ImportError, ValueError):
        raise HTTPException(503, "Configure FORECAST_DATABASE_URL and install the forecast-worker extra") from None
    try:
        yield store
    except SQLAlchemyError:
        logger.exception("SQL forecasting request failed")
        raise HTTPException(503, "Database unavailable; check connection and run the worker init command") from None
    finally:
        store.engine.dispose()


def tenant_store(tenant_id: str, key: Annotated[str | None, Depends(api_key)],
                 store=Depends(get_store)):
    if not key or not store.authenticate(tenant_id, key):
        raise HTTPException(401, "Invalid tenant ID or X-Tenant-Key")
    return store


@router.post("/tenants", status_code=201)
def register_company(request: CompanyRegistration, store=Depends(get_store)):
    from sqlalchemy.exc import IntegrityError
    try:
        return store.register_tenant(request.tenant_id, request.company_name)
    except IntegrityError:
        raise HTTPException(409, "Tenant ID already exists; choose a different ID") from None


@router.post("/tenants/{tenant_id}/upload-csv", status_code=201)
def upload_csv(tenant_id: str, file: Annotated[UploadFile, File()],
               options: Annotated[str, Form()] = "{}", store=Depends(tenant_store)):
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(400, "Upload a UTF-8 .csv file")
    try:
        cfg = CSVOptions.model_validate_json(options).worker_config(tenant_id)
    except (ValidationError, ValueError) as exc:
        raise HTTPException(422, str(exc)) from None
    content = file.file.read(10 * 1024 * 1024 + 1)
    if len(content) > 10 * 1024 * 1024:
        raise HTTPException(413, "CSV upload exceeds 10 MiB")
    try:
        frame = pd.read_csv(BytesIO(content), dtype=str, keep_default_na=False, encoding="utf-8-sig")
    except (ValueError, UnicodeError, pd.errors.ParserError) as exc:
        raise HTTPException(422, "Invalid or empty UTF-8 CSV") from None
    if frame.empty:
        raise HTTPException(422, "CSV contains no data rows")
    load_id = uuid4().hex
    # All columns are preserved, including fields unused by the demand model.
    store.save_raw(tenant_id, load_id, frame.to_dict("records"), config=asdict(cfg))
    from ml.forecasting.worker import clean_load
    try:
        quality = clean_load(cfg, store, load_id)
    except ValueError as exc:
        raise HTTPException(422, {"load_id": load_id, "message": str(exc),
                                  "raw_data_preserved": True}) from None
    return {"tenant_id": tenant_id, "load_id": load_id, "status": "clean", "quality": quality,
            "training_command": f"python -m ml.forecasting.worker run-upload --tenant-id {tenant_id} --load-id {load_id}"}


@router.get("/tenants/{tenant_id}/loads")
def list_loads(tenant_id: str, limit: int = Query(100, ge=1, le=1000),
               offset: int = Query(0, ge=0), store=Depends(tenant_store)):
    return store.list_loads(tenant_id, limit, offset)


@router.get("/tenants/{tenant_id}/runs")
def list_runs(tenant_id: str, limit: int = Query(100, ge=1, le=1000),
              offset: int = Query(0, ge=0), store=Depends(tenant_store)):
    return store.list_runs(tenant_id, limit, offset)


@router.get("/tenants/{tenant_id}/loads/{load_id}/raw")
def raw_details(tenant_id: str, load_id: str, limit: int = Query(100, ge=1, le=1000),
                offset: int = Query(0, ge=0), store=Depends(tenant_store)):
    rows = store.read_raw_page(tenant_id, load_id, limit, offset)
    if rows is None:
        raise HTTPException(404, "Load not found for this tenant")
    return rows


@router.get("/tenants/{tenant_id}/predictions")
def predictions(tenant_id: str, run_id: str | None = None,
                limit: int = Query(100, ge=1, le=1000), offset: int = Query(0, ge=0),
                store=Depends(tenant_store)):
    return store.read_forecasts(tenant_id, run_id, limit, offset)
