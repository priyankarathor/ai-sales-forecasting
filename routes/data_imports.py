from dataclasses import dataclass
from io import BytesIO
from typing import Annotated, Callable

from fastapi import APIRouter, File, HTTPException, UploadFile
import pandas as pd
from pydantic import BaseModel, ValidationError
from starlette.datastructures import UploadFile as StarletteUploadFile

from models import product as product_model
from models import region_sales as region_sales_model
from models import sales as sales_model
from models import sales_forecasting as sales_forecasting_model
from models import user_salesforce as user_salesforce_model
from schemas.product import ProductSchema
from schemas.region_sales import RegionSalesSchema
from schemas.sales import SalesSchema
from schemas.user_salesforce import UserSalesforceSchema

router = APIRouter(
    prefix="/data-imports",
    tags=["Data Imports"],
)

@dataclass(frozen=True)
class ImportConfig:
    schema: type[BaseModel]
    insert: Callable[[list[dict]], int]
    form_field: str
    success_message: str
    total_key: str


IMPORTS: dict[str, ImportConfig] = {
    "sales_forecasting": ImportConfig(
        schema=SalesSchema,
        insert=lambda records: sales_forecasting_model.insert_sales_forecasting(records),
        form_field="sales_forecasting",
        success_message="Sales forecasting records inserted successfully",
        total_key="total_sales",
    ),
    "user_salesforce": ImportConfig(
        schema=UserSalesforceSchema,
        insert=lambda records: user_salesforce_model.insert_user_salesforce(records),
        form_field="user_salesforce",
        success_message="Salesforce users inserted successfully",
        total_key="total_users",
    ),
    "products": ImportConfig(
        schema=ProductSchema,
        insert=lambda records: product_model.insert_products(records),
        form_field="product",
        success_message="Products inserted successfully",
        total_key="total_products",
    ),
    "sales": ImportConfig(
        schema=SalesSchema,
        insert=lambda records: sales_model.insert_sales(records),
        form_field="sales",
        success_message="Sales inserted successfully",
        total_key="total_sales",
    ),
    "region-sales": ImportConfig(
        schema=RegionSalesSchema,
        insert=lambda records: region_sales_model.insert_region_sales(records),
        form_field="region_sales",
        success_message="Region sales inserted successfully",
        total_key="total_region_sales",
    ),
}


def _select_upload(
    config: ImportConfig,
    file: UploadFile | str | None,
    dataset_file: UploadFile | str | None,
) -> UploadFile:
    uploads = [
        value
        for value in (file, dataset_file)
        if isinstance(value, StarletteUploadFile)
    ]

    if len(uploads) > 1:
        raise HTTPException(
            status_code=400,
            detail=f"Upload one Excel file using either 'file' or '{config.form_field}'",
        )

    if not uploads:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Attach an Excel file using the 'file' or '{config.form_field}' form field. "
                "In Postman, choose form-data and set the field type to File, "
                "then select the .xls or .xlsx file. A filename sent as Text is not a file upload."
            ),
        )

    upload = uploads[0]
    if not upload.filename or not upload.filename.lower().endswith((".xls", ".xlsx")):
        raise HTTPException(
            status_code=400,
            detail="Upload an .xls or .xlsx Excel file",
        )

    return upload


def _read_validated_rows(upload: UploadFile, schema: type[BaseModel]) -> list[dict]:
    required_columns = list(schema.model_fields)

    content = upload.file.read()
    df = pd.read_excel(BytesIO(content))
    df.columns = df.columns.map(
        lambda column: column.strip() if isinstance(column, str) else column
    )

    missing = [column for column in required_columns if column not in df.columns]
    if missing:
        raise HTTPException(
            status_code=400,
            detail=f"Missing columns: {missing}",
        )

    if df.empty:
        raise HTTPException(
            status_code=400,
            detail="Excel file is empty",
        )

    df = df[required_columns]
    df = df.astype(object).where(pd.notna(df), None)

    records = []
    for index, row in df.iterrows():
        try:
            values = {
                column: None if pd.isna(value) else value
                for column, value in row.items()
            }
            record = schema.model_validate(values)
            records.append(record.model_dump())
        except ValidationError as exc:
            raise HTTPException(
                status_code=422,
                detail={
                    "excel_row": int(index) + 2,
                    "errors": exc.errors(
                        include_input=False,
                        include_context=False,
                    ),
                },
            ) from exc

    return records


@router.post("/{dataset}/upload-excel")
def upload_data_import(
    dataset: str,
    file: Annotated[UploadFile | str | None, File()] = None,
    product: Annotated[UploadFile | str | None, File()] = None,
    sales: Annotated[UploadFile | str | None, File()] = None,
    sales_forecasting: Annotated[UploadFile | str | None, File()] = None,
    region_sales: Annotated[UploadFile | str | None, File()] = None,
    user_salesforce: Annotated[UploadFile | str | None, File()] = None,
):
    config = IMPORTS.get(dataset)
    if config is None:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown data import '{dataset}'. Use one of: {list(IMPORTS)}",
        )

    dataset_uploads = {
        "product": product,
        "sales": sales,
        "sales_forecasting": sales_forecasting,
        "region_sales": region_sales,
        "user_salesforce": user_salesforce,
    }
    upload = _select_upload(config, file, dataset_uploads[config.form_field])

    try:
        records = _read_validated_rows(upload, config.schema)
        count = config.insert(records)
        return {
            "message": config.success_message,
            config.total_key: count,
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Excel upload failed: {str(exc)}",
        ) from exc
