from fastapi import APIRouter, UploadFile, File, HTTPException
from pydantic import ValidationError
import pandas as pd
from io import BytesIO
from typing import Annotated
from starlette.datastructures import UploadFile as StarletteUploadFile

from schemas.region_sales import RegionSalesSchema
from models.region_sales import insert_region_sales

router = APIRouter(
    prefix="/region_sales",
    tags=["Region Sales"]
)

REQUIRED_COLUMNS = list(RegionSalesSchema.model_fields)

@router.post("/upload-excel")
def upload_region_sales(
    file: Annotated[UploadFile | str | None, File()] = None,
    region_sales: Annotated[UploadFile | str | None, File()] = None,
):
    uploads = [
        value for value in (file, region_sales)
        if isinstance(value, StarletteUploadFile)
    ]
    if len(uploads) > 1:
        raise HTTPException(
            status_code=400,
            detail="Upload one Excel file using either 'file' or 'region_sales'",
        )

    if not uploads:
        raise HTTPException(
            status_code=422,
            detail=(
                "Attach an Excel file using the 'file' or 'region_sales' form field. "
                "In Postman, choose form-data and set the field type to File, "
                "then select the .xls or .xlsx file. A filename sent as Text is not a file upload."
            ),
        )
    file = uploads[0]

    if not file.filename or not file.filename.lower().endswith((".xls", ".xlsx")):
        raise HTTPException(
            status_code=400,
            detail="Upload an .xls or .xlsx Excel file"
        )

    try:
        content = file.file.read()
        df = pd.read_excel(BytesIO(content))
        df.columns = df.columns.map(
            lambda column: column.strip() if isinstance(column, str) else column
        )

        missing = [
            col for col in REQUIRED_COLUMNS
            if col not in df.columns
        ]

        if missing:
            raise HTTPException(
                status_code=400,
                detail=f"Missing columns: {missing}"
            )

        if df.empty:
            raise HTTPException(
                status_code=400,
                detail="Excel file is empty"
            )

        df = df[REQUIRED_COLUMNS]
        df = df.astype(object).where(pd.notna(df), None)

        records = []

        for index, row in df.iterrows():
            try:
                values = {
                    column: None if pd.isna(value) else value
                    for column, value in row.items()
                }
                record = RegionSalesSchema.model_validate(values)
                records.append(record.model_dump())

            except ValidationError as exc:
                raise HTTPException(
                    status_code=422,
                    detail={
                        "excel_row": int(index) + 2,
                        "errors": exc.errors(include_input=False, include_context=False)
                    }
                )

        count = insert_region_sales(records)

        return {
            "message": "Region sales inserted successfully",
            "total_region_sales": count
        }

    except HTTPException:
        raise

    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Excel upload failed: {str(exc)}"
        )