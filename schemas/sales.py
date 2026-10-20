from datetime import date, datetime

from pydantic import BaseModel, Field, field_validator

class SalesSchema(BaseModel):
    sale_id : str = Field(min_length=1)
    sale_date : str
    product_id : str
    region : str 
    quantity_sold : str 
    mrp_inr : str
    discount_pct : str
    net_revenue_inr : str

    @field_validator("*", mode="before")
    @classmethod
    def normalize_excel_values(cls, value):
        if isinstance(value, (date, datetime)):
            return value.isoformat()
        if isinstance(value, (int, float)):
            return str(value)
        return value

