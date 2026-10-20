from datetime import date, datetime

from pydantic import BaseModel, Field, field_validator

class RegionSalesSchema(BaseModel):
    region_sales_id : str = Field(min_length=1)
    sales_month: str = Field(min_length=1)
    region_id:str
    region : str
    hub_city:str
    product_id:str
    units_sold:str
    target_units:str
    avg_discount_pct :str
    net_revenue_inr:str = ""

    @field_validator("*", mode="before")
    @classmethod
    def normalize_excel_values(cls, value):
        if isinstance(value, (date, datetime)):
            return value.isoformat()
        if isinstance(value, (int, float)):
            return str(value)
        return value

