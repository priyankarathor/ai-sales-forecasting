from datetime import date, datetime

from pydantic import BaseModel, Field, field_validator

class ProductSchema(BaseModel):
    product_id : str = Field(min_length=1)
    product_category: str = Field(min_length=1)
    brand:str
    product_name : str
    size_model:str
    product_type:str
    cost_price_inr:str
    mrp_inr:str
    launch_date :str
    substitute_products:str = ""
    related_products:str = ""
    supplier:str
    opening_stock:str
    reorder_level:str
    unit:str
    status:str

    @field_validator("*", mode="before")
    @classmethod
    def normalize_excel_values(cls, value):
        if isinstance(value, (date, datetime)):
            return value.isoformat()
        if isinstance(value, (int, float)):
            return str(value)
        return value

    @field_validator("substitute_products", "related_products", mode="before")
    @classmethod
    def normalize_empty_related_products(cls, value):
        return "" if value is None else value
