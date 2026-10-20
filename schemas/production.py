from datetime import date
from typing import Literal

from pydantic import BaseModel, Field


class ProductionPlanRequest(BaseModel):
    product_id: str = Field(min_length=1)
    region: str = Field(min_length=1)
    periods: int = Field(default=4, ge=1, le=90, strict=True)
    calendar_month: bool = False
    units_are_period_totals: Literal[True]
    stock_as_of: date
    current_stock: float = Field(ge=0, allow_inf_nan=False)
    safety_stock: float = Field(ge=0, allow_inf_nan=False)
    incoming_stock: float = Field(ge=0, allow_inf_nan=False)
    available_capacity: float | None = Field(default=None, ge=0, allow_inf_nan=False)
