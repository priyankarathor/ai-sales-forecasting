from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints
from typing import Annotated


Identifier = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
ModelType = Literal["linear_regression", "random_forest", "auto"]


class TrainingRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    product_id: Identifier
    region: Identifier
    model_type: ModelType = "linear_regression"
    frequency: Literal["auto", "D", "7D"] = "auto"
    test_fraction: float = Field(default=0.2, ge=0.1, le=0.4, allow_inf_nan=False)


class PredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    product_id: Identifier
    region: Identifier
    days: int = Field(default=7, ge=1, le=90, strict=True)
    model_type: ModelType = "linear_regression"
    include_details: bool = False
