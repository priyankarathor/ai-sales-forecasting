"""Monthly ML APIs for React training, inspection and scenario prediction."""

import logging
from typing import Literal

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, ConfigDict, Field

from services import monthly_ml


router = APIRouter(prefix="/predictions/monthly", tags=["Monthly ML"])
logger = logging.getLogger(__name__)


class LivePredictionRequest(BaseModel):
    """Optional price/discount missing hone par historical averages lagte hain."""
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    product_id: str = Field(min_length=1)
    country_id: str = Field(min_length=1)
    month: int = Field(default=1, ge=1, le=12, strict=True)
    selling_price: float | None = Field(default=None, ge=0)
    discount: float | None = Field(default=None, ge=0, le=100)


def invoke(function, *args):
    """ML failures ko consistent HTTP errors mein convert karein."""
    try:
        return function(*args)
    except BlockingIOError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except (FileNotFoundError, OSError) as exc:
        raise HTTPException(status_code=503, detail="Monthly artifacts unavailable. Run POST /predictions/monthly/train.") from exc
    except LookupError as exc:
        raise HTTPException(status_code=404, detail="Unknown product, country or model result") from exc
    except Exception as exc:
        logger.exception("Monthly ML operation failed")
        raise HTTPException(status_code=503, detail="Monthly ML operation failed. Check server logs.") from exc


@router.get("/details")
def details():
    """Data, features, hyperparameters, metrics aur feature importance return karein."""
    return invoke(monthly_ml.model_details)


@router.get("/history")
def history(product_id: str = Query(min_length=1), country_id: str = Query(min_length=1)):
    """Selected combination ki training sales history dikhayein."""
    return invoke(monthly_ml.sales_history, product_id, country_id)


@router.get("/evaluation")
def evaluation(model: Literal["XGBoost", "Prophet", "Random Forest"] | None = None,
               product_id: str | None = None, country_id: str | None = None):
    """React actual-vs-predicted graph ke liye 2025 holdout rows dein."""
    return invoke(monthly_ml.test_predictions, model, product_id, country_id)


@router.post("/predict")
def predict(request: LivePredictionRequest):
    """Live saved-model prediction aur actual input features return karein."""
    return invoke(monthly_ml.live_prediction, request)


@router.post("/train")
def train():
    """Uploaded monthly sales par train karein; request complete hone tak wait karega."""
    return invoke(monthly_ml.train_monthly)
