import logging

from fastapi import APIRouter, HTTPException, Query
from pymongo.errors import PyMongoError

from ml.predict import predict_next_day, predict_periods
from ml.train import train_model
from ml.monthly_predictions import get_monthly_predictions
from ml.monthly_predictions import get_monthly_predictions
from schemas.prediction import PredictionRequest, TrainingRequest
from schemas.production import ProductionPlanRequest
from services.production import build_production_plan

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/predictions",
    tags=["Sales Predictions"],
)


@router.post("/train")
def train_prediction_model(request: TrainingRequest):
    try:
        return train_model(**request.model_dump())
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except PyMongoError as exc:
        logger.exception("Training data could not be fetched")
        raise HTTPException(status_code=503, detail="Sales database is unavailable.") from exc
    except Exception as exc:
        logger.exception("Sales model training failed")
        raise HTTPException(status_code=500, detail="Training failed. Check server logs.") from exc


@router.post("/future")
def post_future_predictions(request: PredictionRequest):
    return _get_prediction(periods=1, **request.model_dump())


@router.get("/next-day")
def get_next_day_prediction():
    result = _get_prediction()
    if result.get("forecast_horizon_days", 1) != 1:
        raise HTTPException(
            status_code=422,
            detail="Saved model uses weekly observations. Use /predictions/next-period.",
        )
    return result


@router.get("/monthly")
def get_monthly_forecast(
    product_id: str | None = Query(default=None, min_length=1),
    country_id: str | None = Query(default=None, min_length=1),
    month: int | None = Query(default=None, ge=1, le=12),
):
    """Naye monthly model ke saved 2026 forecasts Postman/browser mein dikhayein."""
    try:
        result = get_monthly_predictions(product_id, country_id, month)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail="Monthly forecasts unavailable. Run python -m ml.monthly_demand_forecast --from-mongo-sales --skip-mongo first.") from exc
    except (OSError, ValueError, KeyError, TypeError) as exc:
        logger.exception("Monthly forecast artifacts could not be read")
        raise HTTPException(status_code=503, detail="Monthly forecast files are invalid or unreadable. Rerun monthly training.") from exc
    if result["count"] == 0:
        raise HTTPException(status_code=404, detail="No monthly forecast found for these product/country/month filters.")
    return result


@router.get("/monthly")
def get_monthly_forecast(
    product_id: str | None = Query(default=None, min_length=1),
    country_id: str | None = Query(default=None, min_length=1),
    month: int | None = Query(default=None, ge=1, le=12),
):
    """Naye monthly model ke saved 2026 forecasts Postman/browser mein dikhayein."""
    try:
        result = get_monthly_predictions(product_id, country_id, month)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail="Monthly forecasts unavailable. Run python -m ml.monthly_demand_forecast --from-mongo-sales --skip-mongo first.") from exc
    except (OSError, ValueError, KeyError, TypeError) as exc:
        logger.exception("Monthly forecast artifacts could not be read")
        raise HTTPException(status_code=503, detail="Monthly forecast files are invalid or unreadable. Rerun monthly training.") from exc
    if result["count"] == 0:
        raise HTTPException(status_code=404, detail="No monthly forecast found for these product/country/month filters.")
    return result


@router.get("/next-period")
def get_next_period_prediction():
    return _get_prediction()


@router.get("/dashboard-data")
def get_dashboard_data():
    return _get_prediction(include_details=True)


@router.get("/future")
def get_future_predictions(
    periods: int = Query(default=4, ge=1, le=90),
    calendar_month: bool = False,
):
    return _get_prediction(periods=periods, calendar_month=calendar_month)


@router.post("/production-plan")
def get_production_plan(request: ProductionPlanRequest):
    if request.calendar_month:
        raise HTTPException(status_code=422, detail="Choose consecutive periods for production planning, not next-month-only demand.")
    forecast = _get_prediction(periods=request.periods)
    try:
        return build_production_plan(forecast, request)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


def _get_prediction(
    *, include_details: bool = False, periods: int | None = None,
    calendar_month: bool = False,
    **parameters,
):
    try:
        if periods is not None:
            return predict_periods(periods=periods, calendar_month=calendar_month,
                                   include_details=include_details, **parameters)
        if include_details:
            return predict_next_day(include_details=True)
        return predict_next_day()

    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=503,
            detail="Trained model is unavailable. Run training for this product, region and model_type first.",
        ) from exc

    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail=str(exc),
        ) from exc

    except PyMongoError as exc:
        logger.exception("Prediction data could not be fetched")
        raise HTTPException(status_code=503, detail="Sales database is unavailable.") from exc

    except Exception as exc:
        logger.exception("Sales prediction failed")

        raise HTTPException(
            status_code=500,
            detail="Prediction failed. Check server logs.",
        ) from exc
