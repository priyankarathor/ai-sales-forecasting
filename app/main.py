from fastapi import FastAPI
from fastapi.responses import FileResponse
from pathlib import Path
import os
from fastapi.middleware.cors import CORSMiddleware
from routes.data_imports import router as data_imports_router
from routes.tenant_forecasts import router as tenant_forecasts_router
from routes.product import router as product_router
from routes.sales import router as sales_router
from routes.region_sales import router as region_sales_router
from routes import forecast
from routes.predictions import router as predictions_router
from routes.monthly_data import router as monthly_data_router
from routes.monthly_ml import router as monthly_ml_router
from routes.monthly_read import router as monthly_read_router
from routes.monthly_read import router as monthly_read_router

app = FastAPI(
    title="AI Sales Forecasting API"
)

# React/Vite local origins allow karein; custom origins CORS_ORIGINS mein dein.
app.add_middleware(CORSMiddleware, allow_origins=[origin.strip() for origin in os.getenv(
    "CORS_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000,http://localhost:5173,http://127.0.0.1:5173"
).split(",") if origin.strip()], allow_methods=["GET", "POST", "OPTIONS"], allow_headers=["Content-Type", "Authorization", "X-Tenant-Key"])

app.include_router(product_router)
app.include_router(sales_router)
app.include_router(region_sales_router)
app.include_router(data_imports_router)
app.include_router(tenant_forecasts_router)
app.include_router(forecast.router)
app.include_router(predictions_router)
app.include_router(monthly_data_router)
app.include_router(monthly_ml_router)
app.include_router(monthly_read_router)
app.include_router(monthly_read_router)


@app.get("/dashboard", include_in_schema=False)
def dashboard():
    return FileResponse(
        Path(__file__).resolve().parents[1] / "static" / "dashboard.html"
    )


@app.get("/")
def home():
    return {
        "message": "Sales Forecasting API is running"
    }
