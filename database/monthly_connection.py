"""Shared connection configuration for monthly forecasting workflows."""

import os
from pathlib import Path

from dotenv import load_dotenv


def get_monthly_mongo_uri():
    """Monthly override, existing app URI, phir localhost fallback choose karein."""
    # Project ki .env read karein; process environment ki values overwrite na hon.
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
    return (os.getenv("MONTHLY_MONGO_URL") or os.getenv("MONGO_URL")
            or "mongodb://localhost:27017")
