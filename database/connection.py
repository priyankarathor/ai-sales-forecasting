import os
from dotenv import load_dotenv
from pymongo import MongoClient

load_dotenv()

MONGO_URL = os.getenv("MONGO_URL")

if not MONGO_URL:
    raise ValueError("MONGO_URL is missing in .env")

client = MongoClient(
    MONGO_URL,
    serverSelectionTimeoutMS=5000
)

db = client["sales_forcasting"]
products_collection = db["product"]
sales_collection = db["sales"]
sales_forecasting_collection = db["sales_forecasting"]
region_sales_collection = db["region_sales_collection"]
user_salesforce_collection = db["user_salesforce"]
