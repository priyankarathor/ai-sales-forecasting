from datetime import datetime, timezone
from pathlib import Path
from pprint import pprint

import pandas as pd
from pymongo import MongoClient
from pymongo.errors import PyMongoError
from database.monthly_connection import get_monthly_mongo_uri

CSV_COLUMNS = [
    "Month", "Product_ID", "Country_ID", "Units_Sold", "Selling_Price",
    "Discount", "RM_Cost", "Shipping", "Profit_per_unit", "Month_Number",
    "Is_Festival_Month",
]


def net_price_expression():
    """Discount ke baad ek unit ka selling price return karein."""
    return {"$multiply": [
        "$Selling_Price", {"$subtract": [1, {"$divide": ["$Discount_%", 100]}]},
    ]}


def product_totals_pipeline():
    """A1: Har product ke total units aur discounted revenue nikalein."""
    # Pehle sale ki net value, phir product ke hisaab se sum karein.
    return [
        {"$group": {
            "_id": "$Product_ID", "Total_Units": {"$sum": "$Units_Sold"},
            "Revenue": {"$sum": {"$multiply": ["$Units_Sold", net_price_expression()]}},
        }},
        {"$project": {"_id": 0, "Product_ID": "$_id", "Total_Units": 1, "Revenue": 1}},
        {"$sort": {"Product_ID": 1}},
    ]


def monthly_sales_pipeline():
    """A2: Country, product aur month ke hisaab se sales dikhayein."""
    # Month BSON date hi rakhein; sorting chronological hogi.
    return [
        {"$group": {
            "_id": {"Month": "$Month", "Product_ID": "$Product_ID", "Country_ID": "$Country_ID"},
            "Total_Units": {"$sum": "$Units_Sold"},
            "Revenue": {"$sum": {"$multiply": ["$Units_Sold", net_price_expression()]}},
        }},
        {"$project": {
            "_id": 0, "Month": "$_id.Month", "Product_ID": "$_id.Product_ID",
            "Country_ID": "$_id.Country_ID", "Total_Units": 1, "Revenue": 1,
        }},
        {"$sort": {"Country_ID": 1, "Product_ID": 1, "Month": 1}},
    ]


def growth_expression(current, previous):
    """YoY growth nikalein; missing ya zero base par null dein."""
    return {"$cond": [
        {"$and": [{"$ne": [current, None]}, {"$ne": [previous, None]}, {"$ne": [previous, 0]}]},
        {"$multiply": [{"$divide": [{"$subtract": [current, previous]}, previous]}, 100]},
        None,
    ]}


def yearly_growth_pipeline():
    """A3: 2023 baseline se 2024 aur 2025 ki units/revenue growth nikalein."""
    # Missing year ko zero sales na samjhein; null se farq clear rahega.
    pivot = {"_id": "$_id.Product_ID"}
    projection = {"_id": 0, "Product_ID": "$_id"}
    for year in (2023, 2024, 2025):
        for metric in ("Units", "Revenue"):
            key = f"{metric}_{year}"
            pivot[key] = {"$max": {"$cond": [{"$eq": ["$_id.Year", year]}, f"${metric}", None]}}
            projection[key] = 1
            if year > 2023:
                projection[f"{metric}_Growth_{year}_Pct"] = growth_expression(
                    f"${key}", f"${metric}_{year - 1}",
                )
    return [
        {"$match": {"Month": {"$gte": datetime(2023, 1, 1, tzinfo=timezone.utc),
                               "$lt": datetime(2026, 1, 1, tzinfo=timezone.utc)}}},
        {"$group": {
            "_id": {"Product_ID": "$Product_ID", "Year": {"$year": "$Month"}},
            "Units": {"$sum": "$Units_Sold"},
            "Revenue": {"$sum": {"$multiply": ["$Units_Sold", net_price_expression()]}},
        }},
        {"$group": pivot}, {"$project": projection}, {"$sort": {"Product_ID": 1}},
    ]


def top_countries_pipeline():
    """A4: Har product ke top 3 countries total units ke basis par nikalein."""
    # Tie hone par Country_ID se order fix rahega; maximum 3 countries milenge.
    return [
        {"$group": {"_id": {"Product_ID": "$Product_ID", "Country_ID": "$Country_ID"},
                     "Total_Units": {"$sum": "$Units_Sold"}}},
        {"$sort": {"_id.Product_ID": 1, "Total_Units": -1, "_id.Country_ID": 1}},
        {"$group": {"_id": "$_id.Product_ID", "Countries": {"$push": {
            "Country_ID": "$_id.Country_ID", "Total_Units": "$Total_Units",
        }}}},
        {"$project": {"_id": 0, "Product_ID": "$_id", "Top_3_Countries": {"$slice": ["$Countries", 3]}}},
        {"$sort": {"Product_ID": 1}},
    ]


def lookup_stage(collection, variables, conditions, output):
    """Multiple keys par lookup banayein, bina rows multiply kiye."""
    return {"$lookup": {
        "from": collection, "let": variables,
        "pipeline": [{"$match": {"$expr": {"$and": conditions}}}], "as": output,
    }}


def bom_cost_pipeline():
    """BOM ki har material line ko usi month ke INR cost mein badlein."""
    # $$sale_month outer sales lookup se aata hai; latest price use nahi hota.
    return [
        {"$match": {"$expr": {"$eq": ["$Product_ID", "$$product"]}}},
        lookup_stage("raw_materials", {"rm": "$RM_ID"}, [{"$eq": ["$RM_ID", "$$rm"]}], "material"),
        lookup_stage("rm_price", {"rm": "$RM_ID", "month": "$$sale_month"}, [
            {"$eq": ["$RM_ID", "$$rm"]}, {"$eq": ["$Month", "$$month"]},
        ], "prices"),
        {"$set": {"material_row": {"$arrayElemAt": ["$material", 0]},
                  "price_row": {"$arrayElemAt": ["$prices", 0]}}},
        lookup_stage("currency_rates", {"currency": "$material_row.Currency", "month": "$$sale_month"}, [
            {"$eq": ["$Currency", "$$currency"]}, {"$eq": ["$Month", "$$month"]},
        ], "rates"),
        # INR raw material ke liye exchange rate seedha 1 hai.
        {"$set": {"rate": {"$cond": [{"$eq": ["$material_row.Currency", "INR"]}, 1,
                                                   {"$arrayElemAt": ["$rates.Rate_to_INR", 0]}]}}},
        {"$project": {
            "_id": 0, "RM_ID": 1,
            "Cost": {"$multiply": ["$Qty_per_unit", "$price_row.Price_per_unit", "$rate"]},
            "Valid": {"$and": [
                {"$eq": [{"$size": "$material"}, 1]}, {"$eq": [{"$size": "$prices"}, 1]},
                {"$isNumber": "$Qty_per_unit"}, {"$gte": ["$Qty_per_unit", 0]},
                {"$isNumber": "$price_row.Price_per_unit"}, {"$gte": ["$price_row.Price_per_unit", 0]},
                {"$ne": [{"$ifNull": ["$material_row.Currency", None]}, None]},
                {"$or": [{"$eq": ["$material_row.Currency", "INR"]},
                         {"$eq": [{"$size": "$rates"}, 1]}]},
                {"$isNumber": "$rate"}, {"$gt": ["$rate", 0]},
            ]},
        }},
    ]


def profit_pipeline():
    """B: Ek hi calculation se monthly ML rows aur product-country profit dein."""
    # Arrays ko unwind nahi karte, taaki duplicate joins units/profit na badhayein.
    stages = [
        lookup_stage("products", {"product": "$Product_ID"}, [{"$eq": ["$Product_ID", "$$product"]}], "products"),
        lookup_stage("shipping_duty", {"product": "$Product_ID", "country": "$Country_ID"}, [
            {"$eq": ["$Product_ID", "$$product"]}, {"$eq": ["$Country_ID", "$$country"]},
        ], "shipping"),
        {"$lookup": {"from": "bom", "let": {"product": "$Product_ID", "sale_month": "$Month"},
                     "pipeline": bom_cost_pipeline(), "as": "bom_costs"}},
        {"$set": {"product": {"$arrayElemAt": ["$products", 0]},
                  "shipping_row": {"$arrayElemAt": ["$shipping", 0]},
                  "Net_Price": net_price_expression(), "RM_Cost": {"$sum": "$bom_costs.Cost"}}},
    ]
    # Missing/duplicate costs ko zero maan kar misleading CSV mat banayein.
    checks = [
        {"$eq": [{"$size": "$products"}, 1]}, {"$eq": [{"$size": "$shipping"}, 1]},
        {"$gt": [{"$size": "$bom_costs"}, 0]},
        {"$allElementsTrue": ["$bom_costs.Valid"]},
        {"$eq": [{"$size": "$bom_costs"}, {"$size": {"$setUnion": ["$bom_costs.RM_ID", []]}}]},
    ]
    for field in ("Units_Sold", "Selling_Price", "Discount_%", "product.Labour_hours_per_unit",
                  "product.Labour_Rate_per_hour", "shipping_row.Shipping_per_unit", "shipping_row.Import_Duty_%"):
        checks.extend([{"$isNumber": f"${field}"}, {"$gte": [f"${field}", 0]}])
    checks.extend([{"$lte": ["$Discount_%", 100]}, {"$lte": ["$shipping_row.Import_Duty_%", 100]}])
    stages.extend([
        {"$set": {
            "Cost_Data_Valid": {"$and": checks},
            "Profit_per_unit": {"$subtract": [
                {"$multiply": ["$Net_Price", {"$subtract": [1, {"$divide": ["$shipping_row.Import_Duty_%", 100]}]}]},
                {"$add": ["$RM_Cost", {"$multiply": ["$product.Labour_hours_per_unit", "$product.Labour_Rate_per_hour"]},
                          "$shipping_row.Shipping_per_unit"]},
            ]},
        }},
        {"$set": {"Revenue": {"$multiply": ["$Units_Sold", "$Net_Price"]},
                  "Profit": {"$multiply": ["$Units_Sold", "$Profit_per_unit"]}}},
        # Facet se summary aur month-level CSV dono SAME source rows se bante hain.
        {"$facet": {
            "invalid_rows": [
                {"$match": {"Cost_Data_Valid": False}},
                {"$project": {"Month": 1, "Product_ID": 1, "Country_ID": 1}},
            ],
            "product_country_profit": [
                {"$match": {"Cost_Data_Valid": True}},
                {"$group": {
                    "_id": {"Product_ID": "$Product_ID", "Country_ID": "$Country_ID"},
                    "Total_Units": {"$sum": "$Units_Sold"}, "Total_Revenue": {"$sum": "$Revenue"},
                    "Total_Profit": {"$sum": "$Profit"},
                }},
                {"$project": {
                    "_id": 0, "Product_ID": "$_id.Product_ID", "Country_ID": "$_id.Country_ID",
                    "Total_Units": 1, "Total_Revenue": 1, "Total_Profit": 1,
                    # Weighted margin: total profit / total discounted revenue.
                    "Profit_Margin_Pct": {"$cond": [{"$eq": ["$Total_Revenue", 0]}, None,
                        {"$multiply": [{"$divide": ["$Total_Profit", "$Total_Revenue"]}, 100]}]},
                }},
                {"$sort": {"Product_ID": 1, "Country_ID": 1}},
            ],
            "monthly_rows": [
                {"$match": {"Cost_Data_Valid": True}},
                {"$project": {
                    "_id": 0, "Month": 1, "Product_ID": 1, "Country_ID": 1, "Units_Sold": 1,
                    "Selling_Price": 1, "Discount": "$Discount_%", "RM_Cost": 1,
                    "Shipping": "$shipping_row.Shipping_per_unit", "Profit_per_unit": 1,
                    "Month_Number": {"$month": "$Month"},
                    "Is_Festival_Month": {"$cond": [{"$in": [{"$month": "$Month"}, [10, 11]]}, 1, 0]},
                }},
                {"$sort": {"Month": 1, "Product_ID": 1, "Country_ID": 1}},
            ],
        }},
    ])
    return stages


def export_ml_dataset(rows, output_path="ml_dataset.csv"):
    """C: Monthly profit rows ko exact requested columns ke CSV mein save karein."""
    # RM_Cost aur Shipping per unit INR hain; Discount percentage points hai.
    frame = pd.DataFrame(rows, columns=CSV_COLUMNS)
    frame["Month"] = pd.to_datetime(frame["Month"], utc=True).dt.strftime("%Y-%m-%d")
    # Decimal128 bhi pandas/CSV ke liye numeric ban jaye.
    for column in CSV_COLUMNS[3:]:
        frame[column] = frame[column].map(lambda value: float(str(value)))
    for column in ("Month_Number", "Is_Festival_Month"):
        frame[column] = frame[column].astype(int)
    frame.to_csv(Path(output_path), index=False)
    return frame


def print_sample(title, rows, limit=5):
    """Har pipeline ke actual output ki pehli kuch rows print karein."""
    print(f"\n{title} (total rows: {len(rows)}, sample: {limit})")
    pprint(rows[:limit], sort_dicts=False)


def main():
    """Local database se reports chalayein aur connection hamesha close karein."""
    client = None
    try:
        client = MongoClient(get_monthly_mongo_uri(), serverSelectionTimeoutMS=5000, tz_aware=True)
        client.admin.command("ping")
        db = client["sales_forecasting"]
        if db.sales.count_documents({}) == 0:
            raise ValueError("sales_forecasting.sales empty hai; pehle requested schema ka data import karein.")
        for title, pipeline in (
            ("A1 Product totals", product_totals_pipeline()),
            ("A2 Monthly country-product sales", monthly_sales_pipeline()),
            ("A3 Yearly growth (2023 baseline)", yearly_growth_pipeline()),
            ("A4 Top 3 countries per product", top_countries_pipeline()),
        ):
            print_sample(title, list(db.sales.aggregate(pipeline, allowDiskUse=True)))

        result = next(db.sales.aggregate(profit_pipeline(), allowDiskUse=True))
        if result["invalid_rows"]:
            print_sample("Invalid cost data: missing/duplicate joins ya invalid numeric values", result["invalid_rows"])
            raise ValueError("Cost data sahi karein; incomplete profit report aur CSV export rok diya gaya.")
        print_sample("B Product-country profit (INR), fixed costs excluded", result["product_country_profit"])
        frame = export_ml_dataset(result["monthly_rows"])
        print("\nC ML dataset sample:")
        print(frame.head().to_string(index=False))
        print(f"\nSaved {len(frame)} rows: {Path('ml_dataset.csv').resolve()}")
        return 0
    except (PyMongoError, ValueError, TypeError, OSError) as exc:
        print(f"Sales analysis failed: {exc}")
        return 1
    finally:
        if client is not None:
            client.close()


if __name__ == "__main__":
    raise SystemExit(main())
