"""Read-only checks of sales series before training."""
import pandas as pd

from ml.data_loader import load_sales
from ml.preprocessing import prepare_daily_sales, prepare_model_series


def sales_quality(product_id: str | None = None, region: str | None = None) -> dict:
    sales = load_sales(product_id=product_id, region=region)
    issues = int(sales[["product_id", "region"]].isna().any(axis=1).sum())
    results = []
    for (product, area), rows in sales.groupby(["product_id", "region"], dropna=False):
        result = {"product_id": None if pd.isna(product) else str(product),
                  "region": None if pd.isna(area) else str(area), "source_rows": len(rows)}
        duplicate_ids = int(rows["sale_id"].dropna().duplicated().sum()) if "sale_id" in rows else 0
        result["duplicate_sale_ids"] = duplicate_ids
        try:
            series, frequency = prepare_model_series(prepare_daily_sales(rows, product, area))
            missing = int(series["units_sold"].isna().sum())
            result.update(
                frequency=frequency, observations=len(series), missing_observations=missing,
                duplicate_sale_ids=duplicate_ids,
                first_sales_date=series["date"].min().date().isoformat(),
                last_sales_date=series["date"].max().date().isoformat(),
                ready_for_training=len(series) >= 42 and missing == 0 and duplicate_ids == 0,
            )
        except (ValueError, TypeError) as exc:
            result.update(ready_for_training=False, error=str(exc))
        results.append(result)
    return {"source": "sales", "total_records": len(sales), "invalid_identifier_rows": issues,
            "total_series": len(results),
            "ready_series": sum(row["ready_for_training"] for row in results), "series": results}
