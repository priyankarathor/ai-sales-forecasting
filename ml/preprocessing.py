import pandas as pd

from ml.data_loader import load_sales


def prepare_daily_sales(
    sales: pd.DataFrame,
    product_id: str,
    region: str,
) -> pd.DataFrame:
    required = {
        "sale_date",
        "product_id",
        "region",
        "quantity_sold",
    }

    missing = required - set(sales.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    # Prepare one product-region series at a time.
    df = sales.loc[
        (sales["product_id"] == product_id)
        & (sales["region"] == region)
    ].copy()

    if df.empty:
        raise ValueError("No sales found for this product and region")

    if "sale_id" in df and df["sale_id"].dropna().duplicated().any():
        raise ValueError("Duplicate sale_id values found; resolve duplicate transactions before forecasting")

    # Current database values may be strings.
    df["sale_date"] = pd.to_datetime(
        df["sale_date"],
        errors="coerce",
    )

    df["quantity_sold"] = pd.to_numeric(
        df["quantity_sold"],
        errors="coerce",
    )

    invalid = (
        df["sale_date"].isna()
        | df["quantity_sold"].isna()
        | df["quantity_sold"].isin([float("inf"), float("-inf")])
    )

    if invalid.any():
        raise ValueError(
            f"{int(invalid.sum())} rows contain invalid dates or quantities"
        )

    # This example expects nonnegative sales quantities.
    if (df["quantity_sold"] < 0).any():
        raise ValueError(
            "Negative quantities found; decide how to handle returns first"
        )

    df["date"] = df["sale_date"].dt.normalize()

    # Sum transactions belonging to the same day.
    daily = (
        df.groupby("date")["quantity_sold"]
        .sum()
        .sort_index()
    )

    # Missing dates stay NaN until we decide what they mean.
    daily = daily.asfreq("D")

    return daily.rename("units_sold").reset_index()


def prepare_model_series(
    daily: pd.DataFrame, frequency: str | None = None,
) -> tuple[pd.DataFrame, str]:
    """Preserve observed units; recognize regularly weekly dated observations."""
    observed = daily.dropna(subset=["units_sold"]).sort_values("date")
    if frequency is None:
        gaps = observed["date"].diff().dropna().dt.days
        weekly = (
            len(observed) >= 8
            and (gaps % 7 == 0).all()
            and (gaps == 7).any()
        )
        frequency = "7D" if weekly else "D"
    if frequency not in {"D", "7D"}:
        raise ValueError("Unsupported model frequency; retrain the model")
    if frequency == "D":
        return daily, frequency
    if observed.empty:
        raise ValueError("No observed sales available")
    offset_days = (observed["date"] - observed["date"].iloc[0]).dt.days
    if (offset_days % 7 != 0).any():
        raise ValueError("Sales dates no longer match the weekly model; retrain first")
    weekly = observed.set_index("date").asfreq("7D").reset_index()
    return weekly, frequency


if __name__ == "__main__":
    sales = load_sales()

    combinations = (
        sales[["product_id", "region"]]
        .drop_duplicates()
        .sort_values(["product_id", "region"])
    )

    print(combinations.to_string(index=False))
