import pandas as pd

from ml.data_loader import load_sales
from ml.preprocessing import prepare_daily_sales


FEATURE_COLUMNS = [
    "lag_1",
    "lag_7",
    "rolling_mean_7",
    "day_of_week",
]

TARGET_COLUMN = "units_sold"


def create_features(daily: pd.DataFrame, frequency: str = "D") -> pd.DataFrame:
    df = daily.copy()
    df = df.sort_values("date").reset_index(drop=True)

    if df["date"].duplicated().any():
        raise ValueError("Expected one row per observation date")

    if df["units_sold"].isna().any():
        raise ValueError(
            "Missing sales observations found. Resolve them before creating features."
        )

    expected_dates = pd.date_range(
        start=df["date"].min(),
        periods=len(df),
        freq=frequency,
    )

    if not pd.DatetimeIndex(df["date"]).equals(expected_dates):
        raise ValueError("Expected consecutive dates at the selected frequency")

    df["lag_1"] = df["units_sold"].shift(1)
    df["lag_7"] = df["units_sold"].shift(7)

    df["rolling_mean_7"] = (
        df["units_sold"]
        .shift(1)
        .rolling(window=7, min_periods=7)
        .mean()
    )

    df["day_of_week"] = df["date"].dt.dayofweek

    # The first seven days do not have enough history.
    df = df.dropna(subset=FEATURE_COLUMNS).reset_index(drop=True)

    if df.empty:
        raise ValueError(
            "At least eight consecutive days are needed to create one feature row"
        )

    return df


if __name__ == "__main__":
    sales = load_sales()

    print(
        sales[["product_id", "region"]]
        .drop_duplicates()
        .to_string(index=False)
    )
