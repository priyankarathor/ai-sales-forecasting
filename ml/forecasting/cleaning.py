"""Repeatable cleaning of preserved source rows, independent of training."""
import numpy as np
import pandas as pd


KEYS = ["product_id", "region"]
SALES_COLUMNS = ["sale_date", *KEYS, "units_sold"]


def clean_sales(records, cfg):
    frame = pd.DataFrame(records)
    missing = set(cfg.mapping.values()) - set(frame.columns)
    if frame.empty or missing:
        raise ValueError(f"Empty source or missing mapped columns: {sorted(missing)}")
    # Select only mapped fields, without modifying the preserved raw payload.
    rename = {source: name for name, source in cfg.mapping.items()}
    frame = frame[list(rename)].rename(columns=rename)
    dates = pd.to_datetime(frame["date"], errors="coerce", utc=True, format="mixed")
    frame["sale_date"] = dates.dt.tz_localize(None)
    frame["product_id"] = frame["product"].astype("string").str.strip()
    frame["region"] = (frame["region"].fillna("").astype("string").str.strip()
                       if "region" in frame else "")
    frame["units_sold"] = pd.to_numeric(frame["quantity"], errors="coerce")
    invalid = (frame.sale_date.isna() | frame.product_id.isna() | frame.product_id.eq("")
               | ~np.isfinite(frame.units_sold.to_numpy(dtype=float)))
    rejects = [{"source_row": int(i), "reason": "Invalid date, product or quantity"}
               for i in frame.index[invalid]]
    report = {"rows_in": len(frame), "rejects": rejects, "duplicates_removed": 0,
              "returns_excluded": 0, "gaps": [], "outliers": []}
    frame = frame.loc[~invalid].copy()
    if cfg.source_grain == "transaction":
        if frame.record_id.isna().any() or frame.record_id.astype(str).str.strip().eq("").any():
            raise ValueError("Missing transaction record_id")
        duplicate = frame.duplicated("record_id", keep=False)
        for _, group in frame.loc[duplicate].groupby("record_id"):
            if len(group.drop_duplicates()) != 1:
                raise ValueError("Conflicting rows for the same transaction record_id")
        before = len(frame)
        frame = frame.drop_duplicates("record_id")
        report["duplicates_removed"] = before - len(frame)
    # A period source is already aggregated: ambiguous duplicate keys must fail.
    negative = frame.units_sold < 0
    if negative.any() and cfg.returns_policy == "reject":
        raise ValueError("Negative sales: configure an agreed returns_policy")
    if cfg.returns_policy == "exclude":
        report["returns_excluded"] = int(negative.sum())
        frame = frame.loc[~negative].copy()
    frame["sale_date"] = (frame.sale_date.dt.to_period("M").dt.to_timestamp()
                          if cfg.frequency == "MS" else frame.sale_date.dt.normalize())
    if cfg.source_grain == "period" and frame.duplicated(["sale_date", *KEYS]).any():
        raise ValueError("Duplicate period/product/region; resolve upstream or configure transaction grain")
    clean = frame.groupby(["sale_date", *KEYS], as_index=False).units_sold.sum()
    if clean.empty or (clean.units_sold < 0).any():
        raise ValueError("No usable sales or negative net sales in a period")
    for key, group in clean.groupby(KEYS):
        expected = pd.date_range(group.sale_date.min(), group.sale_date.max(), freq=cfg.frequency)
        gaps = expected.difference(group.sale_date)
        report["gaps"].extend({"product_id": key[0], "region": key[1], "date": d.date().isoformat()} for d in gaps)
        median = group.units_sold.median()
        mad = (group.units_sold - median).abs().median()
        # Report only; no outlier observations are removed or changed.
        unusual = (group.units_sold - median).abs() > 6 * max(float(mad), 1.0)
        report["outliers"].extend({"product_id": key[0], "region": key[1],
                                    "date": row.sale_date.date().isoformat()}
                                   for row in group.loc[unusual].itertuples())
    report["rows_rejected"] = len(rejects)
    report["rows_out"] = len(clean)
    return clean.sort_values([*KEYS, "sale_date"]).reset_index(drop=True), report
