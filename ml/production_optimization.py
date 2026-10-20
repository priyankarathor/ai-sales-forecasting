"""Annual 2026 production LP: python -m ml.production_optimization.

SciPy HiGHS maximizes contribution profit by minimizing its negative.
This is an annual allocation, not a month-by-month production schedule.
"""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linprog
from database.monthly_connection import get_monthly_mongo_uri


# %% Editable assumptions: in values ko business ke hisaab se badlein.
PLAN_YEAR = 2026
ANNUAL_LABOUR_HOURS = 40_000.0  # Assumption: saare products ke liye shared yearly hours.
ANNUAL_RM_BUDGET_INR = 20_000_000.0  # Assumption: 2 crore sirf raw materials ke liye.
CAPACITY_MONTHS = 12  # Assumption: monthly capacity poore saal available hai.
COST_MONTH = "2025-12-01"  # Assumption: Dec-2025 RM prices/FX saare 2026 mein constant.
FX_SHOCK = 0.05  # Assumption: sirf USD/EUR Rate_to_INR 5% badhega.
DEMAND_REDUCTION = 0.10  # Assumption: demand har combination mein 10% ghategi.
SENSITIVITY_RELAXATION = 0.01  # Comparable constraint ranking: har RHS ko 1% relax karein.
TOLERANCE = 1e-7
KEYS = ["Product_ID", "Country_ID"]
SCENARIOS = ("Base", "USD_EUR_plus_5pct", "Demand_minus_10pct")
# Assumption: x continuous units hai, integer lot sizes/minimum orders nahi hain.
# Assumption: inventory, lead times, monthly bottlenecks aur fixed overhead excluded hain.
# Assumption: historical mean Profit_per_unit 2026 base contribution margin hai.
# Historical profit mein labour/shipping/duty already included hain: dobara subtract nahi.
# Current shipping_duty sirf reference/validation hai; changes base margin mein reprice nahi.
# Assumption: 2025 Units_Sold baseline production ka proxy hai (inventory change unknown).
# Negative/zero margins par production compulsory nahi: koi minimum-supply constraint nahi.


def numeric(frame, columns, nonnegative=()):
    """Required numbers validate karein; missing costs ko zero na samjhein."""
    result = frame.copy()
    for column in columns:
        if column not in result:
            raise ValueError(f"Missing numeric field: {column}")
        result[column] = result[column].map(lambda value: float(str(value)))
        if not np.isfinite(result[column]).all():
            raise ValueError(f"Non-finite values: {column}")
        if column in nonnegative and (result[column] < 0).any():
            raise ValueError(f"Negative values: {column}")
    return result


def normalize_ids(frame, columns):
    """CSV/MongoDB IDs strings mein normalize karein, collisions reject karein."""
    result = frame.copy()
    for column in columns:
        if column not in result or result[column].isna().any():
            raise ValueError(f"Missing ID: {column}")
        result[column] = result[column].astype("string").str.strip()
        if result[column].eq("").any():
            raise ValueError(f"Blank ID: {column}")
    return result


def parse_months(frame):
    """Month ko first-of-month UTC dates mein validate karein."""
    result = frame.copy()
    result["Month"] = pd.to_datetime(result["Month"], utc=True, errors="raise").dt.tz_localize(None)
    if result["Month"].isna().any() or not result["Month"].eq(result["Month"].dt.to_period("M").dt.to_timestamp()).all():
        raise ValueError("Month must be first-of-month midnight UTC")
    return result


def require_unique(frame, columns, label):
    """Ambiguous joins ko fail karein, taaki demand/cost multiply na ho."""
    if frame.duplicated(columns).any():
        raise ValueError(f"Duplicate keys in {label}: {columns}")


def load_csv_inputs(forecast_path, dataset_path):
    """600 demand rows aur complete 2023-2025 historical margins load karein."""
    forecast = parse_months(normalize_ids(pd.read_csv(forecast_path, dtype={k: "string" for k in KEYS}), KEYS))
    history = parse_months(normalize_ids(pd.read_csv(dataset_path, dtype={k: "string" for k in KEYS}), KEYS))
    forecast = numeric(forecast, ["Predicted_Units"], ["Predicted_Units"])
    history = numeric(history, ["Profit_per_unit", "Units_Sold"], ["Units_Sold"])
    for frame, dates, label in (
        (forecast, pd.date_range(f"{PLAN_YEAR}-01-01", periods=12, freq="MS"), "forecast"),
        (history, pd.date_range("2023-01-01", "2025-12-01", freq="MS"), "history"),
    ):
        require_unique(frame, ["Month", *KEYS], label)
        groups = list(frame.groupby(KEYS))
        if len(groups) != 50 or frame.Product_ID.nunique() != 5 or frame.Country_ID.nunique() != 10:
            raise ValueError(f"{label}: expected all 5 x 10 combinations")
        for key, group in groups:
            if not pd.DatetimeIndex(group.Month.sort_values()).equals(dates):
                raise ValueError(f"{label}: incomplete/unexpected dates for {key}")
    demand = forecast.groupby(KEYS, as_index=False).agg(Annual_Demand=("Predicted_Units", "sum"))
    # User ne average margin maanga hai: simple monthly average, units-weighted nahi.
    margins = history.groupby(KEYS, as_index=False).agg(Profit_per_unit=("Profit_per_unit", "mean"))
    baseline = history.loc[history.Month.dt.year == 2025].groupby(KEYS, as_index=False).agg(Baseline_2025_Units=("Units_Sold", "sum"))
    panel = demand.merge(margins, on=KEYS, how="outer", validate="one_to_one").merge(baseline, on=KEYS, how="outer", validate="one_to_one")
    if panel.isna().any().any():
        raise ValueError("Historical and forecast combinations do not match")
    print("Step 1: 50 annual demands, mean margins and 2025 sales baseline loaded")
    return panel


def load_mongo_inputs(uri, database):
    """Master/cost collections read karein aur MongoDB connection close karein."""
    from pymongo import MongoClient

    collections = ("products", "bom", "raw_materials", "rm_price", "currency_rates", "shipping_duty")
    with MongoClient(uri, serverSelectionTimeoutMS=5000) as client:
        client.admin.command("ping")
        result = {name: pd.DataFrame(client[database][name].find({}, {"_id": 0})) for name in collections}
    for name, frame in result.items():
        if frame.empty:
            raise ValueError(f"Empty MongoDB collection: {name}")
    print("Step 2: MongoDB products/BOM/prices/FX/shipping loaded")
    return result


def prepare_costs(panel, tables, cost_month=COST_MONTH):
    """BOM se snapshot RM cost aur USD/EUR exposure per product calculate karein."""
    products = normalize_ids(tables["products"], ["Product_ID"])
    products = numeric(products, ["Capacity_per_month", "Labour_hours_per_unit"], ["Capacity_per_month", "Labour_hours_per_unit"])
    require_unique(products, ["Product_ID"], "products")
    bom = numeric(normalize_ids(tables["bom"], ["Product_ID", "RM_ID"]), ["Qty_per_unit"], ["Qty_per_unit"])
    require_unique(bom, ["Product_ID", "RM_ID"], "bom")
    materials = normalize_ids(tables["raw_materials"], ["RM_ID", "Currency"])
    require_unique(materials, ["RM_ID"], "raw_materials")
    prices = parse_months(normalize_ids(tables["rm_price"], ["RM_ID"]))
    prices = numeric(prices, ["Price_per_unit"], ["Price_per_unit"])
    rates = parse_months(normalize_ids(tables["currency_rates"], ["Currency"]))
    rates = numeric(rates, ["Rate_to_INR"], ["Rate_to_INR"])
    if (rates.Rate_to_INR <= 0).any():
        raise ValueError("FX rates must be positive")
    month = pd.Timestamp(cost_month)
    if month != month.to_period("M").to_timestamp() or month >= pd.Timestamp(f"{PLAN_YEAR}-01-01"):
        raise ValueError("Cost snapshot must be a month start before the plan year")
    prices = prices.loc[prices.Month == month]
    rates = rates.loc[rates.Month == month]
    require_unique(prices, ["RM_ID"], "snapshot rm_price")
    require_unique(rates, ["Currency"], "snapshot currency_rates")
    lines = bom.loc[bom.Product_ID.isin(panel.Product_ID)].merge(
        materials[["RM_ID", "Currency"]], on="RM_ID", how="left", validate="many_to_one",
    ).merge(prices[["RM_ID", "Price_per_unit"]], on="RM_ID", how="left", validate="many_to_one")
    lines = lines.merge(rates[["Currency", "Rate_to_INR"]], on="Currency", how="left", validate="many_to_one")
    lines.loc[lines.Currency == "INR", "Rate_to_INR"] = 1.0
    if lines[["Currency", "Price_per_unit", "Rate_to_INR"]].isna().any().any():
        raise ValueError(f"Missing RM master/price/FX for snapshot {cost_month}")
    lines["RM_Cost_INR"] = lines.Qty_per_unit * lines.Price_per_unit * lines.Rate_to_INR
    lines["USD_EUR_RM_INR"] = lines.RM_Cost_INR.where(lines.Currency.isin(["USD", "EUR"]), 0)
    costs = lines.groupby("Product_ID", as_index=False)[["RM_Cost_INR", "USD_EUR_RM_INR"]].sum()
    shipping = normalize_ids(tables["shipping_duty"], KEYS)
    shipping = numeric(shipping, ["Shipping_per_unit", "Import_Duty_%"], ["Shipping_per_unit", "Import_Duty_%"])
    if (shipping["Import_Duty_%"] > 100).any():
        raise ValueError("Import duty cannot exceed 100%")
    require_unique(shipping, KEYS, "shipping_duty")
    result = panel.merge(products[["Product_ID", "Capacity_per_month", "Labour_hours_per_unit"]], on="Product_ID", how="left", validate="many_to_one")
    result = result.merge(costs, on="Product_ID", how="left", validate="many_to_one")
    result = result.merge(shipping[[*KEYS, "Shipping_per_unit", "Import_Duty_%"]], on=KEYS, how="left", validate="one_to_one")
    if result.isna().any().any():
        raise ValueError("Missing product/BOM/shipping master data for a forecast combination")
    result["Annual_Capacity"] = result.Capacity_per_month * CAPACITY_MONTHS
    print(f"Step 3: INR RM cost + USD/EUR exposure prepared using {cost_month}")
    return result.sort_values(KEYS).reset_index(drop=True)


def solve_scenario(panel, scenario, labour_hours=ANNUAL_LABOUR_HOURS, rm_budget=ANNUAL_RM_BUDGET_INR):
    """Annual LP solve karein; profit-max shadow prices aur comparable sensitivity dein."""
    if scenario not in SCENARIOS or not np.isfinite([labour_hours, rm_budget]).all() or min(labour_hours, rm_budget) < 0:
        raise ValueError("Invalid scenario or resource limits")
    data = panel.copy().reset_index(drop=True)
    data["Scenario_Profit_per_unit"] = data.Profit_per_unit
    data["Scenario_RM_Cost"] = data.RM_Cost_INR
    data["Scenario_Demand"] = data.Annual_Demand
    if scenario == "USD_EUR_plus_5pct":
        extra = data.USD_EUR_RM_INR * FX_SHOCK
        # FX shock margin ko ghatata hai aur same units ke budget use ko badhata hai.
        data["Scenario_RM_Cost"] += extra
        data["Scenario_Profit_per_unit"] -= extra
    elif scenario == "Demand_minus_10pct":
        data["Scenario_Demand"] *= 1 - DEMAND_REDUCTION
    rows, limits, labels, units = [], [], [], []
    for product, group in data.groupby("Product_ID", sort=True):
        rows.append(data.Product_ID.eq(product).astype(float).to_numpy())
        limits.append(float(group.Annual_Capacity.iloc[0]))
        labels.append(f"Capacity:{product}")
        units.append("INR / extra annual product unit")
    rows.extend([data.Labour_hours_per_unit.to_numpy(), data.Scenario_RM_Cost.to_numpy()])
    limits.extend([labour_hours, rm_budget])
    labels.extend(["Labour", "RM_Budget"])
    units.extend(["INR / extra labour hour", "INR / extra budget INR"])
    # Demand rows bhi explicit matrix mein hain: inke shadow prices bhi milenge.
    rows.extend(np.eye(len(data)))
    limits.extend(data.Scenario_Demand.to_numpy())
    labels.extend(f"Demand:{row.Product_ID}:{row.Country_ID}" for row in data.itertuples())
    units.extend(["INR / extra demand unit"] * len(data))
    matrix, rhs = np.asarray(rows, dtype=float), np.asarray(limits, dtype=float)
    objective = -data.Scenario_Profit_per_unit.to_numpy(dtype=float)
    solved = linprog(objective, A_ub=matrix, b_ub=rhs, bounds=(0, None), method="highs")
    if not solved.success:
        raise RuntimeError(f"{scenario}: solver failed: {solved.message}")
    if np.any(matrix @ solved.x > rhs + 1e-5) or np.any(solved.x < -TOLERANCE):
        raise RuntimeError("Solver result violates constraints")
    data["Recommended_Units"] = np.maximum(0, solved.x)
    data.loc[data.Recommended_Units.abs() < TOLERANCE, "Recommended_Units"] = 0.0
    data["Expected_Profit_INR"] = data.Recommended_Units * data.Scenario_Profit_per_unit
    data["Change_vs_2025"] = data.Recommended_Units - data.Baseline_2025_Units
    data["Action"] = np.select(
        [data.Recommended_Units <= TOLERANCE, data.Change_vs_2025 > TOLERANCE, data.Change_vs_2025 < -TOLERANCE],
        ["Band", "Zyada bhejo", "Kam bhejo"], default="Same",
    )
    # SciPy minimization marginal negative hota hai; maximize ke liye sign reverse.
    shadows = np.maximum(0, -np.asarray(solved.ineqlin.marginals))
    sensitivity = pd.DataFrame({"Constraint": labels, "Limit": rhs, "Used": matrix @ solved.x,
                                "Slack": solved.ineqlin.residual, "Shadow_Price": shadows, "Shadow_Unit": units})
    sensitivity["Binding"] = np.isclose(sensitivity.Slack, 0, atol=1e-6, rtol=0)
    sensitivity["Local_1pct_Estimated_Gain_INR"] = shadows * rhs * SENSITIVITY_RELAXATION
    # Different units wale raw shadow prices directly compare na karein.
    # 1% single-constraint re-solve se finite, comparable profit gain nikalein.
    gains = []
    for index, bound in enumerate(rhs):
        if bound <= 0 or shadows[index] <= TOLERANCE:
            gains.append(0.0)
            continue
        relaxed = rhs.copy()
        relaxed[index] += bound * SENSITIVITY_RELAXATION
        alternative = linprog(objective, A_ub=matrix, b_ub=relaxed, bounds=(0, None), method="highs")
        if not alternative.success:
            raise RuntimeError("Sensitivity re-solve failed")
        gains.append(max(0, float(solved.fun - alternative.fun)))
    sensitivity["Resolved_1pct_Gain_INR"] = gains
    sensitivity = sensitivity.sort_values(["Resolved_1pct_Gain_INR", "Constraint"], ascending=[False, True]).reset_index(drop=True)
    summary = {"Scenario": scenario, "Total_Profit_INR": float(data.Expected_Profit_INR.sum()),
               "Total_Units": float(data.Recommended_Units.sum()),
               "Labour_Hours": float(data.Recommended_Units @ data.Labour_hours_per_unit),
               "RM_Spend_INR": float(data.Recommended_Units @ data.Scenario_RM_Cost)}
    print(f"Step 4: {scenario} optimal; profit INR {summary['Total_Profit_INR']:,.2f}")
    return data, sensitivity, summary


def summarize_plan(allocation):
    """Product aur country tables banayein, 2025 sales se increase/decrease dikhayein."""
    product = allocation.groupby("Product_ID", as_index=False).agg(
        Recommended_Production=("Recommended_Units", "sum"), Baseline_2025_Units=("Baseline_2025_Units", "sum"),
        Expected_Profit_INR=("Expected_Profit_INR", "sum"), Annual_Capacity=("Annual_Capacity", "first"),
    )
    product["Change_vs_2025"] = product.Recommended_Production - product.Baseline_2025_Units
    country = allocation.groupby("Country_ID", as_index=False).agg(
        Recommended_Units=("Recommended_Units", "sum"), Baseline_2025_Units=("Baseline_2025_Units", "sum"),
        Expected_Profit_INR=("Expected_Profit_INR", "sum"),
    )
    country["Change_vs_2025"] = country.Recommended_Units - country.Baseline_2025_Units
    country["Action"] = np.select(
        [country.Recommended_Units <= TOLERANCE, country.Change_vs_2025 > TOLERANCE, country.Change_vs_2025 < -TOLERANCE],
        ["Band", "Zyada bhejo", "Kam bhejo"], default="Same",
    )
    increase = product.loc[(product.Change_vs_2025 > TOLERANCE) & (product.Expected_Profit_INR > TOLERANCE)].sort_values("Expected_Profit_INR", ascending=False)
    recommendation = ("In products ka production badhao: " + ", ".join(increase.Product_ID.astype(str))) if not increase.empty else "Kisi product ka production badhane ka profit-based support nahi hai."
    return product, country.sort_values("Expected_Profit_INR", ascending=False), recommendation


def plot_scenarios(comparison, path):
    """Teen scenarios ke optimized profits ka matplotlib bar chart save karein."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 5), dpi=140)
    bars = ax.bar(comparison.Scenario, comparison.Total_Profit_INR / 100_000, color=["#247d91", "#d48c32", "#7062a1"])
    ax.bar_label(bars, fmt="%.2f", padding=4)
    ax.set(ylabel="Optimized contribution profit (INR lakh)", title="2026 annual production scenarios")
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def save_plan(uri, database, results, assumptions):
    """Har scenario ka complete plan MongoDB mein upsert karein; repeat runs safe hain."""
    from pymongo import MongoClient

    with MongoClient(uri, serverSelectionTimeoutMS=5000) as client:
        client.admin.command("ping")
        collection = client[database]["production_plan"]
        collection.create_index([("Year", 1), ("Scenario", 1)], unique=True)
        for scenario, (allocation, sensitivity, summary) in results.items():
            product, country, recommendation = summarize_plan(allocation)
            document = {
                "Year": PLAN_YEAR, **summary, "Assumptions": assumptions,
                "Generated_At": datetime.now(timezone.utc), "Recommendation": recommendation,
                "Allocation": allocation.to_dict("records"), "Products": product.to_dict("records"),
                "Countries": country.to_dict("records"), "Sensitivity": sensitivity.to_dict("records"),
            }
            collection.update_one({"Year": PLAN_YEAR, "Scenario": scenario}, {"$set": document}, upsert=True)
    print("Step 7: all 3 scenarios saved to MongoDB production_plan")


def main(argv=None):
    """Inputs se optimized plans, tables, chart, sensitivity aur Mongo export banayein."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--forecast", type=Path, default=Path("forecast_2026.csv"))
    parser.add_argument("--dataset", type=Path, default=Path("ml_dataset.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/production_plan"))
    parser.add_argument("--mongo-uri", default=None)
    parser.add_argument("--database", default="sales_forecasting")
    parser.add_argument("--cost-month", default=COST_MONTH)
    parser.add_argument("--labour-hours", type=float, default=ANNUAL_LABOUR_HOURS)
    parser.add_argument("--rm-budget", type=float, default=ANNUAL_RM_BUDGET_INR)
    parser.add_argument("--skip-save-mongo", action="store_true")
    args = parser.parse_args(argv)
    try:
        args.mongo_uri = args.mongo_uri or get_monthly_mongo_uri()
        panel = load_csv_inputs(args.forecast, args.dataset)
        panel = prepare_costs(panel, load_mongo_inputs(args.mongo_uri, args.database), args.cost_month)
        assumptions = {
            "annual_labour_hours": args.labour_hours, "annual_rm_budget_inr": args.rm_budget,
            "cost_month": args.cost_month, "capacity_months": CAPACITY_MONTHS,
            "fx_shock": FX_SHOCK, "demand_reduction": DEMAND_REDUCTION,
            "sensitivity_relaxation": SENSITIVITY_RELAXATION,
            "profit": "Simple historical mean Profit_per_unit, excludes fixed costs; FX scenario subtracts incremental USD/EUR RM cost",
            "shipping": "Already included in historical margin; Mongo shipping/duty are references, no double subtraction",
            "baseline": "2025 actual units sold proxy production; no inventory data",
            "model": "Continuous annual allocation; no monthly scheduling, inventory, minimum orders, or fixed overhead",
            "rm_cost": "BOM quantity x snapshot material-currency price x snapshot FX; constant through 2026",
        }
        results = {scenario: solve_scenario(panel, scenario, args.labour_hours, args.rm_budget) for scenario in SCENARIOS}
        args.output_dir.mkdir(parents=True, exist_ok=True)
        for scenario, (allocation, sensitivity, _) in results.items():
            product, country, _ = summarize_plan(allocation)
            for name, table in (("allocation", allocation), ("products", product), ("countries", country), ("sensitivity", sensitivity)):
                table.to_csv(args.output_dir / f"{scenario}_{name}.csv", index=False)
        comparison = pd.DataFrame([result[2] for result in results.values()])
        comparison.to_csv(args.output_dir / "scenario_comparison.csv", index=False)
        (args.output_dir / "assumptions.json").write_text(json.dumps(assumptions, indent=2), encoding="utf-8")
        allocation, sensitivity, _ = results["Base"]
        product, country, recommendation = summarize_plan(allocation)
        print("\nProduct recommendations (base):\n" + product.to_string(index=False))
        print("\nCountry recommendations (base):\n" + country.to_string(index=False))
        print("\nProduct-country routing (base):\n" + allocation[[*KEYS, "Recommended_Units", "Change_vs_2025", "Action"]].to_string(index=False))
        print("\nScenario comparison:\n" + comparison.to_string(index=False))
        print("\nBase shadow prices (units shown separately):\n" + sensitivity.to_string(index=False))
        top = sensitivity.iloc[0]
        if top.Resolved_1pct_Gain_INR > TOLERANCE:
            print(f"Most limiting at {SENSITIVITY_RELAXATION:.0%} individual relaxation: {top.Constraint}; extra profit INR {top.Resolved_1pct_Gain_INR:,.2f}")
        else:
            print("No single constraint has positive profit gain under the tested relaxation; inspect margins and jointly binding constraints.")
        plot_scenarios(comparison, args.output_dir / "scenario_profit_comparison.png")
        print(f"Step 6: tables, shadow prices, assumptions and chart saved to {args.output_dir}")
        print("Shadow prices local LP values hain; degenerate optima par dual values nonunique ho sakte hain.")
        if not args.skip_save_mongo:
            save_plan(args.mongo_uri, args.database, results, assumptions)
        else:
            print("Step 7: Mongo save skipped; source collections were still read")
        print("\n" + recommendation)
        return 0
    except Exception as exc:
        print(f"Production optimization failed ({type(exc).__name__}): {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
