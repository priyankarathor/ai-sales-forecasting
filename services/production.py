import math

from schemas.production import ProductionPlanRequest


def build_production_plan(forecast: dict, request: ProductionPlanRequest) -> dict:
    if (request.product_id, request.region) != (forecast["product_id"], forecast["region"]):
        raise ValueError("Saved product or region changed. Refresh the dashboard before planning production.")
    if request.stock_as_of.isoformat() != forecast["last_sales_date"]:
        raise ValueError(
            "Stock date must match the latest sales date. Update sales history or supply inventory as of that date."
        )
    if request.calendar_month:
        raise ValueError(
            "Production planning uses consecutive periods starting after the stock date. "
            "Select a period count; a next-month-only forecast omits intervening demand."
        )
    demand = forecast["total_predicted_units"]
    required = math.ceil(max(
        0, demand + request.safety_stock - request.current_stock - request.incoming_stock,
    ))
    capacity = request.available_capacity
    return {
        "forecast": forecast,
        "stock_as_of": request.stock_as_of.isoformat(),
        "current_stock": request.current_stock,
        "safety_stock": request.safety_stock,
        "incoming_stock": request.incoming_stock,
        "required_production_units": required,
        "available_capacity": capacity,
        "capacity_shortfall_units": None if capacity is None else max(0, required - math.floor(capacity)),
        "units_are_period_totals": True,
        "planning_note": (
            "Volume estimate for this saved product and region only. Incoming stock must arrive within "
            "the forecast window. Lead times, stockouts within the window, shared regional inventory, "
            "and production scheduling are not modeled."
        ),
    }
