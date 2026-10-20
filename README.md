# Sales Forecasting API

## Vercel deployment: monthly inference

The monthly XGBoost artifact requires its runtime library even for `/details`
and `/history`, because those endpoints deserialize `best_model.joblib`.
Runtime dependencies now include the trained model's pinned XGBoost,
scikit-learn, NumPy, pandas and joblib versions. Linux uses `xgboost-cpu` to
avoid shipping CUDA/NCCL. `requirements.txt` is generated from `uv.lock`:

```powershell
uv export --no-hashes --no-dev --no-emit-project --format requirements-txt --output-file requirements.txt
```

Vercel's entrypoint is `app.main:app`, with Python 3.13 selected through the
project's Python requirement. `vercel.json` includes trained artifacts and
excludes local `.env`, virtual environments, tests, examples and chart PNGs.
Deploy the complete active model directory: `active_run.json` refers to
`artifacts/monthly_demand/runs/<run_id>/`, whose model, metadata, forecast and
evaluation CSVs must all be present. Do not deploy only the pointer file.

Set `MONGO_URL` (or `MONTHLY_MONGO_URL`) and `CORS_ORIGINS` in Vercel project
environment settings. Local `.env` is not deployed. Redeploy the changed
`pyproject.toml`, `uv.lock`, `requirements.txt`, `vercel.json`, API code and
artifacts; if necessary clear the deployment build cache. Verify:

```text
GET /predictions/monthly/details
GET /predictions/monthly/history?product_id=P1&country_id=C1
POST /predictions/monthly/predict
```

The inference endpoints run on Vercel. `/predictions/monthly/train` explicitly
returns 501 there because this implementation writes persistent local runs.
Train locally or on a persistent training worker, then deploy the resulting
artifact directory. Missing model packages now return a structured 503 with
`monthly_dependency_missing` and the dependency name; incompatible imports
return `monthly_dependency_incompatible`. Other failures still require the
Vercel function log traceback to diagnose.

References: [Vercel Python runtime](https://vercel.com/docs/functions/runtimes/python)
and [XGBoost CPU installation](https://xgboost.readthedocs.io/en/stable/install.html).

## React: MongoDB GET APIs

All routes below read the `sales_forecasting` database using the configured
monthly/app MongoDB URI. They do not modify documents.

| GET endpoint | Result |
|---|---|
| `/monthly-data/summary` | All supported collection names, document counts and sample field names |
| `/monthly-data/all?skip=0&limit=100` | A separate page from each of the 11 collections |
| `/monthly-data/products` | Product records |
| `/monthly-data/countries` | Country records |
| `/monthly-data/sales` | Actual monthly sales |
| `/monthly-data/raw_materials` | Material master records |
| `/monthly-data/bom` | Bill of materials |
| `/monthly-data/rm_price` | Monthly material prices |
| `/monthly-data/currency_rates` | Monthly exchange rates |
| `/monthly-data/shipping_duty` | Shipping and duty records |
| `/monthly-data/fixed_costs` | Monthly overhead |
| `/monthly-data/forecast_2026` | Forecast documents saved in MongoDB |
| `/monthly-data/production_plan` | Production-plan documents saved in MongoDB |

Base URL is `http://127.0.0.1:8000`. Individual collection responses contain
`database`, `collection`, `total`, `returned`, `skip`, `limit`, `has_more`,
`next_skip` and `data`. Default limit is 100; maximum is 1,000. Use the returned
`next_skip` for the next page. `/all` is also paginated **per collection**;
it is not an unlimited dump. Pagination uses stable `_id` ordering, but does
not provide a database snapshot across concurrent edits.

Optional exact-match filters: `product_id`, `country_id`, `rm_id`, `currency`,
and `month` (first-of-month date, for example `2025-01-01`):

```text
/monthly-data/sales?product_id=P1&country_id=C1&limit=100
/monthly-data/rm_price?rm_id=R1&month=2025-01-01
```

Apply filters to collections containing those fields; unsupported combinations
simply return no matching rows. `_id` values are strings, dates are ISO strings,
and Decimal128 values are exact decimal strings. Empty collections return
`data: []`. Local CSV forecasts are available from `/predictions/monthly`;
they appear in `/monthly-data/forecast_2026` only after being saved in MongoDB.

Use [monthlyDataApi.js](examples/react/monthlyDataApi.js) in React:

```jsx
import { createMonthlyDataApi } from "./monthlyDataApi";
const api = createMonthlyDataApi();
// Inside an async handler/effect; catch errors and update your React state.
const page = await api.page("sales", { product_id: "P1", country_id: "C1" });
console.log(page.data);
const allSales = await api.allRecords("sales"); // follows every page
```

## React: monthly ML API

Base URL: `http://127.0.0.1:8000`. Swagger: `/docs`, tag **Monthly ML**.
These APIs use the new monthly sales model and are separate from legacy
daily/weekly `/predictions/train` and `/predictions/future` routes.

| Method | Endpoint | Use in React |
|---|---|---|
| GET | `/predictions/monthly/details` | Model, training coverage, product/country IDs, features, hyperparameters, metrics, importance |
| GET | `/predictions/monthly/history?product_id=P1&country_id=C1` | 36 actual training observations for a chart |
| GET | `/predictions/monthly/evaluation?model=XGBoost&product_id=P1&country_id=C1` | 2025 actual vs predicted rows; model/filter parameters optional |
| GET | `/predictions/monthly?product_id=P1&country_id=C1&month=1` | Saved 2026 forecasts; filters optional |
| POST | `/predictions/monthly/predict` | Fresh saved-model inference with actual input features and optional price/discount scenarios |
| POST | `/predictions/monthly/train` | Retrain/compare all three models from monthly MongoDB sales; no request body |

Live prediction JSON:

```json
{"product_id":"P1","country_id":"C1","month":1,"selling_price":227,"discount":3.75}
```

`month` is 1-12 in **2026**, default 1. Optional price/discount default to the
series' last 12 actual months' averages. Overrides apply to all months from
January through the requested month; intermediate sales lags use predictions.
Response `prediction.Predicted_Units` contains the requested month's demand,
`prediction.inputs` contains actual model inputs, and `forecast_path` contains
the intermediate predictions. Tree inputs include one-hot ID source values,
lags and shifted rolling means; Prophet reports its actual regressors.
Scenario predictions describe model behavior, not causal price effects.

Copy [monthlyMlApi.js](examples/react/monthlyMlApi.js) into your React source:

```jsx
import { createMonthlyMlApi } from "./monthlyMlApi";
const ml = createMonthlyMlApi("http://127.0.0.1:8000");

// In an async button handler; put errors/loading state in your React UI.
try {
  const result = await ml.predict({ product_id: "P1", country_id: "C1", month: 1 });
  console.log(result.prediction.Predicted_Units, result.prediction.inputs);
  const info = await ml.details();
  console.log(info.metrics, info.feature_importance);
} catch (error) {
  console.error(error.message);
}
```

The client also provides `history`, `evaluation`, `forecasts`, and `train`.
Training is synchronous and can take 1-2 minutes; show a loading state and use
a suitable proxy/request timeout. Concurrent training in the same API process
returns 409. Successful runs are saved under `artifacts/monthly_demand/runs/`
and activated using an atomic `active_run.json` pointer. Reads keep serving
the previous complete run during training. Use a single Uvicorn worker for
this local training workflow. These endpoints save local artifacts and do
not write model predictions to MongoDB.

CORS permits `localhost` and `127.0.0.1` on React ports **3000** and **5173**.
Set comma-separated `CORS_ORIGINS` in `.env` for other frontend origins and
restart the backend. Unknown IDs return 404; invalid inputs return 422;
missing artifacts/training failures return 503.

## Import the nine monthly JSON files

Start the server and open <http://127.0.0.1:8000/docs>:

```powershell
.\.venv\Scripts\python.exe -m uvicorn app.main:app --reload
```

Under **Monthly JSON Imports**, use
`POST /monthly-data/{collection}/upload-json`. Select a collection from the
dropdown, click **Try it out**, choose its `.json` file in `file`, then execute.
For Postman, use **Body > form-data**, key `file`, type **File**.

| JSON file | POST URL |
|---|---|
| products.json | `http://127.0.0.1:8000/monthly-data/products/upload-json` |
| countries.json | `http://127.0.0.1:8000/monthly-data/countries/upload-json` |
| raw_materials.json | `http://127.0.0.1:8000/monthly-data/raw_materials/upload-json` |
| bom.json | `http://127.0.0.1:8000/monthly-data/bom/upload-json` |
| rm_price.json | `http://127.0.0.1:8000/monthly-data/rm_price/upload-json` |
| currency_rates.json | `http://127.0.0.1:8000/monthly-data/currency_rates/upload-json` |
| shipping_duty.json | `http://127.0.0.1:8000/monthly-data/shipping_duty/upload-json` |
| fixed_costs.json | `http://127.0.0.1:8000/monthly-data/fixed_costs/upload-json` |
| sales.json | `http://127.0.0.1:8000/monthly-data/sales/upload-json` |

These endpoints write to **sales_forecasting**. All monthly workflows read
connection settings from the project `.env`: `MONTHLY_MONGO_URL` takes priority,
then the existing `MONGO_URL`, then `mongodb://localhost:27017` if neither is
configured. This supports the existing Atlas connection. The old Excel-import API continues to use its separate schema.
Upload all nine files before running monthly analysis/forecasting.

For **Body > raw > JSON**, use `POST /monthly-data/{collection}` without
`/upload-json`, and send a non-empty array, for example:

```json
[
  {
    "Month": {"$date": "2023-01-01T00:00:00Z"},
    "Product_ID": "P1",
    "Country_ID": "C1",
    "Units_Sold": 196,
    "Selling_Price": 214,
    "Discount_%": 5
  }
]
```

MongoDB Extended JSON dates and ISO date strings are converted to BSON dates.
Month must be the first day at midnight UTC. The complete payload is validated
before writes; numeric fields must be finite nonnegative JSON numbers, and
percentages cannot exceed 100. Duplicate natural keys within a file fail.
Imported `_id` fields are ignored. Natural-key upserts and unique indexes make
repeat uploads update the same records. MongoDB bulk writes are not a
transaction: a database-level failure can leave partial writes; retry the same
file after resolving the error. Responses report received/inserted/matched/
modified counts. This route imports data; analysis and forecasting remain
separate commands below.

## Annual 2026 production optimization

`ml/production_optimization.py` solves the requested annual linear program
using SciPy HiGHS. It maximizes the simple historical mean `Profit_per_unit`
times each product-country allocation, subject to annual forecast demand,
product capacity (`Capacity_per_month * 12`), shared labour and RM budget.

```powershell
uv sync --extra monthly-forecast --extra production-planning
.\.venv\Scripts\python.exe -m ml.production_optimization --forecast artifacts/monthly_demand/forecast_2026.csv --dataset ml_dataset.csv
```

For a forecast in the project root, omit `--forecast`. MongoDB defaults to
the configured monthly/app URI, database `sales_forecasting`. The script reads `products`,
`bom`, `rm_price`, `currency_rates`, `shipping_duty`, and `raw_materials`;
the latter supplies each material's currency. Default cost coefficients use
December 2025 prices and FX, held constant across 2026. Missing snapshot
prices/rates fail rather than substituting zero. INR uses a rate of 1.

Editable assumptions are at the top of the module, including 40,000 annual
labour hours, INR 20,000,000 (2 crore) annual RM budget, FX inflation and
demand reduction. CLI overrides include `--labour-hours`, `--rm-budget`,
`--cost-month`, `--mongo-uri`, `--database`, and `--output-dir`.

Base margins use the requested unweighted mean of all 36 historical monthly
profit-per-unit observations. Shipping, labour and import duty already enter
that historical profit and are not subtracted again. MongoDB shipping/duty
fields are validated and reported as references; current changes are not
repriced into the historical margin. Base RM budget uses the snapshot cost.
The USD/EUR +5% scenario increases only USD/EUR BOM costs in both budget use
and incremental profit deductions. The demand -10% scenario reduces all
annual demand caps. Each scenario is independently optimized.

Outputs in `artifacts/production_plan/` include product recommendations,
country summaries, all 50 product-country routing decisions, shadow prices
for every scenario, `scenario_comparison.csv`, assumptions JSON and a profit
bar chart. Changes are compared with 2025 actual units sold as a production
proxy. Country/product routing labels identify increases, decreases and zero
shipments. The final printed line identifies products with profitable
recommended increases, or states that no increase is supported. Exactly zero
margins do not justify a profit-based increase; no minimum output is imposed.

Variables are continuous, so fractional units are allowed. This annual model
does not schedule monthly capacity, inventory, lead times or fixed overhead.
Expected profit is contribution under the specified margin assumptions.
HiGHS minimizes negative profit, so reported maximum-profit shadow prices
reverse the solver marginal signs. Shadow prices have different units and
are local values. The most limiting constraint is ranked by actual profit
gain after a separate 1% RHS relaxation and re-solve, with local shadow-price
estimates also reported. Degenerate optima can have nonunique shadow prices.

The final step upserts one document per year/scenario into `production_plan`
with allocations, product/country tables, assumptions and sensitivity.
`--skip-save-mongo` disables writes while still reading source collections.
Local files remain available if MongoDB persistence fails.
Solver reference: [SciPy linprog](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linprog.html).

## Monthly demand forecast: Prophet, XGBoost and Random Forest

After training, view saved 2026 monthly predictions in a browser or Postman
using **GET** (no request body):

```text
http://127.0.0.1:8000/predictions/monthly?product_id=P1&country_id=C1
```

Add `&month=1` for January only. Omit filters for all 600 predictions. The
response includes the selected model, training period, forecast rows, total
units and range interpretation. This endpoint reads the saved monthly CSV
and metadata without retraining. It returns 404 for unmatched filters and
503 when the monthly forecast artifacts are missing or invalid.

To train demand models directly on the uploaded monthly `sales` collection
without requiring RM prices or profit calculations, run:

```powershell
.\.venv\Scripts\python.exe -m ml.monthly_demand_forecast --from-mongo-sales --skip-mongo
```

This reads only monthly sales dates, product/country IDs, units, prices and
discounts. It does not fabricate cost/profit columns or write an incomplete
`ml_dataset.csv`. The selected refitted model is saved separately as
`artifacts/monthly_demand/best_model.joblib` with training history, feature
columns and holdout metrics. For Prophet it contains per-series serialized
model JSON. Existing daily/weekly prediction API models remain separate.
The `--skip-mongo` flag keeps forecasts local; omit it to save forecasts in
the configured database's `forecast_2026` collection.

`ml/monthly_demand_forecast.py` is a separate monthly forecasting workflow with
notebook-style `# %%` cells, function docstrings and Hinglish comments.
Install the optional dependencies, generate the monthly input CSV, then run:

```powershell
uv sync --extra monthly-forecast
.\.venv\Scripts\python.exe -m ml.mongo_sales_analysis
.\.venv\Scripts\python.exe -m ml.monthly_demand_forecast --input ml_dataset.csv
```

Input must contain exactly 36 consecutive monthly observations (January 2023
through December 2025) for each of the 50 combinations of 5 products and
10 countries. Missing months and duplicate keys fail validation. IDs are read
as strings, preserving leading zeros. Negative profit is allowed; missing or
invalid required numeric values stop the run.

The comparison fits on 2023-2024 and forecasts all of 2025 from the December
2024 origin. Global tree models use grouped lags of 1/3/12 months, shifted
3/6-month rolling means, calendar features and training-only one-hot IDs.
Prophet fits a separate seasonal model per combination. Future price and
discount are held at each series' last 12 observed months' average. During
evaluation those averages use 2024 only; recursive lags use predictions after
the forecast origin. Real 2025 targets never enter 2025 forecast features.
`Profit_per_unit`, `RM_Cost` and `Shipping` are excluded from predictors.

Lowest pooled 2025 RMSE selects the winner, with MAE as tie-breaker; MAPE is
reported as a percentage excluding explicitly counted zero-actual rows.
Selection uses this holdout, so the winning score is not an independent final
test. The winner is then refitted through December 2025 to generate 600 rows
for January-December 2026. `Lower` and `Upper` are approximate empirical 90%
ranges using 2025 absolute-error quantiles per horizon across the 50 series.
They share the selection holdout, have no formal coverage guarantee, and do
not model uncertainty in future prices or discounts.

Outputs default to `artifacts/monthly_demand/`: `forecast_2026.csv` (requested
six columns), model comparison, all three models' test predictions, run
metadata, XGBoost importance values, and three matplotlib PNG charts. The
product forecast chart contains one panel per product and a line per country.
XGBoost importance is calculated from its 2023-2024 comparison model.
Override the directory with `--output-dir .` to save files in the project root.

The default final step upserts forecasts into
`sales_forecasting.forecast_2026` on the configured monthly/app URI, with a unique
month/product/country index. Repeated runs update existing records. IDs in
this collection are strings. Options include `--mongo-uri`, `--database` and
`--skip-mongo` for local artifacts only. MongoDB errors retain local files
and cause a nonzero exit status.

API references: [Prophet](https://facebook.github.io/prophet/docs/quick_start.html),
[XGBoost](https://xgboost.readthedocs.io/en/stable/python/python_api.html),
and [Random Forest](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html).

## Monthly MongoDB analysis (separate schema)

Run the requested monthly-schema reports from the project root:

```powershell
.\.venv\Scripts\python.exe -m ml.mongo_sales_analysis
```

The standalone module `ml/mongo_sales_analysis.py` connects to
the configured monthly/app URI, database `sales_forecasting`. This database is separate from
the API's `sales_forcasting` database. It expects the provided capitalized
fields and collections: `products`, `countries`, `sales`, `raw_materials`,
`bom`, `rm_price`, `currency_rates`, `shipping_duty`, and `fixed_costs`.
All `Month` values must be BSON dates on the first day of each month in UTC;
join IDs must have the same BSON type, and numeric fields must be numbers.

The command prints actual samples for all four sales aggregations, a
product-country profit summary, and the DataFrame. It writes `ml_dataset.csv`
in the current directory. CSV rows retain each monthly sales document so
price/discount information is preserved. `RM_Cost` and `Shipping` are per-unit
INR costs, and `Discount` is a percentage (10 means 10%).

Revenue is after discount and before import duty. Profit margin is
`100 * total profit / total revenue`, with null for zero revenue. Yearly
growth includes units and revenue: 2023 is the baseline, followed by
2024 versus 2023 and 2025 versus 2024. A missing year or zero base gives null.
INR materials use an exchange rate of 1; other materials require that month's
currency rate. Missing/duplicate cost joins stop export instead of treating
unknown costs as zero. The supplied profit formula excludes `fixed_costs`;
allocating factory overhead to products/countries needs an allocation rule.
`Capacity_per_month` and country currency are not needed for this calculation.

The nested joins use MongoDB's documented
[$lookup](https://www.mongodb.com/docs/manual/reference/operator/aggregation/lookup/)
syntax. A [$facet](https://www.mongodb.com/docs/manual/reference/operator/aggregation/facet/)
returns the summary and monthly rows from the same calculation. This is sized
for the stated 1,800 sales documents; facet output must fit MongoDB's 16 MiB
BSON document limit (and each facet's 100 MB memory limit).

FastAPI and MongoDB backend with daily or weekly units forecasts, linear
regression and Random Forest pipelines, and isolated product/region models.

## POST training and forecasts

Inspect `GET /forecast/all-summary` for collection counts and
`GET /forecast/sales-quality` for training readiness, date ranges, cadence,
missing observations, and duplicate sale IDs for every product/region.
The quality endpoint accepts optional `product_id` and `region` filters.
Forecast training reads the `sales` collection. Products, regional summaries,
users, and imported `sales_forecasting` rows remain separate datasets and
are available through `GET /forecast/{dataset}`; they are not combined as sales.

Train using `POST /predictions/train` with JSON:

```json
{
  "product_id": "119",
  "region": "North",
  "model_type": "linear_regression",
  "frequency": "auto",
  "test_fraction": 0.2
}
```

`model_type` accepts `linear_regression` (default), `random_forest`, or `auto`.
Linear regression uses a StandardScaler and LinearRegression pipeline, fit
only on the training split during evaluation. `auto` compares both models
and the seven-observation baseline by chronological holdout MAE. Explicit
model choices retain that method even when the baseline has lower error.
`frequency` accepts `auto`, `D`, or `7D`. `test_fraction` ranges from 0.1
to 0.4, with at least seven holdout observations. Responses include MAE,
RMSE, WAPE (null when actual demand totals zero), baseline MAE, comparison
points, and source-history counts. These are one-observation-ahead holdout
metrics, not an independent test of automatic model selection or long-horizon
accuracy. Forecast outputs are bounded at zero.

Then use `POST /predictions/future`:

```json
{
  "product_id": "119",
  "region": "North",
  "model_type": "linear_regression",
  "days": 28,
  "include_details": true
}
```

`days` is a strict integer from 1 to 90. Daily models return one prediction
per day; weekly models require multiples of seven and return one observation
per week (28 days = four forecasts). Weekly values are recorded observations,
not inferred daily demand. Predictions begin after the latest sales date,
which can be in the past when history is stale. The response reports
`requested_days`, `forecast_window_days`, per-date units, and total units.
Details include history, evaluation metrics, and the first forecast's lag inputs.
Unknown fields are rejected to prevent silently ignored request parameters.
Price, promotions, inventory, and discounts are not forecast inputs.

Models are saved atomically under `artifacts/models` using the product,
region, and requested model type as the key. Train each requested combination
first; a missing combination returns 503 instead of using another product's
model. Training also updates `artifacts/sales_model.joblib`, which is used by
the legacy GET dashboard and production-planning endpoints.

## Run locally (PowerShell)

Run commands from the project root. Install the dependencies with `uv sync`
if the virtual environment is not already set up. Configure `MONGO_URL` in
your local `.env` file and make sure MongoDB is reachable. The database name
used by the application is `sales_forcasting`.

```powershell
.\.venv\Scripts\python.exe -m uvicorn app.main:app --reload
```

Open <http://127.0.0.1:8000/docs> to upload data and try the API.
Open <http://127.0.0.1:8000/dashboard> for the graphical interface: sales
history, next-observation forecast, model and baseline MAE, feature inputs, and
actual versus predicted evaluation charts. Refresh reloads current data.
The interface uses local SVG charts without external chart libraries.
Its data is available at `GET /predictions/dashboard-data`.
Retrain after this update to save evaluation comparison points; older
artifacts still show history and forecasts but may lack the evaluation chart.

Upload the sales workbook through `POST /sales/upload-excel`, using the
`sales` or `file` form field. Required columns are `sale_id`, `sale_date`,
`product_id`, `region`, `quantity_sold`, `mrp_inr`, `discount_pct`, and
`net_revenue_inr`.

## Train and predict

1. Provide at least **42 consecutive observations** for the product
   and region you want to train: seven periods build lag features and at least
   35 usable periods remain. Quantities must be finite and nonnegative.
   Transactions on the same day are summed. Missing calendar dates remain
   unknown; add a confirmed zero-sales observation only when zero is correct.
   Training recognizes observations on a consistent weekly schedule (at least
   eight observations on the same weekday, with gaps in multiples of seven
   days and at least one seven-day gap). Other series use daily frequency.
   Missing weeks are retained as unknown and block training. Weekly dated
   values are treated as recorded units, not assumed to be whole-week totals.
2. Run training and select the row number for your product and region:

   ```powershell
   .\.venv\Scripts\python.exe -m ml.train
   ```

   Training reserves the latest 20% (at least seven periods) for one-period-ahead
   evaluation. MAE measures the average absolute error in units; lower is
   better. The baseline predicts the units recorded seven observation periods
   earlier (seven days for daily data, seven weeks for weekly data).
   CLI training uses linear regression by default and refits on all usable
   history after evaluation. Use the POST training endpoint with `auto` to
   compare linear regression, Random Forest, and the baseline. The saved
   artifact records `model_type`, `frequency`, evaluation errors, and
   comparison points. The dashboard identifies the selected method.
3. Generate a prediction from the command line or API:

   ```powershell
   .\.venv\Scripts\python.exe -m ml.predict
   Invoke-RestMethod http://127.0.0.1:8000/predictions/next-period
   ```

The response includes `product_id`, `region`, `last_sales_date`,
`forecast_date`, `predicted_units`, and `model_trained_through`.
The response also includes `frequency`, `forecast_horizon_days`, and `model_type`.
The forecast date is one day or seven days after the latest recorded date,
according to the saved frequency; it may be in the past if history is stale.
The latest seven observation periods must have complete sales observations.
`GET /predictions/next-day` remains available for daily models and rejects
weekly models with a message directing callers to `/predictions/next-period`.
Older artifacts without frequency metadata are treated as daily models.

HTTP 503 means the saved model is missing; 422 means the sales history
cannot support prediction; 500 means an unexpected prediction failure
(inspect server logs).

## Future forecasts and production planning

The dashboard includes a future forecast selector, date-by-date chart and
table, and a production planning form for the currently saved product/region.
No training changes are needed to use the existing daily or weekly artifact.

```text
GET /predictions/future?periods=1
GET /predictions/future?periods=4
GET /predictions/future?calendar_month=true
POST /predictions/production-plan
```

`periods` ranges from 1 to 90. For a weekly artifact, 1 means the next weekly
dated observation and 4 means the next four, not an exact calendar month.
For daily artifacts, 7 means the next seven days. Predictions start after
the latest recorded sales date, not today; upload current history when
forecasts start in the past. Later predictions feed back into lag features.
The saved MAE validates only one-period-ahead evaluation, not the recursive
multi-period forecasts. No confidence intervals are estimated.

`calendar_month=true` forecasts through the end of the month after the latest
sales date, then returns only observation dates within that month. For weekly
data this is a sum of dated observations, not prorated calendar-month demand.
We cannot infer whole-week totals from Monday-only records. Production
planning therefore requires the user to confirm that each observation is
whole-period demand, and only supports consecutive forecast periods from
the inventory date. Next-month-only demand excludes intervening demand and
is not accepted by the planner.

Example production request (inventory values below are illustrative):

```json
{
  "product_id": "120",
  "region": "South",
  "periods": 4,
  "units_are_period_totals": true,
  "stock_as_of": "2026-10-05",
  "current_stock": 10,
  "safety_stock": 5,
  "incoming_stock": 0,
  "available_capacity": 20
}
```

Use the saved product and region, the actual latest sales date for `stock_as_of`, and inventory applicable
to this product and region on that date. The form requires explicit stock
values; it does not assume `opening_stock` is current stock. Incoming stock
must be available within the window. Production is
`ceil(max(0, forecast_units + safety_stock - current_stock - incoming_stock))`.
Capacity is optional and covers the entire selected window; its shortfall
is reported without reducing the calculated requirement. Plans are not saved
and do not model production lead times, intra-window shortages, shared
regional stock, or a factory schedule.

## Verify

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Prediction tests use a temporary saved model and mocked sales data without
reading or changing MongoDB records.

## Current scope and next steps

POST forecasts use separate artifacts for each product, region, and requested
model type. Legacy GET endpoints use the most recently trained model.
Predictions are computed on request
and are not stored in MongoDB. Evaluation uses actual prior observations
at each held-out date, not recursive forecasts across multiple periods.

After checking a real prediction, track forecast versus actual units over
time and retrain as new history arrives. Extend to forecast persistence
and validated longer horizons when those
are needed. Only load model artifacts you trust.
