# Local PostgreSQL: company registration, CSV and predictions

The SQL API uses the existing forecasting worker. MongoDB APIs remain separate.
SQLAlchemy and psycopg are installed by the `forecast-worker` extra. Run from
the repository root using Python 3.13 and the project virtual environment.

## Configure the local database

In pgAdmin, create a new dedicated database named `forecasting`. Add this value
to your existing `.env`, using your local PostgreSQL user and password:

```dotenv
FORECAST_DATABASE_URL=postgresql+psycopg://USER:PASSWORD@localhost:5432/forecasting
```

URL-encode special characters in credentials. Do not paste passwords into chat
or commit `.env`. An example is available in `.env.example`.

```powershell
uv sync --extra forecast-worker
.\.venv\Scripts\python.exe -m ml.forecasting.worker check
.\.venv\Scripts\python.exe -m ml.forecasting.worker init
.\.venv\Scripts\python.exe -m uvicorn app.main:app --reload
```

`check` executes `SELECT 1`. `init` explicitly creates missing tables; requests
never initialize the schema. The new `tenant_profiles` and `upload_configs`
tables are additive. `init` does not migrate or alter existing table definitions.
Keep the API process running and use another terminal for training.

Open `http://127.0.0.1:8000/docs` and find **PostgreSQL forecasting**.

## Register a company

`POST /sql/tenants` with JSON:

```json
{"tenant_id": "acme", "company_name": "Acme Ltd"}
```

The response returns an `api_key` once. Save it. Only its SHA-256 hash is stored.
Use Swagger's **Authorize** button to supply it as `X-Tenant-Key` for all of
that company's uploads and reads. A different tenant's key cannot access the
data. Tenant IDs already created by the CLI cannot be claimed by registration.
This registers a company and issues a company key; it does not create individual
user accounts or implement password login/key recovery.

## Upload a complete sales CSV

`POST /sql/tenants/acme/upload-csv` is multipart form-data:

- `file`: a UTF-8 `.csv` file, at most 10 MiB.
- `options`: JSON text, default `{}`.

Default columns are `sale_date`, `product_id`, `region`, `quantity_sold`.
Dates may be daily (`frequency: D`) or monthly (`frequency: MS`, the default).
The default source grain is one row per period/product/region.

For the existing monthly column names, use:

```json
{
  "mapping": {
    "date": "Month",
    "product": "Product_ID",
    "region": "Country_ID",
    "quantity": "Units_Sold"
  },
  "frequency": "MS",
  "horizon": 3,
  "candidate_models": ["random_forest", "hist_gb", "xgboost"],
  "source_grain": "period",
  "returns_policy": "reject"
}
```

For transaction CSVs, use `source_grain: transaction` and add a stable unique
transaction column through `mapping.record_id`, for example `sale_id`.
Duplicates with conflicting transaction IDs fail. Returns require an explicit
`returns_policy`: `reject`, `exclude`, or `subtract`.

Every upload is a **full snapshot**. Successful cleaning replaces only that
company's current clean sales. Upload the full history, not just new rows.
All original columns and string values, including leading-zero IDs and optional
business details, are retained in `raw_sales.raw_payload`. Only demand fields
are normalized into `sales`; extra columns are not automatically model inputs.

Invalid rows or missing periods return 422 with a `load_id`; raw data and the
failure report remain saved, and previous clean sales remain intact.

## Train and save predictions

A successful upload returns `load_id` and the exact training command. Run it
from the second terminal on the same computer/database:

```powershell
.\.venv\Scripts\python.exe -m ml.forecasting.worker run-upload --tenant-id acme --load-id LOAD_ID
```

Training reads the saved upload configuration and snapshot. It evaluates the
configured models, including sklearn `HistGradientBoostingRegressor` (`hist_gb`).
Monthly defaults require at least 27 complete observed months per series, with
aligned histories; daily defaults require at least 22 complete days.
Only a model beating the seasonal baseline on the final holdout is promoted.
A rejected run is saved in `model_runs` and produces no new forecasts.
Successful runs save models under `backend/models/acme/VERSION/` and versioned
predictions in the SQL `forecasts` table. Training is a CLI/worker operation,
not an HTTP request or an in-process background task.

## Read all details and predictions

All these endpoints require the company's `X-Tenant-Key` header:

| Endpoint | Result |
| --- | --- |
| `GET /sql/tenants/acme/loads` | Upload history, status and quality reports |
| `GET /sql/tenants/acme/loads/LOAD_ID/raw` | Original CSV rows with every column |
| `GET /sql/tenants/acme/runs` | Training runs and evaluation metadata |
| `GET /sql/tenants/acme/predictions` | Saved versioned predictions |

Lists accept `limit` (1–1000, default 100) and `offset` (default 0).
Predictions also accept `run_id` to select one model version. An empty list means
no matching saved predictions; uploading alone does not train a model.

## Verify offline

```powershell
.\.venv\Scripts\python.exe -B -m unittest discover -s tests -p test_tenant_forecasts.py -v
.\.venv\Scripts\python.exe -B -m unittest discover -s tests -p test_forecasting_worker.py -v
```

These tests use isolated SQLite databases. They cover registration, duplicate
tenant conflicts, access keys, cross-tenant isolation, CSV preservation, quality
failures, worker training and prediction readback. They do not prove a live
PostgreSQL connection; run `check` and `init` against your configured local DB.
