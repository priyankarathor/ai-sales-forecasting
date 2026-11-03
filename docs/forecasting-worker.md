# Forecasting worker: first backend implementation

The new worker lives in `ml/forecasting/`. It reads a tenant's CSV or MongoDB
snapshot, preserves raw rows in our own SQL database, separately cleans them,
reads back saved clean sales, evaluates models, and stores versioned predictions.
Existing HTTP routes and existing monthly artifacts have not been migrated to
this worker. Run it from a persistent worker/CLI; do not call training from GET.

The additive `/sql` API now supports company registration, CSV uploads and
saved forecast reads. See [PostgreSQL API](postgresql-api.md). Uploaded snapshots
can be trained using `run-upload --tenant-id TENANT --load-id LOAD_ID`.

## Install and initialize

From the repository root, use Python 3.13:

```powershell
uv sync --extra forecast-worker
# Alternative, inside the existing venv:
python -m pip install -r requirements-forecast-worker.txt

# Supply an application-owned PostgreSQL connection through the environment.
# Example format only; substitute your credentials outside Git.
$env:FORECAST_DATABASE_URL = 'postgresql+psycopg://USER:PASSWORD@HOST:5432/forecasting'
python -m ml.forecasting.worker init
```

Use a **new, dedicated database**. `init` creates the initial schema explicitly;
it is not a migration system for previously deployed schemas. Subsequent schema
changes will need versioned migrations. Tables are `tenants`, `sales_loads`,
`raw_sales`, `sales`, `model_runs`, and `forecasts`. A missing region is stored as
an empty string, avoiding nullable primary-key problems. Quantities allow
fractional units for businesses selling weight/volume.

SQLite is available for offline verification only:

```powershell
$env:FORECAST_DATABASE_URL = 'sqlite:///artifacts/forecast_worker/demo.sqlite'
New-Item -ItemType Directory -Force artifacts/forecast_worker | Out-Null
python -m ml.forecasting.worker init
```

## Demo and customer data

```powershell
python -m ml.forecasting.worker demo
python -m ml.forecasting.worker run --config backend/configs/demo.json
```

The demo generates 60 months for two products and two regions, with trend,
seasonality, and seeded noise. A successful synthetic run does not establish
accuracy on customer data.

`backend/configs/monthly_mongo.example.json` maps the existing monthly schema.
Set `MONTHLY_READONLY_MONGO_URL` to credentials restricted to reads, then run:

```powershell
python -m ml.forecasting.worker run --config backend/configs/monthly_mongo.example.json
```

All source operations are reads. The connector cannot itself restrict a
credential's database permissions: provision a read-only source user. A shared
source collection must have the appropriate tenant restriction in `source.filter`.
One config represents one tenant, with credentials referenced by environment
variable name. The initial connectors support CSV and MongoDB; SQL customer
connectors, including MySQL/MSSQL, remain a later phase.

## Data contract and repeatable cleaning

Each pull is a **complete snapshot**, not an incremental feed. Raw CSV values
are strings, including leading-zero IDs, with every original column retained.
MongoDB payloads preserve BSON types using Extended JSON; values are decoded
only in a copy passed to cleaning. A unique `load_id` identifies each snapshot.

The cleaner standardizes dates to monthly starts (`MS`) or daily dates (`D`),
trims identifiers, parses quantities, and aggregates transactions. Product IDs
remain case sensitive. Dates are interpreted in UTC in this initial release;
tenant business timezone support is still needed for local daily boundaries.

`source_grain: period` requires one row per product/region/period. Duplicates are
ambiguous and fail. `source_grain: transaction` requires a stable `record_id`
column in mapping. Repeated IDs with identical fields are deduplicated;
conflicting IDs fail. Distinct transactions sharing a date/product/region are
summed, including transactions with otherwise identical values.

Negative quantities require an agreed `returns_policy`: `reject`, `exclude`, or
`subtract`. Negative net demand in a period fails instead of being silently
clipped. Cancellation handling, unit conversion, and aliases must currently be
resolved upstream; no business rule is inferred.

Invalid rows and missing periods are reported and block the worker. Missing
periods are never filled with zeros. Outliers are flagged and retained. Models
currently require a complete panel with aligned start/end dates; cold-start
products and partial histories need explicit handling before training.

Quality reports are saved in `sales_loads.quality`. A quality failure keeps the
previous clean snapshot intact and preserves the newly pulled raw rows.
Re-clean an existing snapshot without re-reading the customer's database:

```powershell
python -m ml.forecasting.worker clean --config backend/configs/demo.json --load-id LOAD_ID
python -m ml.forecasting.worker run --config backend/configs/demo.json --load-id LOAD_ID
```

PostgreSQL advisory locks serialize upload/cleaning and training operations for
the same tenant. A concurrent operation receives a conflict instead of racing
the tenant's full-snapshot clean table. SQLite does not provide this protection
and is only for offline verification. Chunked ingestion is still needed for
larger snapshots.

## Honest model evaluation

Features include product/region, shifted lags, shifted rolling means, and calendar
values. Prices, profit and website traffic are excluded from this initial model,
so no unknown future actual values enter inference.

Configured RandomForest, HistGradientBoosting, and optional XGBoost candidates
are compared on multiple earlier rolling-origin folds. Each fold predicts the
entire configured horizon recursively. The last horizon remains untouched
during candidate selection, then gates the winning candidate against a recursive
seasonal-naive baseline. No random split is used. WAPE is the selection/gate
metric; MAE and RMSE are also recorded. WAPE is stored as a ratio (0.10 = 10%).
All-zero evaluation windows have undefined WAPE and cannot justify promotion.
Missing XGBoost logs a warning and skips that candidate.

Forecast dates follow the latest observed period, not today's date. Stale source
history therefore produces forecasts for an old horizon; refresh source data
before using results for current purchasing or production decisions.

Only a winner strictly beating the baseline on the final holdout is refitted
on all history and saved. A rejected run writes a `model_runs` entry and writes
no model or forecast. The prior promoted version remains in history. The worker
does not automatically generate replacement forecasts from that older model.

Each successful run saves a unique immutable folder:
`backend/models/{tenant_id}/{model_version}/`, containing the fitted pipeline,
metadata (configuration, features, cutoff, parameters, package versions, seed,
backtests, final metrics), and optional forecast CSV. The database is the
authoritative forecast store. An artifact from a failed database transaction
is not promoted and must not be loaded by directory timestamp alone.

## Accuracy after actual sales arrive

Ingest a new full snapshot containing the actuals, then run:

```powershell
python -m ml.forecasting.worker accuracy --config backend/configs/demo.json --output artifacts/forecast_worker/accuracy.json
```

The worker joins saved forecasts to actuals by **tenant, product, region, and
target date**, and reports metrics separately per forecast run/version. Old
forecasts are never overwritten. Only dates with observed actuals are scored;
report `matched_rows` alongside the metrics. Current accuracy is recalculated
against the latest clean actuals, which may include source revisions. This
command writes a JSON report; accuracy dashboard integration remains a later phase.

## Verification and remaining work

```powershell
python -B -m unittest discover -s tests -p test_forecasting_worker.py -v
python -B -m unittest discover -s tests -v
```

Tests cover repeatable cleaning, duplicate IDs vs legitimate transactions,
returns, missing periods, leakage, recursive prediction, chronological splits,
baseline ties, zero denominators, optional XGBoost, tenant isolation, version
history, and regional accuracy joins. SQL integration uses isolated SQLite.
Real PostgreSQL integration must be verified with `FORECAST_DATABASE_URL` before
deployment. Authenticated job APIs, scheduler, monitoring/drift alerts, row-level
security, versioned migrations, backups, traffic features and the Next.js UI are
not implemented by this worker phase.

## Production deployment gate

Run FastAPI and the persistent worker/CLI as separately managed processes. Do
not run training in a serverless function or inside an API request. Before
exposing a deployment to customers:

1. Use PostgreSQL on a persistent host. Apply schema changes through a reviewed,
   versioned migration process; `init` only creates missing tables and is not a
   migration tool.
2. Set `FORECAST_DATABASE_URL`, source MongoDB URI environment variables, and
   `CORS_ORIGINS` in the deployment secret/configuration manager. Use a database
   role with only required privileges and source MongoDB users restricted to
   read-only access. Never ship `.env`.
3. Keep model artifacts on persistent, access-controlled storage. The database
   is authoritative for promotion. Back up the database and model artifact
   volume, and rehearse restoration.
4. Configure probes: `GET /health/live` checks process liveness and
   `GET /health/ready` checks the forecasting database. Readiness returns 503
   when that database is unavailable.
5. Run persistent workers. PostgreSQL advisory locks prevent same-tenant
   training and upload/cleaning overlap. Monitoring, scheduling, alerting, and
   capacity management must be supplied by the deployment host.
6. Verify staging with a real PostgreSQL instance, representative tenant data,
   a restore test, and a reviewed business acceptance threshold. The synthetic
   demo and SQLite checks do not establish production accuracy or availability.

This repository does not include an infrastructure-specific deployment
manifest or migration history. Production readiness depends on completing the
host-specific operations above; application code alone cannot certify a live
deployment.

Implementation references: [scikit-learn leakage and pipelines](https://scikit-learn.org/stable/common_pitfalls.html)
and [SQLAlchemy transaction contexts](https://docs.sqlalchemy.org/en/20/core/connections.html#connect-and-begin-once-from-the-engine).
