"""Application-owned SQL storage, separate from every customer source.

PostgreSQL is the production target. SQLite is supported for offline integration tests.
Schema creation is explicit through the worker's init command, never during a request.
"""
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import hmac
import os
import secrets

from sqlalchemy import (Column, Date, DateTime, Float, ForeignKeyConstraint, Integer,
                        Index, JSON, MetaData, String, Table, create_engine,
                        delete, insert, select, update)


metadata = MetaData()
tenants = Table("tenants", metadata, Column("tenant_id", String, primary_key=True))
tenant_profiles = Table("tenant_profiles", metadata,
    Column("tenant_id", String, primary_key=True),
    Column("company_name", String, nullable=False),
    Column("api_key_hash", String, nullable=False),
    Column("created_at", DateTime(timezone=True), nullable=False),
    ForeignKeyConstraint(["tenant_id"], ["tenants.tenant_id"]))
loads = Table("sales_loads", metadata,
    Column("tenant_id", String, primary_key=True), Column("load_id", String, primary_key=True),
    Column("loaded_at", DateTime(timezone=True), nullable=False),
    Column("status", String, nullable=False), Column("quality", JSON))
raw_sales = Table("raw_sales", metadata,
    Column("tenant_id", String, primary_key=True), Column("load_id", String, primary_key=True),
    Column("row_number", Integer, primary_key=True), Column("raw_payload", JSON, nullable=False),
    ForeignKeyConstraint(["tenant_id", "load_id"], ["sales_loads.tenant_id", "sales_loads.load_id"]))
upload_configs = Table("upload_configs", metadata,
    Column("tenant_id", String, primary_key=True), Column("load_id", String, primary_key=True),
    Column("config", JSON, nullable=False),
    ForeignKeyConstraint(["tenant_id", "load_id"], ["sales_loads.tenant_id", "sales_loads.load_id"]))
sales = Table("sales", metadata,
    Column("tenant_id", String, primary_key=True), Column("sale_date", Date, primary_key=True),
    Column("product_id", String, primary_key=True), Column("region", String, primary_key=True),
    Column("units_sold", Float, nullable=False), Column("load_id", String, nullable=False),
    ForeignKeyConstraint(["tenant_id", "load_id"], ["sales_loads.tenant_id", "sales_loads.load_id"]))
model_runs = Table("model_runs", metadata,
    Column("tenant_id", String, primary_key=True), Column("run_id", String, primary_key=True),
    Column("model_version", String, nullable=False), Column("load_id", String, nullable=False),
    Column("status", String, nullable=False), Column("trained_at", DateTime(timezone=True), nullable=False),
    Column("metadata", JSON, nullable=False),
    ForeignKeyConstraint(["tenant_id", "load_id"], ["sales_loads.tenant_id", "sales_loads.load_id"]))
forecasts = Table("forecasts", metadata,
    Column("tenant_id", String, primary_key=True), Column("run_id", String, primary_key=True),
    Column("product_id", String, primary_key=True), Column("region", String, primary_key=True),
    Column("target_date", Date, primary_key=True), Column("prediction_date", Date, nullable=False),
    Column("model_version", String, nullable=False), Column("predicted_units", Float, nullable=False),
    ForeignKeyConstraint(["tenant_id", "run_id"], ["model_runs.tenant_id", "model_runs.run_id"]))
Index("idx_forecast_tenant_product_target", forecasts.c.tenant_id, forecasts.c.product_id, forecasts.c.target_date)


class ForecastStore:
    def __init__(self, url=None):
        from dotenv import load_dotenv
        load_dotenv()
        url = url or os.environ.get("FORECAST_DATABASE_URL")
        if not url:
            raise ValueError("Set FORECAST_DATABASE_URL for the application-owned database")
        self.engine = create_engine(url, pool_pre_ping=True)

    def create_schema(self):
        metadata.create_all(self.engine)

    @contextmanager
    def tenant_lock(self, tenant_id):
        """Serialize full-snapshot writes and training for a tenant on PostgreSQL."""
        if self.engine.dialect.name != "postgresql":
            # SQLite is for local/offline use only; production must use PostgreSQL.
            yield
            return
        connection = self.engine.connect()
        locked = False
        try:
            locked = bool(connection.exec_driver_sql(
                "SELECT pg_try_advisory_lock(hashtextextended(%s, 0))", (tenant_id,)
            ).scalar_one())
            if not locked:
                raise RuntimeError(f"A forecasting operation is already active for tenant {tenant_id}")
            yield
        finally:
            if locked:
                connection.exec_driver_sql(
                    "SELECT pg_advisory_unlock(hashtextextended(%s, 0))", (tenant_id,)
                )
            connection.close()

    def register_tenant(self, tenant_id, company_name):
        key = secrets.token_urlsafe(32)
        with self.engine.begin() as conn:
            # Existing CLI tenants must not be claimed through public registration.
            conn.execute(insert(tenants).values(tenant_id=tenant_id))
            conn.execute(insert(tenant_profiles).values(
                tenant_id=tenant_id, company_name=company_name,
                api_key_hash=hashlib.sha256(key.encode()).hexdigest(),
                created_at=datetime.now(timezone.utc)))
        return {"tenant_id": tenant_id, "company_name": company_name, "api_key": key}

    def authenticate(self, tenant_id, key):
        with self.engine.connect() as conn:
            expected = conn.execute(select(tenant_profiles.c.api_key_hash).where(
                tenant_profiles.c.tenant_id == tenant_id)).scalar_one_or_none()
        return expected is not None and hmac.compare_digest(
            expected, hashlib.sha256(key.encode()).hexdigest())

    def read_upload_config(self, tenant_id, load_id):
        with self.engine.connect() as conn:
            return conn.execute(select(upload_configs.c.config).where(
                upload_configs.c.tenant_id == tenant_id,
                upload_configs.c.load_id == load_id)).scalar_one_or_none()

    def list_loads(self, tenant_id, limit=100, offset=0):
        with self.engine.connect() as conn:
            return [dict(row) for row in conn.execute(select(loads).where(
                loads.c.tenant_id == tenant_id).order_by(loads.c.loaded_at.desc(), loads.c.load_id)
                .limit(limit).offset(offset)).mappings()]

    def read_raw_page(self, tenant_id, load_id, limit=100, offset=0):
        with self.engine.connect() as conn:
            exists = conn.execute(select(loads.c.load_id).where(
                loads.c.tenant_id == tenant_id, loads.c.load_id == load_id)).first()
            if exists is None:
                return None
            return list(conn.execute(select(raw_sales.c.raw_payload).where(
                raw_sales.c.tenant_id == tenant_id, raw_sales.c.load_id == load_id)
                .order_by(raw_sales.c.row_number).limit(limit).offset(offset)).scalars())

    def read_forecasts(self, tenant_id, run_id=None, limit=100, offset=0):
        query = select(forecasts).where(forecasts.c.tenant_id == tenant_id)
        if run_id is not None:
            query = query.where(forecasts.c.run_id == run_id)
        with self.engine.connect() as conn:
            return [dict(row) for row in conn.execute(query.order_by(
                forecasts.c.run_id, forecasts.c.target_date, forecasts.c.product_id, forecasts.c.region)
                .limit(limit).offset(offset)).mappings()]

    def list_runs(self, tenant_id, limit=100, offset=0):
        with self.engine.connect() as conn:
            return [dict(row) for row in conn.execute(select(model_runs).where(
                model_runs.c.tenant_id == tenant_id).order_by(model_runs.c.trained_at.desc(), model_runs.c.run_id)
                .limit(limit).offset(offset)).mappings()]

    def save_raw(self, tenant_id, load_id, records, config=None):
        # Caller supplies full snapshots, not increments. Preserve every row before cleaning.
        with self.engine.begin() as conn:
            if conn.execute(select(tenants.c.tenant_id).where(tenants.c.tenant_id == tenant_id)).first() is None:
                conn.execute(insert(tenants).values(tenant_id=tenant_id))
            conn.execute(insert(loads).values(tenant_id=tenant_id, load_id=load_id,
                         loaded_at=datetime.now(timezone.utc), status="raw"))
            if config is not None:
                conn.execute(insert(upload_configs).values(tenant_id=tenant_id, load_id=load_id, config=config))
            if records:
                conn.execute(insert(raw_sales), [{"tenant_id": tenant_id, "load_id": load_id,
                             "row_number": i, "raw_payload": row} for i, row in enumerate(records)])

    def read_raw(self, tenant_id, load_id):
        with self.engine.connect() as conn:
            return list(conn.execute(select(raw_sales.c.raw_payload).where(
                raw_sales.c.tenant_id == tenant_id, raw_sales.c.load_id == load_id
            ).order_by(raw_sales.c.row_number)).scalars())

    def save_clean(self, tenant_id, load_id, frame, report):
        # Replace only this tenant's full snapshot, transactionally and repeatably.
        records = [{"tenant_id": tenant_id, "load_id": load_id, "sale_date": row.sale_date.date(),
                    "product_id": str(row.product_id), "region": str(row.region),
                    "units_sold": float(row.units_sold)} for row in frame.itertuples()]
        with self.engine.begin() as conn:
            conn.execute(delete(sales).where(sales.c.tenant_id == tenant_id))
            if records:
                conn.execute(insert(sales), records)
            conn.execute(update(loads).where(loads.c.tenant_id == tenant_id, loads.c.load_id == load_id)
                         .values(status="clean", quality=report))

    def quality_failure(self, tenant_id, load_id, report):
        with self.engine.begin() as conn:
            conn.execute(update(loads).where(loads.c.tenant_id == tenant_id, loads.c.load_id == load_id)
                         .values(status="failed", quality=report))

    def read_sales(self, tenant_id, load_id):
        import pandas as pd
        with self.engine.connect() as conn:
            frame = pd.read_sql(select(sales.c.sale_date, sales.c.product_id, sales.c.region, sales.c.units_sold)
                                .where(sales.c.tenant_id == tenant_id, sales.c.load_id == load_id), conn)
        frame.columns = [str(column) for column in frame.columns]
        frame["sale_date"] = pd.to_datetime(frame.sale_date)
        return frame

    def save_run(self, tenant_id, load_id, run_id, result, prediction=None):
        with self.engine.begin() as conn:
            conn.execute(insert(model_runs).values(tenant_id=tenant_id, load_id=load_id, run_id=run_id,
                         model_version=run_id, status="promoted" if result.get("passed") else result.get("status", "rejected"),
                         trained_at=datetime.now(timezone.utc), metadata=result))
            if prediction is not None:
                conn.execute(insert(forecasts), [{"tenant_id": tenant_id, "run_id": run_id,
                             "model_version": run_id, "prediction_date": datetime.now(timezone.utc).date(),
                             "target_date": row.sale_date.date(), "product_id": str(row.product_id),
                             "region": str(row.region), "predicted_units": float(row.predicted_units)}
                             for row in prediction.itertuples()])

    def accuracy(self, tenant_id):
        """Read and evaluate every historical version against available actuals, in the worker."""
        import numpy as np
        condition = ((forecasts.c.tenant_id == sales.c.tenant_id)
                     & (forecasts.c.product_id == sales.c.product_id)
                     & (forecasts.c.region == sales.c.region)
                     & (forecasts.c.target_date == sales.c.sale_date))
        query = select(forecasts.c.run_id, forecasts.c.model_version, forecasts.c.prediction_date,
                       forecasts.c.product_id, forecasts.c.region, forecasts.c.target_date,
                       forecasts.c.predicted_units, sales.c.units_sold).select_from(forecasts.join(sales, condition))
        with self.engine.connect() as conn:
            rows = [dict(row) for row in conn.execute(query.where(forecasts.c.tenant_id == tenant_id)).mappings()]
        versions = []
        for run_id in sorted({row["run_id"] for row in rows}):
            subset = [row for row in rows if row["run_id"] == run_id]
            actual = np.array([r["units_sold"] for r in subset])
            predicted = np.array([r["predicted_units"] for r in subset])
            error = np.abs(actual - predicted)
            versions.append({"run_id": run_id, "matched_rows": len(subset),
                             "mae": float(error.mean()),
                             "wape": float(error.sum()/np.abs(actual).sum()) if np.abs(actual).sum() else None})
        return {"tenant_id": tenant_id, "versions": versions, "rows": rows}
