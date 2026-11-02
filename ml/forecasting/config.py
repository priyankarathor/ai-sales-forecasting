"""Validated tenant configuration; credentials are environment references."""
from dataclasses import dataclass
import json
from pathlib import Path
import re


@dataclass(frozen=True)
class TenantConfig:
    tenant_id: str
    source: dict
    mapping: dict
    frequency: str = "MS"
    horizon: int = 3
    backtest_folds: int = 2
    candidate_models: tuple = ("random_forest", "hist_gb", "xgboost")
    returns_policy: str = "reject"
    source_grain: str = "period"
    random_state: int = 42

    def __post_init__(self):
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", self.tenant_id):
            raise ValueError("tenant_id must be a safe identifier, max 64 characters")
        if self.frequency not in {"D", "MS"}:
            raise ValueError("frequency must be D (daily) or MS (monthly)")
        if not 1 <= self.horizon <= (3 if self.frequency == "MS" else 30):
            raise ValueError("Use a short horizon: 1-3 months or 1-30 days")
        if not 2 <= self.backtest_folds <= 5:
            raise ValueError("backtest_folds must be between 2 and 5")
        if not self.candidate_models or set(self.candidate_models) - {"random_forest", "hist_gb", "xgboost"}:
            raise ValueError("Unsupported candidate_models")
        if self.returns_policy not in {"reject", "exclude", "subtract"}:
            raise ValueError("returns_policy must be reject, exclude or subtract")
        if self.source_grain not in {"period", "transaction"}:
            raise ValueError("source_grain must be period or transaction")
        if {"date", "product", "quantity"} - self.mapping.keys():
            raise ValueError("mapping requires date, product and quantity")
        if self.source_grain == "transaction" and not self.mapping.get("record_id"):
            raise ValueError("Transaction sources require a stable record_id mapping")
        if self.source.get("type") not in {"csv", "mongo"}:
            raise ValueError("Supported source types: csv, mongo")
        if {"password", "uri", "user", "api_key"} & self.source.keys():
            raise ValueError("Source credentials must use environment references")

    @property
    def season(self):
        return 12 if self.frequency == "MS" else 7

    @classmethod
    def load(cls, path):
        return cls(**json.loads(Path(path).read_text(encoding="utf-8")))
