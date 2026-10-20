"""Stable, isolated model locations and atomic artifact replacement."""
import hashlib
import json
import os
from pathlib import Path
import tempfile

import joblib


def model_path(product_id: str, region: str, model_type: str) -> Path:
    key = json.dumps([product_id, region, model_type], ensure_ascii=False).encode()
    return (Path(__file__).resolve().parents[1] / "artifacts" / "models"
            / f"{hashlib.sha256(key).hexdigest()}.joblib")


def save_artifact(artifact: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, suffix=".joblib")
    os.close(descriptor)
    try:
        joblib.dump(artifact, temporary)
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)
