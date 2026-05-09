from __future__ import annotations

import json
import os
from pathlib import Path

from rinha_api.vectorize import DEFAULT_MCC_RISK, DEFAULT_NORMALIZATION


def resources_dir() -> Path:
    return Path(os.getenv("RINHA_RESOURCES_DIR", "resources"))


def index_dir() -> Path:
    return Path(os.getenv("RINHA_INDEX_DIR", str(resources_dir() / "index")))


def load_json_or_default(path: Path, default: dict[str, float]) -> dict[str, float]:
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as file:
        return {str(key): float(value) for key, value in json.load(file).items()}
