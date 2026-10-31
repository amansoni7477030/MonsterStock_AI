"""Strict, canonical JSON helpers.

Persisted JSON must be valid (no ``NaN``/``Infinity``), deterministic (sorted keys) and free of numpy/pandas
types, otherwise snapshot hashes would be unstable and downstream consumers could choke.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any


def to_jsonable(obj: Any) -> Any:
    """Recursively convert to JSON-safe primitives. Non-finite floats become ``None``."""
    if obj is None or isinstance(obj, (bool, str)):
        return obj
    if isinstance(obj, int):
        return int(obj)
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, Decimal):
        value = float(obj)
        return value if math.isfinite(value) else None
    if isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, date):
        return obj.isoformat()
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set, frozenset)):
        items = sorted(obj, key=str) if isinstance(obj, (set, frozenset)) else obj
        return [to_jsonable(v) for v in items]
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return to_jsonable(dataclasses.asdict(obj))
    # numpy / pandas without importing them eagerly
    module = type(obj).__module__
    if module.startswith("numpy"):
        if hasattr(obj, "tolist"):
            return to_jsonable(obj.tolist())
        if hasattr(obj, "item"):
            return to_jsonable(obj.item())
    if module.startswith("pandas"):
        if hasattr(obj, "isoformat"):
            try:
                return obj.isoformat()
            except (ValueError, TypeError):
                return None
        if hasattr(obj, "to_dict"):
            return to_jsonable(obj.to_dict())
        if hasattr(obj, "tolist"):
            return to_jsonable(obj.tolist())
    if hasattr(obj, "item"):
        try:
            return to_jsonable(obj.item())
        except (ValueError, TypeError):
            pass
    return str(obj)


def dumps(obj: Any, *, pretty: bool = False) -> str:
    """Canonical JSON (sorted keys, compact unless ``pretty``)."""
    safe = to_jsonable(obj)
    if pretty:
        return json.dumps(safe, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False)
    return json.dumps(safe, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def loads(text: str | bytes | None, default: Any = None) -> Any:
    if text is None or text == "" or text == b"":
        return default
    try:
        return json.loads(text)
    except (ValueError, TypeError):
        return default


def content_hash(obj: Any) -> str:
    """SHA-256 of the canonical JSON form."""
    return hashlib.sha256(dumps(obj).encode("utf-8")).hexdigest()
