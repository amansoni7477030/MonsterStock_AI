"""Optional read-only dashboard. Needs the extras: ``pip install -r requirements-dashboard.txt``.

The core application never imports this package at module level; importing it without the extras fails with a
clear message.
"""

from __future__ import annotations

_OPTIONAL = frozenset({"fastapi", "starlette", "pydantic", "uvicorn", "httpx", "anyio"})

try:
    from dashboard.app import create_app, run_dashboard, serve
except ImportError as exc:
    if (exc.name or "").partition(".")[0] in _OPTIONAL:
        raise ImportError(
            "The dashboard needs its optional dependencies (fastapi, starlette, pydantic, uvicorn). "
            "Install them with: pip install -r requirements-dashboard.txt"
        ) from exc
    raise

__all__ = ["create_app", "run_dashboard", "serve"]
