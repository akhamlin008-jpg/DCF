"""
cache_layer.py — disk-persistent, daily-TTL key/value cache.

WHY THIS EXISTS
---------------
Streamlit's @st.cache_data is in-memory and per-function. It dies on every
Codespaces rebuild / app reboot, and the Factor page and Risk page each get
their OWN copy because they wrap different functions. That is why opening the
app 100x/day means re-hammering Yahoo: every cold start starts from nothing.

This cache lives on disk and is shared by every module that imports it, so a
ticker fetched once is served from a local file for the rest of the TTL window,
no matter how many times any page is opened or whether the process restarted.

Pure stdlib (json) for dict payloads; pandas/parquet helpers for frames. No
network, no streamlit — unit-testable and reusable from the batch snapshot job.

KEY SCHEME
----------
Use hierarchical string keys: "fund:NVDA", "edgar:facts:AAPL", "market:universe".
Keys are sanitized to safe filenames. Values must be JSON-serializable for
get/put; use get_df/put_df for DataFrames.
"""
from __future__ import annotations

import datetime as dt
import json
import os
import time
from pathlib import Path

# Point DCF_CACHE_DIR at a path that survives your environment's restarts.
# Under the repo (e.g. ".cache") persists in a committed Codespace; /tmp does not.
CACHE_DIR = Path(os.environ.get("DCF_CACHE_DIR", ".cache"))
CACHE_DIR.mkdir(parents=True, exist_ok=True)

DAY = 24 * 3600


def _safe(key: str) -> str:
    for ch in ('/', '\\', ':', ' ', '"', "'"):
        key = key.replace(ch, "_")
    return key


def _path(key: str, ext: str = "json") -> Path:
    return CACHE_DIR / f"{_safe(key)}.{ext}"


# --------------------------------------------------------------------------- #
# JSON payloads (dicts of floats / None / lists — e.g. a fetched ticker row)
# --------------------------------------------------------------------------- #
def get(key: str, max_age_sec: int = DAY):
    """Return cached value if present AND fresher than max_age_sec, else None."""
    p = _path(key)
    if not p.exists():
        return None
    if time.time() - p.stat().st_mtime > max_age_sec:
        return None
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def put(key: str, value) -> bool:
    try:
        _path(key).write_text(json.dumps(value, default=str))
        return True
    except Exception:
        return False


def delete(key: str) -> None:
    for ext in ("json", "parquet"):
        p = _path(key, ext)
        if p.exists():
            p.unlink()


# --------------------------------------------------------------------------- #
# DataFrame payloads (returns matrix, etc.) via parquet
# --------------------------------------------------------------------------- #
def get_df(key: str, max_age_sec: int = DAY):
    import pandas as pd
    p = _path(key, "parquet")
    if not p.exists():
        return None
    if time.time() - p.stat().st_mtime > max_age_sec:
        return None
    try:
        return pd.read_parquet(p)
    except Exception:
        return None


def put_df(key: str, df) -> bool:
    try:
        df.to_parquet(_path(key, "parquet"))
        return True
    except Exception:
        return False


# --------------------------------------------------------------------------- #
# Freshness introspection (used by the UI to show "data as of …")
# --------------------------------------------------------------------------- #
def as_of(key: str):
    for ext in ("json", "parquet"):
        p = _path(key, ext)
        if p.exists():
            return dt.datetime.fromtimestamp(p.stat().st_mtime)
    return None


def fresh(key: str, max_age_sec: int = DAY) -> bool:
    ts = as_of(key)
    return ts is not None and (time.time() - ts.timestamp()) <= max_age_sec


def age_seconds(key: str):
    ts = as_of(key)
    return None if ts is None else time.time() - ts.timestamp()
