"""
net_layer.py — one shared, hardened network session for every Yahoo call, plus
a rate-limit-aware retry wrapper.

WHY
---
Yahoo Finance throttles/blocks request *patterns* (datacenter IPs especially).
Two cheap mitigations, applied in ONE place so App.py and data_layer.py can't
drift apart:

  1. A curl_cffi session that impersonates a real Chrome TLS fingerprint. This
     is not a silver bullet on a flagged cloud IP, but it measurably reduces
     hard 429s vs the default python-requests fingerprint.
  2. with_backoff(): exponential backoff + jitter on "Too Many Requests", so a
     transient rate-limit retries instead of failing the whole load.

If curl_cffi isn't installed, everything still works — yfinance just uses its
default session.
"""
from __future__ import annotations

import random
import time

try:
    from curl_cffi import requests as _cffi
    # impersonate="chrome" tracks a current Chrome build; if a future
    # curl_cffi/yfinance mismatch raises ImpersonateError, pin versions
    # (see requirements.txt) or change to a specific build like "chrome124".
    SESSION = _cffi.Session(impersonate="chrome")
except Exception:  # curl_cffi missing or impersonation unsupported
    SESSION = None


def ticker(symbol, yf):
    """yf.Ticker bound to the shared session when available."""
    return yf.Ticker(symbol, session=SESSION) if SESSION else yf.Ticker(symbol)


def is_rate_limit(exc: Exception) -> bool:
    s = str(exc).lower()
    return "too many requests" in s or "429" in s or "rate limit" in s


def with_backoff(fn, *args, tries: int = 4, base: float = 1.5, **kwargs):
    """Call fn(*args, **kwargs); on a rate-limit error, back off and retry.
    Returns fn's result, or re-raises the last non-rate-limit error, or None if
    every retry was rate-limited."""
    last = None
    for i in range(tries):
        try:
            return fn(*args, **kwargs)
        except Exception as e:  # noqa: BLE001 - we re-raise non-rate-limit below
            last = e
            if is_rate_limit(e) and i < tries - 1:
                time.sleep(base * (2 ** i) + random.random())
                continue
            if is_rate_limit(e):
                return None       # exhausted retries on a 429 — let caller degrade
            raise
    if last is not None and is_rate_limit(last):
        return None
    return None
