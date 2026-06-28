"""
fetch_snapshot.py — batch data refresh, run on a schedule (e.g. GitHub Actions),
NOT inside the Streamlit request path.

It calls the same data_layer.load_universe() the app uses, which populates the
on-disk cache (.cache/*.json fundamentals + *.parquet price frames). Commit that
cache (see .github/workflows/refresh.yml) and the app reads it instantly with
zero live data calls — so opening the app 100x/day triggers no network at all.

This is the architecture that makes app reliability independent of Yahoo: a
batch job only has to succeed ONCE per schedule (and retries on the next run),
whereas a live app must succeed on EVERY page load.

Run locally:   python fetch_snapshot.py
Env toggles:   USE_EDGAR=1  INCLUDE_HOLDERS_INSIDER=1  DCF_CACHE_DIR=.cache
"""
from __future__ import annotations

import datetime as dt
import json
import sys

import cache_layer as kv
import data_layer as dl
from universe import TICKERS


def main() -> int:
    started = dt.datetime.now()
    print(f"[snapshot] {started:%Y-%m-%d %H:%M} · {len(TICKERS)} tickers · "
          f"USE_EDGAR={dl.USE_EDGAR} HOLDERS={dl.INCLUDE_HOLDERS_INSIDER}")

    payload = dl.load_universe(TICKERS)

    rows = payload["rows"]
    loaded = sum(1 for r in rows if not r.get("error"))
    failed = [r["ticker"] for r in rows if r.get("error")]
    rets = payload["returns"]

    meta = {
        "as_of": started.isoformat(),
        "tickers": list(TICKERS),
        "loaded": loaded,
        "failed": failed,
        "source_mix": payload.get("source_mix", {}),
        "price_rows": int(len(rets)) if rets is not None else 0,
        "price_cols": int(rets.shape[1]) if (rets is not None and not rets.empty) else 0,
    }
    kv.put("snapshot:meta", meta)

    print(f"[snapshot] loaded {loaded}/{len(TICKERS)} · "
          f"sources={meta['source_mix']} · "
          f"prices {meta['price_rows']}x{meta['price_cols']}")
    if failed:
        print(f"[snapshot] failed: {', '.join(failed)}")

    # Exit non-zero only on a total wipeout, so the Action surfaces real outages
    # but tolerates a few thin names.
    if loaded == 0:
        print("[snapshot] ERROR: zero tickers loaded — data sources unreachable.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
