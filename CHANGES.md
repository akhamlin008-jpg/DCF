# CHANGES — reliability rewrite

Drop these files into your repo (they mirror your existing structure). Below is
exactly what changed, what was deliberately left alone, and what you must verify
before trusting numbers.

## What I did NOT touch (and why)

`dcf_core.py`, `factor_core.py`, `risk_core.py`, `Test_core.py`,
`test_factor_risk.py`, `Config`, `.devcontainer/devcontainer.json`.

These are pure, tested, network-free math (Ledoit-Wolf shrinkage, Euler risk
decomposition, rank-normal factor pipeline). Rewriting correct, tested code adds
risk and removes nothing. The reliability problem was entirely in the data/app
layer. **Both offline test suites still pass unchanged** after the rewrite.

## Files REWRITTEN

| File | What changed |
|------|--------------|
| `universe.py` | De-duplicated (was ~160 entries / ~45 unique → **40 unique**). `BRK.B`→`BRK-B`. Dropped invalid `SPCX`. Now the single shared ticker list + theme + freshness CSS. |
| `data_layer.py` | Per-ticker fundamentals **disk-cached**; all Yahoo calls via the shared backoff session; `MAX_WORKERS` 8→2; optional EDGAR fundamentals + Stooq price fallback; the two weakest scrapes (holders/insider) off by default; added `_source_fund` provenance + `source_mix`. **Row schema unchanged** — the factor/risk engines consume it identically. |
| `App.py` | Shares the universe; per-ticker fetch disk-cached + session-memoized; hardened session; `MAX_WORKERS` 8→3; `use_container_width`→`width="stretch"`; freshness line in header; Refresh button now also clears the disk cache. |
| `pages/2_factor_analysis.py` | `width="stretch"` fix; freshness + source-provenance banner; empty-returns guard. |
| `pages/3_risk_report.py` | `width="stretch"` fix; provenance caption; empty-data guard (`st.stop()`). |
| `requirements.txt` | Added `curl_cffi`, `requests`, `pandas-datareader`, `pyarrow`; pinned `yfinance>=0.2.65`. |

## Files ADDED

| File | Purpose |
|------|---------|
| `cache_layer.py` | Disk-persistent, daily-TTL cache shared by every page. **This is what makes 100 opens/day survivable** — survives restarts, unlike `st.cache_data`. |
| `net_layer.py` | One shared `curl_cffi` chrome-impersonating session + exponential-backoff-on-429 wrapper. Graceful fallback if `curl_cffi` is absent. |
| `edgar_layer.py` | SEC EDGAR fundamentals (free, official, no key, cloud-IP-friendly). **Opt-in.** |
| `fetch_snapshot.py` | Batch refresh for the scheduled job; populates the disk cache. |
| `.github/workflows/refresh.yml` | Scheduled GitHub Action: fetch a few times/day, commit `.cache/`, so the app reads pre-fetched files and makes **zero** live calls. |
| `.gitignore` | Ignores `__pycache__`; deliberately does NOT ignore `.cache/` (the Action commits it). |

## Default behavior (conservative)

Out of the box the app behaves like your original **plus** disk caching, backoff,
lower concurrency, and the two weakest factors off — i.e. it still uses Yahoo,
so nothing new can break. The bigger wins are opt-in via env vars:

```bash
USE_EDGAR=1                       # route US fundamentals to SEC EDGAR
EDGAR_USER_AGENT="You you@email"  # REQUIRED for EDGAR or it 403s
INCLUDE_HOLDERS_INSIDER=1         # re-enable the weak institutional/insider scrapes
DCF_CACHE_DIR=.cache              # where the cache lives (default .cache)
```

## ⚠️ What you MUST verify before trusting numbers (I could not test these here)

My build sandbox can reach PyPI/GitHub but **not** `data.sec.gov`, Stooq, or
Yahoo, so the network paths are written-and-logic-tested but **not live-tested**:

1. **EDGAR.** Set a real `EDGAR_USER_AGENT` email, then run
   `python edgar_layer.py` (calls `edgar_smoke_test()`). Confirm AAPL returns
   sane revenue/FCF. The XBRL tag fallback lists in `edgar_layer.py` may need
   tuning per company. EDGAR is **US filers only** — `TSM/ASML/HSBC/ARM` stay on
   Yahoo automatically (`NON_US_EDGAR` in `universe.py`).
2. **Stooq fallback.** Verify your tickers resolve on Stooq (symbol/suffix
   coverage differs from Yahoo). It only triggers if the Yahoo batch download
   fails.
3. **The Action.** Put your email in `refresh.yml`, enable Actions, run it once
   via "Run workflow", confirm it commits `.cache/`. (Actions is free for public
   repos; check current minutes for private.)
4. **Spot-check valuations** for 2–3 names against the 10-K, exactly as your own
   disclaimers already say.

## Suggested order of attack

1. Replace the files, `pip install -r requirements.txt`, run as-is (de-dup +
   disk cache + backoff already make heavy daily use survivable).
2. Set `EDGAR_USER_AGENT`, run the EDGAR smoke test, then flip `USE_EDGAR=1`.
3. Wire up the Action; once `.cache/` is committed, the live app stops calling
   Yahoo entirely.

All offline tests pass; `python Test_core.py` and `python test_factor_risk.py`
to confirm on your end.
