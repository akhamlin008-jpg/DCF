"""
data_layer.py — the ONLY module besides App.py that touches market data. Turns
raw financials into the metric scalars factor_core expects, plus a daily returns
matrix and a 10y-yield change series for risk_core.

WHAT CHANGED FOR RELIABILITY (vs the original)
----------------------------------------------
1. DISK CACHE: per-ticker fundamentals are cached on disk (cache_layer), so 100
   app opens/day = at most one live fetch per ticker per day, and the cache
   survives Codespaces/app restarts. This is the change that actually delivers
   "runs reliably even used 100x/day."
2. SHARED HARDENED SESSION + BACKOFF: every Yahoo call goes through net_layer
   (curl_cffi chrome impersonation + exponential backoff on 429).
3. LOW CONCURRENCY: MAX_WORKERS dropped 8 -> 2. Concurrency is what trips the
   limiter on datacenter IPs.
4. DROPPED THE WEAKEST SCRAPES BY DEFAULT: major_holders (institutional) and
   insider_transactions are off unless INCLUDE_HOLDERS_INSIDER=1. Your own
   honesty ledger flags these as the least reliable factors; the engine
   reweights around the resulting Nones.
5. OPTIONAL EDGAR FUNDAMENTALS: set USE_EDGAR=1 to source value/quality/growth
   inputs from SEC EDGAR (official, free, cloud-IP-friendly) for US filers,
   falling back to Yahoo for ADRs. Off by default so behavior is unchanged
   until you opt in and test.
6. STOOQ PRICE FALLBACK: if the batched Yahoo price download fails/empties, the
   returns matrix is rebuilt from Stooq (free, no key).

DATA-HONESTY LEDGER (unchanged in spirit)
-----------------------------------------
yfinance is unofficial scraped data; reliability is NOT uniform across factors.
Value/quality/growth/momentum/DCF are statement- or price-derived and solid;
short_interest/institutional/insider are stale/often-missing scrapes that are
down-weighted per name. Geographic revenue is NOT available — only HQ country /
reporting currency, labeled as a proxy. Each row carries a per-factor
`availability` map and now a `_source_fund` provenance tag.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import yfinance as yf

import dcf_core as core
import cache_layer as kv
import net_layer as nl

try:
    import edgar_layer as edgar
except Exception:  # edgar_layer optional
    edgar = None

# --------------------------------------------------------------------------- #
# config
# --------------------------------------------------------------------------- #
BENCHMARK = "SPY"
RATE_TICKER = "^TNX"
PRICE_LOOKBACK = "3y"
MAX_WORKERS = 2                  # was 8
DEFAULT_ERP = 0.045
DEFAULT_RF = 0.043
DEFAULT_TERMINAL = 0.025
CAGR_CAP = (-0.10, 0.25)
FALLBACK_G1 = 0.10
FUND_TTL = kv.DAY               # fundamentals refresh at most daily
PRICE_TTL = 12 * 3600          # prices refresh at most twice daily

TRADING_DAYS = 252
MOM_SKIP = 21
MOM_12 = 252
MOM_6 = 126

# Feature flags (env-overridable; defaults keep original behavior minus the two
# weakest scrapes).
INCLUDE_HOLDERS_INSIDER = os.environ.get("INCLUDE_HOLDERS_INSIDER", "0") == "1"
USE_EDGAR = os.environ.get("USE_EDGAR", "0") == "1"

try:
    from universe import NON_US_EDGAR
except Exception:
    NON_US_EDGAR = {"TSM", "ASML", "HSBC", "ARM"}


# --------------------------------------------------------------------------- #
# statement helpers (tolerant of yfinance row-name drift)
# --------------------------------------------------------------------------- #
def _val(df, *names):
    if df is None or getattr(df, "empty", True):
        return None
    for name in names:
        if name in df.index:
            s = df.loc[name].dropna()
            if not s.empty:
                return float(s.iloc[0])
    return None


def _series_oldest_first(df, *names):
    if df is None or getattr(df, "empty", True):
        return []
    for name in names:
        if name in df.index:
            s = df.loc[name].dropna()
            if not s.empty:
                return [float(v) for v in s.iloc[::-1].tolist()]
    return []


def _fcf_series(cf):
    if cf is None or getattr(cf, "empty", True):
        return []
    if "Free Cash Flow" in cf.index:
        s = cf.loc["Free Cash Flow"].dropna()
    else:
        ocf = next((n for n in ("Operating Cash Flow",
                                "Total Cash From Operating Activities")
                    if n in cf.index), None)
        cx = next((n for n in ("Capital Expenditure", "Capital Expenditures")
                   if n in cf.index), None)
        if not (ocf and cx):
            return []
        s = (cf.loc[ocf] + cf.loc[cx]).dropna()
    return [float(v) for v in s.iloc[::-1].tolist()]


def _cagr(series):
    vals = [v for v in series if v is not None]
    if len(vals) < 2 or any(v <= 0 for v in vals):
        if len(vals) >= 2 and vals[0] > 0 and vals[-1] > 0:
            yrs = len(vals) - 1
            return (vals[-1] / vals[0]) ** (1 / yrs) - 1
        return None
    return core.robust_cagr(vals, cap=(-10, 10))


# --------------------------------------------------------------------------- #
# per-ticker fetch (disk-cached) -> raw metric scalars
# --------------------------------------------------------------------------- #
def fetch_one(ticker: str) -> dict:
    """Disk-cached wrapper. Serves a fresh on-disk row when available; otherwise
    fetches live and caches the result. This is the request-count bottleneck, so
    caching it here is what makes heavy daily use survivable."""
    key = f"fund:{ticker}:{int(USE_EDGAR)}:{int(INCLUDE_HOLDERS_INSIDER)}"
    cached = kv.get(key, max_age_sec=FUND_TTL)
    if cached is not None:
        return cached
    row = _fetch_one_live(ticker)
    if not row.get("error"):
        kv.put(key, row)
    return row


def _fetch_one_live(ticker: str) -> dict:
    row = {"ticker": ticker, "avail": {}, "error": None, "_source_fund": "yahoo"}
    ef = None
    try:
        t = nl.ticker(ticker, yf)

        use_edgar_here = (USE_EDGAR and edgar is not None
                          and ticker.upper() not in NON_US_EDGAR)
        cf = bs = inc = None
        if not use_edgar_here:
            cf = nl.with_backoff(lambda: getattr(t, "cashflow", None))
            bs = nl.with_backoff(lambda: getattr(t, "balance_sheet", None))
            inc = nl.with_backoff(lambda: getattr(t, "income_stmt", None))

        fi = getattr(t, "fast_info", {}) or {}

        def fival(*keys):
            for k in keys:
                v = fi.get(k) if hasattr(fi, "get") else None
                if v:
                    return float(v)
            return None

        price = fival("last_price", "lastPrice")
        mktcap = fival("market_cap", "marketCap")
        shares = fival("shares", "shares_outstanding")
        if shares is None and mktcap and price:
            shares = mktcap / price

        if use_edgar_here:
            ef = edgar.fundamentals(ticker)
            if ef.get("error"):
                use_edgar_here = False
                cf = nl.with_backoff(lambda: getattr(t, "cashflow", None))
                bs = nl.with_backoff(lambda: getattr(t, "balance_sheet", None))
                inc = nl.with_backoff(lambda: getattr(t, "income_stmt", None))
            else:
                row["_source_fund"] = "edgar"

        if row["_source_fund"] == "edgar":
            fcf_series = ef.get("fcf_series") or []
            fcf = ef.get("_fcf")
            cash = ef.get("_cash")
            debt = ef.get("_debt")
            revenue = ef.get("_revenue")
            gross = ef.get("_gross")
            ebit = ef.get("_ebit")
            net_income = ef.get("_net_income")
            interest = ef.get("_interest")
            equity = ef.get("_equity")
            assets = ef.get("_assets")
            cfo = ef.get("_cfo")
            rev_series = ef.get("rev_series") or []
            eps_series = ef.get("eps_series") or []
        else:
            fcf_series = _fcf_series(cf)
            fcf = fcf_series[-1] if fcf_series else None
            cash = _val(bs, "Cash Cash Equivalents And Short Term Investments",
                        "Cash And Cash Equivalents And Short Term Investments",
                        "Cash And Cash Equivalents")
            debt = _val(bs, "Total Debt")
            if debt is None:
                ltd = _val(bs, "Long Term Debt") or 0.0
                std = _val(bs, "Current Debt", "Short Term Debt",
                           "Current Debt And Capital Lease Obligation") or 0.0
                debt = (ltd + std) or None
            revenue = _val(inc, "Total Revenue", "Operating Revenue")
            gross = _val(inc, "Gross Profit")
            ebit = _val(inc, "EBIT", "Operating Income")
            net_income = _val(inc, "Net Income", "Net Income Common Stockholders")
            interest = _val(inc, "Interest Expense", "Interest Expense Non Operating")
            equity = _val(bs, "Stockholders Equity",
                          "Total Equity Gross Minority Interest", "Common Stock Equity")
            assets = _val(bs, "Total Assets")
            cfo = _val(cf, "Operating Cash Flow",
                       "Total Cash From Operating Activities")
            rev_series = _series_oldest_first(inc, "Total Revenue", "Operating Revenue")
            eps_series = _series_oldest_first(inc, "Diluted EPS", "Basic EPS")

        ev = None
        if mktcap is not None:
            ev = mktcap + (debt or 0.0) - (cash or 0.0)

        # ---------------- VALUE ----------------
        row["fcf_yield"] = (fcf / mktcap) if (fcf and mktcap) else None
        row["earnings_yield"] = (net_income / mktcap) if (net_income and mktcap) else None
        row["ebit_ev"] = (ebit / ev) if (ebit and ev and ev > 0) else None
        row["sales_ev"] = (revenue / ev) if (revenue and ev and ev > 0) else None

        # ---------------- QUALITY ----------------
        row["gross_margin"] = (gross / revenue) if (gross and revenue) else None
        row["op_margin"] = (ebit / revenue) if (ebit and revenue) else None
        row["roe"] = (net_income / equity) if (net_income and equity and equity > 0) else None
        row["roa"] = (net_income / assets) if (net_income and assets and assets > 0) else None
        row["debt_to_equity"] = (debt / equity) if (debt is not None and equity and equity > 0) else None
        row["interest_coverage"] = (ebit / abs(interest)) if (ebit and interest) else None
        row["accruals"] = ((net_income - cfo) / assets) if (net_income is not None
                            and cfo is not None and assets and assets > 0) else None

        # ---------------- GROWTH ----------------
        row["rev_cagr"] = _cagr(rev_series)
        row["fcf_cagr"] = _cagr(fcf_series)
        row["eps_cagr"] = _cagr(eps_series)

        # ---------------- DCF inputs ----------------
        row["_fcf"], row["_cash"], row["_debt"] = fcf, cash, debt
        row["_shares"], row["_price"], row["_mktcap"] = shares, price, mktcap
        row["_beta"] = None
        row["_interest"], row["_revenue"] = interest, revenue
        row["fcf_series"] = fcf_series

        # ---------------- short interest / sector / geo (one .info scrape) ----
        si_pct = si_ratio = None
        row.setdefault("_country", None)
        row.setdefault("_currency", None)
        row.setdefault("sector", None)
        row.setdefault("industry", None)
        try:
            info = nl.with_backoff(t.get_info)
            if info:
                si_pct = info.get("shortPercentOfFloat")
                si_ratio = info.get("shortRatio")
                row["_country"] = info.get("country")
                row["_currency"] = info.get("financialCurrency") or info.get("currency")
                row["sector"] = info.get("sector")
                row["industry"] = info.get("industry")
        except Exception:
            pass
        row["short_pct_float"] = float(si_pct) if si_pct is not None else None
        row["days_to_cover"] = float(si_ratio) if si_ratio is not None else None

        # ---------------- institutional / insider (weakest; OFF by default) ----
        row["inst_own_pct"] = None
        row["insider_net_ratio"] = None
        if INCLUDE_HOLDERS_INSIDER:
            row["inst_own_pct"] = _scrape_institutional(t)
            row["insider_net_ratio"] = _scrape_insider(t)

    except Exception as e:
        row["error"] = str(e)

    row["avail"] = _row_availability(row)
    return row


def _scrape_institutional(t):
    try:
        mh = t.major_holders
        if mh is not None and not mh.empty:
            txt = mh.astype(str)
            for _, rrow in txt.iterrows():
                joined = " ".join(rrow.values).lower()
                if "institution" in joined:
                    for cell in rrow.values:
                        c = str(cell).replace("%", "").strip()
                        try:
                            return float(c) / (100.0 if float(c) > 1.5 else 1.0)
                        except ValueError:
                            continue
    except Exception:
        pass
    return None


def _scrape_insider(t):
    try:
        tx = t.insider_transactions
        if tx is not None and not tx.empty and "Shares" in tx.columns:
            txt_col = next((c for c in tx.columns
                            if c.lower() in ("transaction", "text")), None)
            buys = sells = 0.0
            for _, r in tx.iterrows():
                sh = float(r.get("Shares") or 0)
                label = str(r.get(txt_col, "")).lower() if txt_col else ""
                if "purchase" in label or "buy" in label:
                    buys += sh
                elif "sale" in label or "sell" in label:
                    sells += sh
            if buys + sells > 0:
                return (buys - sells) / (buys + sells)
    except Exception:
        pass
    return None


def _row_availability(row):
    import factor_core as fc
    out = {}
    for g, metrics in fc.GROUP_METRICS.items():
        if g == "dcf":
            out[g] = 1.0 if row.get("mos") is not None else 0.0
            continue
        present = sum(1 for k, _ in metrics if row.get(k) is not None)
        out[g] = present / len(metrics)
    return out


# --------------------------------------------------------------------------- #
# batched market data (disk-cached frames + Stooq fallback)
# --------------------------------------------------------------------------- #
def _univ_hash(syms):
    return hashlib.md5(",".join(sorted(syms)).encode()).hexdigest()[:10]


def _download_prices(syms):
    """Return (close_df, vol_df). Yahoo batch first; Stooq fallback on failure."""
    try:
        data = nl.with_backoff(
            yf.download, syms, period=PRICE_LOOKBACK, interval="1d",
            auto_adjust=True, progress=False, session=nl.SESSION)
        if data is not None and len(data):
            close = data["Close"] if "Close" in data else data
            vol = data["Volume"] if "Volume" in data else None
            if close is not None and not close.dropna(how="all").empty:
                return close, vol
    except Exception:
        pass
    # ---- Stooq fallback (free, no key) ----
    try:
        import pandas_datareader.data as web
        end = dt.date.today()
        start = end - dt.timedelta(days=3 * 366)
        px = web.DataReader(list(syms), "stooq", start, end)
        close = px["Close"].sort_index() if "Close" in px else px.sort_index()
        vol = px["Volume"].sort_index() if "Volume" in px else None
        return close, vol
    except Exception:
        return None, None


def fetch_market(tickers):
    """Daily adjusted returns (T x N), dollar ADV, betas vs SPY, and the daily
    10y-yield change aligned to the same dates. Frames are disk-cached so this
    survives restarts; betas/adv/dyield are recomputed cheaply from the frames."""
    syms = list(tickers) + [BENCHMARK]
    h = _univ_hash(syms)
    close = kv.get_df(f"market:close:{h}", max_age_sec=PRICE_TTL)
    vol = kv.get_df(f"market:vol:{h}", max_age_sec=PRICE_TTL)
    if close is None:
        close, vol = _download_prices(syms)
        if close is not None:
            kv.put_df(f"market:close:{h}", close)
            if vol is not None:
                kv.put_df(f"market:vol:{h}", vol)

    if close is None or getattr(close, "empty", True):
        empty = pd.DataFrame(index=pd.DatetimeIndex([]))
        return {"returns": empty, "betas": {tk: None for tk in tickers},
                "adv": {tk: None for tk in tickers},
                "dyield": pd.Series(dtype=float), "benchmark": BENCHMARK}

    rets = close.pct_change().dropna(how="all")

    betas = {}
    if BENCHMARK in rets:
        mkt = rets[BENCHMARK]
        var_m = mkt.var()
        for tk in tickers:
            betas[tk] = float(rets[tk].cov(mkt) / var_m) if (tk in rets and var_m) else None
    else:
        betas = {tk: None for tk in tickers}

    adv = {}
    if vol is not None:
        dollar = (close * vol).tail(60)
        for tk in tickers:
            adv[tk] = float(dollar[tk].mean()) if tk in dollar else None
    else:
        adv = {tk: None for tk in tickers}

    try:
        tnx = kv.get_df("market:tnx", max_age_sec=PRICE_TTL)
        if tnx is None:
            raw = nl.with_backoff(
                yf.download, RATE_TICKER, period=PRICE_LOOKBACK, interval="1d",
                auto_adjust=False, progress=False, session=nl.SESSION)
            tnx = raw[["Close"]] if (raw is not None and "Close" in raw) else None
            if tnx is not None:
                kv.put_df("market:tnx", tnx)
        y = tnx["Close"].squeeze()
        if float(np.nanmedian(y)) > 25:
            y = y / 10.0
        y = y / 100.0
        dyield = y.reindex(rets.index).ffill().diff()
    except Exception:
        dyield = pd.Series(0.0, index=rets.index)

    return {"returns": rets, "betas": betas, "adv": adv,
            "dyield": dyield, "benchmark": BENCHMARK}


def fetch_risk_free():
    return _safe_rf()


# --------------------------------------------------------------------------- #
# orchestration
# --------------------------------------------------------------------------- #
def load_universe(tickers, rf=None, erp=DEFAULT_ERP, terminal=DEFAULT_TERMINAL):
    """Parallel (low-concurrency) per-ticker fetch + batched market data, then
    fold in DCF margin-of-safety as the 'mos' factor. Return contract is
    identical to the original plus a `source_mix` provenance tally.
    Row order matches `tickers`."""
    rf = rf if rf is not None else _safe_rf()
    rows_by_tk = {}
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        for r in ex.map(fetch_one, tickers):
            rows_by_tk[r["ticker"]] = r

    mkt = fetch_market(tuple(tickers))
    betas = mkt["betas"]
    rets = mkt["returns"]
    close = (1 + rets).cumprod() if not rets.empty else rets

    for tk in tickers:
        row = rows_by_tk[tk]
        row["_beta"] = betas.get(tk)
        row["mom_12_1"] = _momentum(close, tk, MOM_12, MOM_SKIP)
        row["mom_6_1"] = _momentum(close, tk, MOM_6, MOM_SKIP)

        stock = {"analyst_g5": None,
                 "hist_cagr": core.robust_cagr(row.get("fcf_series") or [], cap=CAGR_CAP),
                 "beta": row["_beta"], "market_cap": row["_mktcap"],
                 "debt": row["_debt"], "interest_expense": row["_interest"],
                 "tax_rate": 0.21}
        assum, _ = core.derive_assumptions(stock, rf, erp, terminal, CAGR_CAP, FALLBACK_G1)
        mos = None
        if assum["r"] is not None and row.get("_fcf") and row.get("_shares"):
            res = core.two_stage_dcf(row["_fcf"], assum["g1"], assum["g2"],
                                     assum["gt"], assum["r"], row["_cash"],
                                     row["_debt"], row["_shares"])
            if not res.error and row.get("_price"):
                mos = (res.fair - row["_price"]) / row["_price"]
        row["mos"] = mos
        row["avail"] = _row_availability(row)

    rows = [rows_by_tk[tk] for tk in tickers]
    availability = _universe_availability(rows)
    source_mix = {}
    for r in rows:
        s = r.get("_source_fund", "yahoo")
        source_mix[s] = source_mix.get(s, 0) + 1
    return {"as_of": dt.datetime.now(), "rows": rows, "returns": rets,
            "dyield": mkt["dyield"], "betas": betas, "adv": mkt["adv"],
            "rf": rf, "availability": availability,
            "benchmark": mkt["benchmark"], "source_mix": source_mix}


def _momentum(close_df, tk, lookback, skip):
    if getattr(close_df, "empty", True) or tk not in close_df or len(close_df) < lookback + skip:
        return None
    s = close_df[tk].dropna()
    if len(s) < lookback + skip:
        return None
    p_now = s.iloc[-1 - skip]
    p_then = s.iloc[-1 - skip - lookback]
    if p_then <= 0:
        return None
    return float(p_now / p_then - 1.0)


def _universe_availability(rows):
    import factor_core as fc
    out = {}
    for g in fc.GROUP_METRICS:
        vals = [r["avail"].get(g, 0.0) for r in rows]
        out[g] = float(np.mean(vals)) if vals else 0.0
    return out


def _safe_rf():
    cached = kv.get("market:rf", max_age_sec=PRICE_TTL)
    if cached is not None:
        return cached
    try:
        t = nl.ticker(RATE_TICKER, yf)
        v = t.fast_info.get("last_price")
        v = float(v)
        v = v / 10.0 if v > 25 else v
        rf = v / 100.0
        kv.put("market:rf", rf)
        return rf
    except Exception:
        return DEFAULT_RF
