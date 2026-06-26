"""
data_layer.py — the ONLY module besides App.py that touches yfinance. Turns the
raw scraped financials into the metric scalars factor_core expects, plus a daily
returns matrix and a 10y-yield change series for risk_core.

DATA-HONESTY LEDGER (read before trusting any factor)
-----------------------------------------------------
yfinance is unofficial scraped Yahoo data. Reliability is NOT uniform across the
eight factors you asked for, and pretending it is would be the real failure:

  RELIABLE (statements / prices):
    value, quality, growth   - from income_stmt / balance_sheet / cashflow
    momentum                 - from adjusted price history
    dcf (margin of safety)   - from your existing dcf_core
  PARTIAL / OFTEN STALE:
    short_interest           - .info shortPercentOfFloat / shortRatio; updated
                               ~bimonthly by exchanges, frequently missing
    institutional            - .major_holders / .institutional_holders; a single
                               stale snapshot, % only, no clean change series
  WEAK / FREQUENTLY ABSENT:
    insider                  - .insider_transactions; schema drifts between
                               yfinance versions, often empty for large caps

  NOT AVAILABLE AT ALL via yfinance (do NOT fabricate):
    revenue-by-geography     - requires 10-K segment disclosure. We expose HQ
                               COUNTRY and REPORTING CURRENCY only, and label
                               them as such. They are a listing/domicile proxy,
                               not real geographic revenue exposure.

Every metric carries into the row dict as a plain float or None; the factor
engine reweights around the Nones. The `availability` summary reports per-factor
coverage so the UI can show you exactly how much evidence each score rests on.

This module is import-heavy (yfinance) and is NOT exercised by the offline test
suite; wrap its entry points in st.cache_data from the pages.
"""
from __future__ import annotations

import datetime as dt
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import yfinance as yf

import dcf_core as core

BENCHMARK = "SPY"
RATE_TICKER = "^TNX"
PRICE_LOOKBACK = "3y"
MAX_WORKERS = 8
DEFAULT_ERP = 0.045
DEFAULT_RF = 0.043
DEFAULT_TERMINAL = 0.025
CAGR_CAP = (-0.10, 0.25)
FALLBACK_G1 = 0.10

TRADING_DAYS = 252
MOM_SKIP = 21          # skip most recent month (reversal)
MOM_12 = 252
MOM_6 = 126


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
    """Log-linear CAGR over all points (reuses the robust estimator), uncapped
    here because growth FACTORS want the real magnitude before standardization."""
    vals = [v for v in series if v is not None]
    if len(vals) < 2 or any(v <= 0 for v in vals):
        # fall back to endpoint CAGR if signs allow
        if len(vals) >= 2 and vals[0] > 0 and vals[-1] > 0:
            yrs = len(vals) - 1
            return (vals[-1] / vals[0]) ** (1 / yrs) - 1
        return None
    return core.robust_cagr(vals, cap=(-10, 10))


# --------------------------------------------------------------------------- #
# per-ticker fetch -> raw metric scalars
# --------------------------------------------------------------------------- #

def fetch_one(ticker: str) -> dict:
    row = {"ticker": ticker, "avail": {}, "error": None}
    try:
        t = yf.Ticker(ticker)
        cf = getattr(t, "cashflow", None)
        bs = getattr(t, "balance_sheet", None)
        inc = getattr(t, "income_stmt", None)
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

        # --- statement levels ---
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

        ev = None
        if mktcap is not None:
            ev = mktcap + (debt or 0.0) - (cash or 0.0)

        # ---------------- VALUE (higher = cheaper = better) ----------------
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
        # Sloan accruals = (NetIncome - CFO) / TotalAssets ; lower is better
        row["accruals"] = ((net_income - cfo) / assets) if (net_income is not None
                            and cfo is not None and assets and assets > 0) else None

        # ---------------- GROWTH ----------------
        row["rev_cagr"] = _cagr(_series_oldest_first(inc, "Total Revenue", "Operating Revenue"))
        row["fcf_cagr"] = _cagr(fcf_series)
        row["eps_cagr"] = _cagr(_series_oldest_first(inc, "Diluted EPS", "Basic EPS"))

        # ---------------- DCF margin of safety ----------------
        row["_fcf"], row["_cash"], row["_debt"] = fcf, cash, debt
        row["_shares"], row["_price"], row["_mktcap"] = shares, price, mktcap
        row["_beta"] = None  # filled after batched beta
        row["_interest"], row["_revenue"] = interest, revenue
        row["fcf_series"] = fcf_series

        # ---------------- SHORT INTEREST (partial; .info scrape) ----------------
        si_pct = si_ratio = None
        try:
            info = t.get_info()
            si_pct = info.get("shortPercentOfFloat")
            si_ratio = info.get("shortRatio")
            row["_country"] = info.get("country")
            row["_currency"] = info.get("financialCurrency") or info.get("currency")
            row["sector"] = info.get("sector")
            row["industry"] = info.get("industry")
        except Exception:
            row["_country"] = row.get("_country")
            row["_currency"] = row.get("_currency")
            row["sector"] = row.get("sector")
        row["short_pct_float"] = float(si_pct) if si_pct is not None else None
        row["days_to_cover"] = float(si_ratio) if si_ratio is not None else None

        # ---------------- INSTITUTIONAL (partial snapshot) ----------------
        inst = None
        try:
            mh = t.major_holders
            if mh is not None and not mh.empty:
                # schema varies; search for the institutional % row
                txt = mh.astype(str)
                for _, rrow in txt.iterrows():
                    joined = " ".join(rrow.values).lower()
                    if "institution" in joined:
                        for cell in rrow.values:
                            c = str(cell).replace("%", "").strip()
                            try:
                                inst = float(c) / (100.0 if float(c) > 1.5 else 1.0)
                                break
                            except ValueError:
                                continue
                    if inst is not None:
                        break
        except Exception:
            pass
        row["inst_own_pct"] = inst

        # ---------------- INSIDER (weak; trailing net buy ratio) ----------------
        net_ratio = None
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
                    net_ratio = (buys - sells) / (buys + sells)   # in [-1, 1]
        except Exception:
            pass
        row["insider_net_ratio"] = net_ratio

    except Exception as e:
        row["error"] = str(e)

    # per-factor availability bookkeeping
    row["avail"] = _row_availability(row)
    return row


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
# batched market data: returns matrix, betas, ADV, rate changes
# --------------------------------------------------------------------------- #

def fetch_market(tickers):
    """One batched download -> daily adjusted returns (T x N), dollar ADV per
    name, regression betas vs SPY, and the daily change in the 10y yield aligned
    to the same dates. Returns a dict."""
    syms = list(tickers) + [BENCHMARK]
    data = yf.download(syms, period=PRICE_LOOKBACK, interval="1d",
                       auto_adjust=True, progress=False)
    close = data["Close"] if "Close" in data else data
    vol = data["Volume"] if "Volume" in data else None
    rets = close.pct_change().dropna(how="all")

    # betas vs SPY
    mkt = rets[BENCHMARK]
    var_m = mkt.var()
    betas = {}
    for tk in tickers:
        if tk in rets and var_m:
            betas[tk] = float(rets[tk].cov(mkt) / var_m)
        else:
            betas[tk] = None

    # dollar ADV (last ~60 sessions)
    adv = {}
    if vol is not None:
        dollar = (close * vol).tail(60)
        for tk in tickers:
            if tk in dollar:
                adv[tk] = float(dollar[tk].mean())
            else:
                adv[tk] = None

    # 10y yield daily change, aligned to return dates
    try:
        tnx = yf.download(RATE_TICKER, period=PRICE_LOOKBACK, interval="1d",
                          auto_adjust=False, progress=False)["Close"]
        y = tnx.squeeze()
        if float(np.nanmedian(y)) > 25:      # legacy x10 quoting
            y = y / 10.0
        y = y / 100.0                          # percent -> decimal yield
        dyield = y.reindex(rets.index).ffill().diff()
    except Exception:
        dyield = pd.Series(0.0, index=rets.index)

    return {"returns": rets, "betas": betas, "adv": adv,
            "dyield": dyield, "benchmark": BENCHMARK}


def fetch_risk_free():
    try:
        v = yf.Ticker(RATE_TICKER).fast_info.get("last_price")
        if v is None:
            return DEFAULT_RF
        v = float(v)
        return (v / 10.0 if v > 25 else v) / 100.0 * 100  # keep decimal below
    except Exception:
        return DEFAULT_RF


# --------------------------------------------------------------------------- #
# orchestration: rows with DCF margin-of-safety filled in
# --------------------------------------------------------------------------- #

def load_universe(tickers, rf=None, erp=DEFAULT_ERP, terminal=DEFAULT_TERMINAL):
    """Parallel per-ticker fetch + batched market data, then fold in the DCF
    margin of safety as the 'mos' factor. Returns
        {as_of, rows[list], returns[DataFrame], dyield, betas, adv, rf,
         availability[dict]}.
    Row order matches `tickers`."""
    rf = rf if rf is not None else _safe_rf()
    rows_by_tk = {}
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        for r in ex.map(fetch_one, tickers):
            rows_by_tk[r["ticker"]] = r

    mkt = fetch_market(tuple(tickers))
    betas = mkt["betas"]
    rets = mkt["returns"]

    # momentum from the returns matrix (price-based, reliable)
    close = (1 + rets).cumprod()
    for tk in tickers:
        row = rows_by_tk[tk]
        row["_beta"] = betas.get(tk)
        row["mom_12_1"] = _momentum(close, tk, MOM_12, MOM_SKIP)
        row["mom_6_1"] = _momentum(close, tk, MOM_6, MOM_SKIP)

        # DCF margin of safety using the existing engine
        stock = {"analyst_g5": None, "hist_cagr": core.robust_cagr(
                     row.get("fcf_series") or [], cap=CAGR_CAP),
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
    return {"as_of": dt.datetime.now(), "rows": rows, "returns": rets,
            "dyield": mkt["dyield"], "betas": betas, "adv": mkt["adv"],
            "rf": rf, "availability": availability,
            "benchmark": mkt["benchmark"]}


def _momentum(close_df, tk, lookback, skip):
    if tk not in close_df or len(close_df) < lookback + skip:
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
    try:
        v = yf.Ticker(RATE_TICKER).fast_info.get("last_price")
        v = float(v)
        v = v / 10.0 if v > 25 else v
        return v / 100.0
    except Exception:
        return DEFAULT_RF
