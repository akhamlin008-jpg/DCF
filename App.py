"""
Two-Stage DCF — scalable, autofilled, editable.

Design notes for the next person (or you in six months):
  * All finance math lives in dcf_core.py (no streamlit/yfinance there) so it's
    unit-testable offline and reusable from a batch scaler.
  * Data layer is PARALLEL (ThreadPoolExecutor) and cached PER TICKER, so adding
    a symbol re-fetches only that symbol instead of nuking the whole cache.
  * Betas come from ONE batched price download (returns regression vs SPY),
    which replaces ~N slow `Ticker.info` scrapes — the old bottleneck.
  * Each ticker panel is an st.fragment: editing one stock's inputs reruns only
    that card, not all of them. This is what lets the page scale past a handful.

yfinance is unofficial scraped data and renames fields between versions. Verify
FCF / cash / debt / shares against the latest 10-K (SEC EDGAR) before trusting
any single number. Field-name drift shows up as "missing"; widen the row-name
lists in _statement_value() / _extract_fcf_series() when it does.
"""
from __future__ import annotations

import datetime as dt
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import streamlit as st
import yfinance as yf

import dcf_core as core

try:
    # Lets worker threads keep Streamlit's script context (silences warnings
    # and keeps cache_data behaving inside the pool).
    from streamlit.runtime.scriptrunner import add_script_run_context
except Exception:  # pragma: no cover - older/newer streamlit
    def add_script_run_context(thread, ctx=None):  # type: ignore
        return thread

# --- Configuration ----------------------------------------------------------- #
TICKERS = [
    "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN",
    "META", "AVGO", "TSLA", "LLY", "WMT",
    "JPM", "V", "ORCL", "XOM", "MA",
    "COST", "HD", "PG", "JNJ", "NFLX",
    "MU", "SNDK", "WDC", "STX", "LRCX", "AMD", "PANW", "NOW",
]
BENCHMARK = "SPY"

DEFAULT_TERMINAL = 0.025     # 2.5%
DEFAULT_ERP = 0.045          # equity risk premium (ASSUMPTION, not a fact)
DEFAULT_RF = 0.043           # risk-free fallback if ^TNX fetch fails
FALLBACK_G1 = 0.10
CAGR_CAP = (-0.10, 0.25)
CACHE_TTL = 24 * 60 * 60
MAX_WORKERS = 8
BETA_LOOKBACK = "3y"         # weekly returns window for regression beta

NVIDIA_GREEN = "#76B900"


# =============================================================================
# Data layer
# =============================================================================
def _statement_value(df, *names):
    if df is None or getattr(df, "empty", True):
        return None
    for name in names:
        if name in df.index:
            s = df.loc[name].dropna()
            if not s.empty:
                return float(s.iloc[0])
    return None


def _extract_fcf_series(cf):
    """Return FCF oldest->newest as a plain list, or []. Handles both the
    direct 'Free Cash Flow' row and the OCF+Capex reconstruction."""
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
    # yfinance columns are newest-first; reverse to chronological
    return [float(v) for v in s.iloc[::-1].tolist()]


@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def fetch_one(ticker: str) -> dict:
    """Per-ticker fetch (cached individually). No `.info` call — fast_info plus
    the statement frames cover everything except beta, which we batch separately."""
    out = {"ticker": ticker, "fcf": None, "fcf_series": [], "sbc": None,
           "shares": None, "cash": None, "debt": None, "price": None,
           "market_cap": None, "interest_expense": None, "tax_rate": 0.21,
           "analyst_g5": None, "hist_cagr": None, "missing": [], "error": None}
    try:
        t = yf.Ticker(ticker)
        cf = getattr(t, "cashflow", None)
        bs = getattr(t, "balance_sheet", None)
        inc = getattr(t, "income_stmt", None)
        fi = getattr(t, "fast_info", {}) or {}

        series = _extract_fcf_series(cf)
        out["fcf_series"] = series
        out["fcf"] = series[-1] if series else None
        out["hist_cagr"] = core.robust_cagr(series, cap=CAGR_CAP)
        out["sbc"] = _statement_value(cf, "Stock Based Compensation")

        # Standardized, CONSISTENT cash definition across every ticker:
        # cash & equivalents + short-term investments (what you'd net vs debt).
        out["cash"] = _statement_value(
            bs, "Cash Cash Equivalents And Short Term Investments",
            "Cash And Cash Equivalents And Short Term Investments",
            "Cash And Cash Equivalents")

        debt = _statement_value(bs, "Total Debt")
        if debt is None:
            ltd = _statement_value(bs, "Long Term Debt") or 0.0
            std = _statement_value(bs, "Current Debt", "Short Term Debt",
                                   "Current Debt And Capital Lease Obligation") or 0.0
            debt = (ltd + std) or None
        out["debt"] = debt

        def _fi(*keys):
            for k in keys:
                v = fi.get(k) if hasattr(fi, "get") else None
                if v:
                    return float(v)
            return None

        out["price"] = _fi("last_price", "lastPrice")
        out["market_cap"] = _fi("market_cap", "marketCap")
        out["shares"] = _fi("shares", "shares_outstanding")
        if out["shares"] is None and out["market_cap"] and out["price"]:
            out["shares"] = out["market_cap"] / out["price"]

        out["interest_expense"] = _statement_value(
            inc, "Interest Expense", "Interest Expense Non Operating")
        tax = _statement_value(inc, "Tax Provision", "Income Tax Expense")
        pretax = _statement_value(inc, "Pretax Income", "Income Before Tax")
        if tax is not None and pretax and pretax > 0:
            out["tax_rate"] = min(max(tax / pretax, 0.0), 0.40)

        # analyst 5y growth (best-effort; shape varies by yfinance version)
        try:
            ge = t.growth_estimates
            if ge is not None and not ge.empty:
                idx = next((i for i in ge.index
                            if str(i).lower().replace(" ", "")
                            in ("+5y", "5y", "next5years", "+5years")), None)
                if idx is not None:
                    row = ge.loc[idx]
                    val = float(row.dropna().iloc[0]) if hasattr(row, "dropna") else float(row)
                    out["analyst_g5"] = val / 100.0 if abs(val) > 1.5 else val
        except Exception:
            pass
    except Exception as e:
        out["error"] = str(e)

    for k in ("fcf", "shares", "cash", "debt", "price"):
        if out[k] is None:
            out["missing"].append(k)
    return out


@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def fetch_betas(tickers: tuple) -> dict:
    """One batched download -> regression beta vs SPY on weekly returns.
    Replaces N slow `.info` scrapes with a single network call."""
    try:
        data = yf.download(list(tickers) + [BENCHMARK], period=BETA_LOOKBACK,
                           interval="1wk", auto_adjust=True, progress=False)
        px = data["Close"] if "Close" in data else data
        rets = px.pct_change().dropna(how="all")
        mkt = rets[BENCHMARK]
        var_m = mkt.var()
        betas = {}
        for tk in tickers:
            if tk in rets and var_m:
                cov = rets[tk].cov(mkt)
                betas[tk] = float(cov / var_m)
            else:
                betas[tk] = None
        return betas
    except Exception:
        return {tk: None for tk in tickers}


@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def fetch_risk_free():
    try:
        v = yf.Ticker("^TNX").fast_info.get("last_price")
        if v is None:
            return None
        v = float(v)
        return v / 10.0 if v > 25 else v   # ^TNX historically quoted x10
    except Exception:
        return None


def fetch_all(tickers: tuple) -> dict:
    """Parallel per-ticker fetch + one batched beta call + risk-free."""
    stocks: dict = {}
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futs = {}
        for tk in tickers:
            fut = ex.submit(fetch_one, tk)
            add_script_run_context(fut)  # propagate streamlit ctx to worker
            futs[fut] = tk
        for fut in futs:
            tk = futs[fut]
            try:
                stocks[tk] = fut.result()
            except Exception as e:
                stocks[tk] = {"ticker": tk, "error": str(e), "missing": [],
                              "fcf_series": []}
    betas = fetch_betas(tickers)
    for tk, b in betas.items():
        if tk in stocks:
            stocks[tk]["beta"] = b
    rf = fetch_risk_free()
    return {"as_of": dt.datetime.now(), "risk_free": rf, "stocks": stocks}


# =============================================================================
# Presentation
# =============================================================================
st.set_page_config(page_title="DCF Engine", layout="wide",
                   initial_sidebar_state="expanded")

st.markdown(f"""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');
  html, body, [class*="css"] {{ font-family: 'Inter', system-ui, sans-serif; }}
  .stApp {{ background:
      radial-gradient(1200px 600px at 80% -10%, #14210a 0%, transparent 55%),
      #0a0a0a; }}
  #MainMenu, footer {{ visibility: hidden; }}
  .hero {{ padding: 8px 0 4px 0; border-bottom: 1px solid #1f1f1f; margin-bottom: 14px; }}
  .hero h1 {{ font-weight: 800; letter-spacing: -0.5px; font-size: 2.05rem; margin: 0;
              color: #f4f4f4; }}
  .hero h1 span {{ color: {NVIDIA_GREEN}; }}
  .hero p {{ color: #9a9a9a; margin: 4px 0 0 0; font-size: 0.92rem; }}
  /* expander = card */
  div[data-testid="stExpander"] {{ background: #111313; border: 1px solid #232323;
      border-radius: 14px; margin-bottom: 10px; }}
  div[data-testid="stExpander"] summary:hover {{ color: {NVIDIA_GREEN}; }}
  div[data-testid="stExpander"] summary {{ font-weight: 600; }}
  /* metrics */
  div[data-testid="stMetricValue"] {{ font-weight: 800; }}
  /* inputs */
  .stNumberInput input {{ background:#0e0e0e; border:1px solid #2a2a2a; color:#eee; }}
  .stButton button {{ background:{NVIDIA_GREEN}; color:#0a0a0a; border:none;
      font-weight:700; border-radius:8px; }}
  .stButton button:hover {{ filter: brightness(1.08); }}
  .badge {{ display:inline-block; padding:2px 9px; border-radius:999px;
      font-size:0.72rem; font-weight:700; }}
  .under {{ background:rgba(118,185,0,.15); color:{NVIDIA_GREEN}; border:1px solid {NVIDIA_GREEN}; }}
  .over  {{ background:rgba(255,77,79,.12); color:#ff6b6b; border:1px solid #ff6b6b; }}
</style>
<div class="hero">
  <h1>DCF <span>Engine</span></h1>
  <p>Two-stage discounted cash flow · assumptions auto-seeded from data, fully editable ·
     verify against the 10-K before you trade on it.</p>
</div>
""", unsafe_allow_html=True)

# --- data load -------------------------------------------------------------- #
with st.spinner("Pulling financials in parallel (cached 24h)…"):
    payload = fetch_all(tuple(TICKERS))

# --- sidebar ---------------------------------------------------------------- #
st.sidebar.header("Macro knobs")
rf = st.sidebar.number_input("Risk-free % (10y UST)",
                             value=round((payload["risk_free"] or DEFAULT_RF) * 100, 2),
                             step=0.1) / 100
erp = st.sidebar.number_input("Equity risk premium %",
                              value=DEFAULT_ERP * 100, step=0.25) / 100
term = st.sidebar.number_input("Terminal growth %",
                               value=DEFAULT_TERMINAL * 100, step=0.25) / 100

st.sidebar.header("Modeling")
fcf_method = st.sidebar.selectbox(
    "FCF base year", ["latest", "mean", "median"], index=0,
    help="Normalize the launch FCF to reduce one-off distortion.")
fcf_n = st.sidebar.slider("…over N years", 2, 5, 3, disabled=(fcf_method == "latest"))
burden_sbc = st.sidebar.toggle(
    "Burden FCF with stock-based comp", value=False,
    help="SaaS FCF is flattered by large SBC add-backs; subtract it for a "
         "stricter owner-earnings base.")
st.sidebar.caption("Edits to any stock recompute only that card (fragments).")
c1, c2 = st.sidebar.columns(2)
if c1.button("↻ Re-seed all"):
    for k in list(st.session_state.keys()):
        if k.rsplit("_", 1)[-1] in ("g1", "g2", "gt", "r", "fcf"):
            del st.session_state[k]
    st.rerun()
if c2.button("🔄 Refresh data"):
    fetch_one.clear(); fetch_betas.clear(); fetch_risk_free.clear()
    st.rerun()

st.caption(f"Data as of **{payload['as_of']:%Y-%m-%d %H:%M}** · "
           f"risk-free {rf*100:.2f}% · ERP {erp*100:.2f}% · "
           f"{len([s for s in payload['stocks'].values() if not s.get('error')])}"
           f"/{len(TICKERS)} loaded")


def _base_fcf(s):
    series = s.get("fcf_series") or ([] if s.get("fcf") is None else [s["fcf"]])
    base = core.normalize_fcf(series, fcf_method, fcf_n) if series else s.get("fcf")
    if burden_sbc and base is not None and s.get("sbc"):
        base = base - abs(s["sbc"])
    return base


def _seeded(s):
    return core.derive_assumptions(s, rf, erp, term, CAGR_CAP, FALLBACK_G1)


# --- summary dashboard ------------------------------------------------------ #
def build_summary():
    recs = []
    for tk in TICKERS:
        s = payload["stocks"][tk]
        if s.get("error"):
            recs.append({"Ticker": tk, "Fair": None, "Price": None,
                         "MoS %": None, "g1 src": "—", "Flags": "fetch failed"})
            continue
        auto, src = _seeded(s)
        if auto["r"] is None:
            recs.append({"Ticker": tk, "Fair": None, "Price": s.get("price"),
                         "MoS %": None, "g1 src": src, "Flags": "no WACC"})
            continue
        res = core.two_stage_dcf(_base_fcf(s), auto["g1"], auto["g2"], auto["gt"],
                                 auto["r"], s.get("cash"), s.get("debt"), s.get("shares"))
        dq = core.data_quality_flags(s)
        if res.error:
            recs.append({"Ticker": tk, "Fair": None, "Price": s.get("price"),
                         "MoS %": None, "g1 src": src, "Flags": res.error})
            continue
        price = s.get("price")
        mos = (res.fair - price) / price * 100 if price else None
        recs.append({"Ticker": tk, "Fair": res.fair, "Price": price, "MoS %": mos,
                     "g1 src": src, "Flags": "; ".join(res.flags + dq) or "—"})
    return pd.DataFrame(recs)


summ = build_summary()
st.subheader("Coverage")
styled = (summ.style
          .format({"Fair": "${:,.2f}", "Price": "${:,.2f}", "MoS %": "{:+.1f}%"},
                  na_rep="—")
          .background_gradient(cmap="RdYlGn", subset=["MoS %"], vmin=-60, vmax=120))
st.dataframe(styled, hide_index=True, use_container_width=True,
             column_config={"Flags": st.column_config.TextColumn(width="large")})

st.subheader("Per-stock detail")


# --- per-ticker fragments --------------------------------------------------- #
@st.fragment
def render_ticker(tk: str):
    s = payload["stocks"][tk]
    auto, src = _seeded(s)
    base_fcf = _base_fcf(s)

    for fld in ("g1", "g2", "gt", "r"):
        key = f"{tk}_{fld}"
        if key not in st.session_state:
            v = auto[fld]
            st.session_state[key] = round((v if v is not None else
                                           (FALLBACK_G1 if fld == "g1" else 0.09)) * 100, 2)
    fkey = f"{tk}_fcf"
    if fkey not in st.session_state:
        st.session_state[fkey] = round(base_fcf / 1e6, 1) if base_fcf else 0.0

    head = f"**{tk}**"
    if s.get("error"):
        head += " · ⚠️ fetch failed"
    elif s.get("missing"):
        head += f" · ⚠️ missing: {', '.join(s['missing'])}"
    head += f"  ·  g1 src: {src}  ·  WACC: {'auto' if auto['r'] else 'fallback'}"

    with st.expander(head):
        if s.get("error"):
            st.error(s["error"]); return

        def m(x): return "—" if x is None else f"${x/1e6:,.0f}M"
        sh = s.get("shares"); be = s.get("beta"); pr = s.get("price")
        L, R = st.columns(2)
        with L:
            st.markdown("**Auto-filled data** (FCF editable)")
            fcf_m = st.number_input("Free Cash Flow ($M)", key=fkey, step=100.0)
            st.write(f"Shares: {'—' if sh is None else f'{sh/1e6:,.0f}M'}"
                     f"  ·  Beta: {'—' if be is None else f'{be:.2f}'}")
            st.write(f"Mkt cap: {m(s.get('market_cap'))}  ·  Cash: {m(s.get('cash'))}"
                     f"  ·  Debt: {m(s.get('debt'))}")
            st.write(f"Current price: {'—' if pr is None else f'${pr:,.2f}'}")
        with R:
            st.markdown("**Assumptions** (auto-seeded)")
            g1 = st.number_input("Yr 1–5 growth %", key=f"{tk}_g1", step=0.5)
            g2 = st.number_input("Yr 6–10 growth %", key=f"{tk}_g2", step=0.5)
            gt = st.number_input("Terminal growth %", key=f"{tk}_gt", step=0.25)
            r = st.number_input("Discount rate (WACC) %", key=f"{tk}_r", step=0.25)

        res = core.two_stage_dcf(fcf_m * 1e6, g1/100, g2/100, gt/100, r/100,
                                 s.get("cash"), s.get("debt"), s.get("shares"))
        if res.error:
            st.warning(res.error); return

        cols = st.columns(3)
        cols[0].metric("Fair value / share", f"${res.fair:,.2f}")
        cols[1].metric("Current price", "—" if pr is None else f"${pr:,.2f}")
        if pr:
            mos = (res.fair - pr) / pr * 100
            badge = "under" if mos > 0 else "over"
            cols[2].metric("Margin of safety", f"{mos:+.1f}%")
            cols[2].markdown(
                f"<span class='badge {badge}'>"
                f"{'Undervalued' if mos > 0 else 'Overvalued'}</span>",
                unsafe_allow_html=True)

        for f in res.flags + core.data_quality_flags(s):
            st.caption(f"⚠️ {f}")

        with st.expander("Projection detail"):
            proj = pd.DataFrame(res.rows)
            proj["FCF"] = (proj["FCF"] / 1e6).round(0)
            proj["PV of FCF"] = (proj["PV of FCF"] / 1e6).round(0)
            st.dataframe(proj, hide_index=True, use_container_width=True)
            st.caption(f"PV(FCF) {m(res.pv_fcf_sum)} · PV(TV) {m(res.pv_tv)} "
                       f"({res.tv_fraction:.0%} of EV) · EV {m(res.ev)} · "
                       f"Equity {m(res.equity)}")


for tk in TICKERS:
    render_ticker(tk)

st.divider()
st.caption("WACC = CAPM cost of equity · E/V + after-tax cost of debt · D/V. "
           "Beta = regression vs SPY on 3y weekly returns. g1 = analyst 5y "
           "estimate (EPS proxy) or capped log-linear FCF CAGR; g2 fades halfway "
           "to terminal. Result is hypersensitive to WACC and terminal growth — "
           "one scenario, not a price target.")
