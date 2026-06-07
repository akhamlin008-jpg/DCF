"""
Two-stage DCF calculator for ~20 large-cap stocks.

Architecture (matches what was specified):
- The SLOW part -- pulling financials from yfinance for all tickers -- runs once
  and is cached for 24 hours (st.cache_data ttl=86400). The first visitor after
  the cache expires triggers the refresh; everyone else reuses it.
- The DCF MATH is instant and recomputes live whenever you change an assumption,
  exactly like dcfcalc.com.

DCF model (identical formula to dcfcalc.com):
  Two-stage, 10-year explicit forecast.
    Years 1-5  grow at g1 (high-growth stage)
    Years 6-10 grow at g2 (transition stage)
  Terminal value (Gordon Growth, at end of year 10):
    TV = FCF_10 * (1 + g_term) / (r - g_term)
  Everything discounted at r, then:
    Enterprise Value = sum(PV of FCFs) + PV(TV)
    Equity Value     = EV + Cash - Debt
    Fair Value/share = Equity Value / Shares Outstanding

IMPORTANT: This computes a CORRECT two-stage DCF *given your inputs*. It is not a
statement of "true" intrinsic value -- the output is driven by the growth and
discount assumptions YOU type, and by yfinance data quality. Verify figures
against a primary source (the company's 10-K / SEC filings) before relying on them.
"""

import datetime as dt

import pandas as pd
import streamlit as st
import yfinance as yf

# --- Configuration -----------------------------------------------------------

# Approximate top-20 S&P 500 by market cap (early 2026). EDIT THIS LIST FREELY.
# Note: JPM, V, MA, BRK-B are financials -- DCF on FCF is a poor fit for them
# (see dcfcalc.com's own FAQ). Left in only to fill the list; replace as you like.
TICKERS = [
    "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN",
    "META", "AVGO", "TSLA", "LLY", "WMT",
    "JPM", "V", "ORCL", "XOM", "MA",
    "COST", "HD", "PG", "JNJ", "NFLX",
]

# Default assumptions applied to every stock until you override per-stock.
DEFAULTS = {
    "g1": 10.0,    # years 1-5 growth, %
    "g2": 6.0,     # years 6-10 growth, %
    "g_term": 2.5, # terminal growth, %
    "r": 9.0,      # discount rate (WACC), %
}

CACHE_TTL_SECONDS = 24 * 60 * 60  # 24 hours

# --- Data layer (cached for 24h) ---------------------------------------------


def _safe_row(df, *candidate_names):
    """Return the most-recent value of the first matching row in a yfinance
    statement DataFrame, or None if none of the candidate row names exist."""
    if df is None or df.empty:
        return None
    for name in candidate_names:
        if name in df.index:
            series = df.loc[name].dropna()
            if not series.empty:
                return float(series.iloc[0])  # most recent period (first column)
    return None


def _fetch_one(ticker: str) -> dict:
    """Pull the raw inputs for one ticker. Returns a dict; any value that could
    not be found is None and is flagged in 'missing'. Values are in raw dollars
    (not millions) except shares (raw count)."""
    out = {
        "ticker": ticker, "fcf": None, "shares": None,
        "cash": None, "debt": None, "price": None,
        "missing": [], "error": None,
    }
    try:
        t = yf.Ticker(ticker)
        cf = getattr(t, "cashflow", None)        # annual cash-flow statement
        bs = getattr(t, "balance_sheet", None)   # annual balance sheet

        # Free Cash Flow: prefer the explicit row; else Operating CF + CapEx
        fcf = _safe_row(cf, "Free Cash Flow")
        if fcf is None:
            ocf = _safe_row(cf, "Operating Cash Flow",
                            "Total Cash From Operating Activities")
            capex = _safe_row(cf, "Capital Expenditure", "Capital Expenditures")
            if ocf is not None and capex is not None:
                fcf = ocf + capex  # capex is reported negative
        out["fcf"] = fcf

        # Cash & equivalents
        out["cash"] = _safe_row(
            bs, "Cash And Cash Equivalents",
            "Cash Cash Equivalents And Short Term Investments",
            "Cash And Cash Equivalents And Short Term Investments",
        )

        # Total debt
        debt = _safe_row(bs, "Total Debt")
        if debt is None:
            ltd = _safe_row(bs, "Long Term Debt") or 0.0
            std = _safe_row(bs, "Current Debt", "Short Term Debt",
                            "Current Debt And Capital Lease Obligation") or 0.0
            debt = (ltd + std) or None
        out["debt"] = debt

        # Shares outstanding + current price (fast_info is the most reliable path)
        fi = getattr(t, "fast_info", {}) or {}
        out["shares"] = fi.get("shares") or (t.info.get("sharesOutstanding")
                                             if hasattr(t, "info") else None)
        out["price"] = fi.get("last_price") or (t.info.get("currentPrice")
                                                if hasattr(t, "info") else None)

    except Exception as e:  # network / parsing failure for this ticker
        out["error"] = str(e)

    for k in ("fcf", "shares", "cash", "debt", "price"):
        if out[k] is None:
            out["missing"].append(k)
    return out


@st.cache_data(ttl=CACHE_TTL_SECONDS, show_spinner=False)
def fetch_all(tickers: tuple) -> dict:
    """Cached for 24h. Keyed on the ticker tuple, so editing TICKERS busts it."""
    data = {tk: _fetch_one(tk) for tk in tickers}
    return {"as_of": dt.datetime.now(), "stocks": data}


# --- DCF math ----------------------------------------------------------------


def two_stage_dcf(fcf0, g1, g2, g_term, r, cash, debt, shares):
    """All growth/discount args are decimals (0.10 == 10%). fcf0/cash/debt in the
    same currency unit; shares in matching unit. Returns a result dict or an error."""
    if r <= g_term:
        return {"error": "Discount rate must be greater than terminal growth "
                         "(otherwise the terminal value is infinite/negative)."}
    if not shares or shares <= 0:
        return {"error": "Shares outstanding unavailable -- cannot compute per share."}
    if fcf0 is None:
        return {"error": "Free cash flow unavailable."}

    rows, pv_sum, fcf = [], 0.0, fcf0
    for year in range(1, 11):
        g = g1 if year <= 5 else g2
        fcf = fcf * (1 + g)
        pv = fcf / (1 + r) ** year
        pv_sum += pv
        rows.append({"Year": year, "FCF": fcf, "PV of FCF": pv})

    tv = fcf * (1 + g_term) / (r - g_term)   # fcf is now FCF_10
    pv_tv = tv / (1 + r) ** 10
    ev = pv_sum + pv_tv
    equity = ev + (cash or 0.0) - (debt or 0.0)
    fair = equity / shares
    return {
        "fair": fair, "ev": ev, "equity": equity,
        "pv_fcf_sum": pv_sum, "pv_tv": pv_tv, "tv": tv,
        "rows": rows, "error": None,
    }


# --- UI ----------------------------------------------------------------------

st.set_page_config(page_title="DCF — Top 20", layout="wide")
st.title("Two-Stage DCF — Top 20 Stocks")
st.caption(
    "Data via yfinance (unofficial; verify against 10-K/SEC filings). "
    "This is a correct DCF *given your assumptions* — not a 'true' intrinsic value."
)

with st.spinner("Pulling financials (cached for 24h)…"):
    payload = fetch_all(tuple(TICKERS))

as_of = payload["as_of"].strftime("%Y-%m-%d %H:%M")
c1, c2 = st.columns([3, 1])
c1.info(f"Data as of **{as_of}** — refreshes automatically after 24h.")
if c2.button("🔄 Force refresh now"):
    fetch_all.clear()
    st.rerun()

# Sidebar: global default assumptions + 'apply to all'
st.sidebar.header("Global assumptions")
st.sidebar.caption("Type your assumptions. These seed every stock; you can "
                   "override each one below.")
gd1 = st.sidebar.number_input("Years 1–5 growth %", value=DEFAULTS["g1"], step=0.5)
gd2 = st.sidebar.number_input("Years 6–10 growth %", value=DEFAULTS["g2"], step=0.5)
gdt = st.sidebar.number_input("Terminal growth %", value=DEFAULTS["g_term"], step=0.25)
gdr = st.sidebar.number_input("Discount rate (WACC) %", value=DEFAULTS["r"], step=0.5)
if st.sidebar.button("Apply global to ALL stocks"):
    for tk in TICKERS:
        st.session_state[f"{tk}_g1"] = gd1
        st.session_state[f"{tk}_g2"] = gd2
        st.session_state[f"{tk}_gt"] = gdt
        st.session_state[f"{tk}_r"] = gdr
    st.rerun()


def fmt_m(x):
    return "—" if x is None else f"${x/1e6:,.0f}M"


for tk in TICKERS:
    d = payload["stocks"][tk]

    # seed per-stock session state from globals on first render
    st.session_state.setdefault(f"{tk}_g1", gd1)
    st.session_state.setdefault(f"{tk}_g2", gd2)
    st.session_state.setdefault(f"{tk}_gt", gdt)
    st.session_state.setdefault(f"{tk}_r", gdr)
    fcf_default_m = round(d["fcf"] / 1e6, 1) if d["fcf"] is not None else 0.0
    st.session_state.setdefault(f"{tk}_fcf", fcf_default_m)

    header = f"**{tk}**"
    if d["error"]:
        header += " — ⚠️ fetch failed"
    elif d["missing"]:
        header += f" — ⚠️ missing: {', '.join(d['missing'])}"

    with st.expander(header):
        if d["error"]:
            st.error(f"yfinance error: {d['error']}")
            continue

        left, right = st.columns(2)
        shares_txt = "—" if d["shares"] is None else f"{d['shares']/1e6:,.0f}M"
        price_txt = "—" if d["price"] is None else f"${d['price']:,.2f}"
        with left:
            st.markdown("**Auto-filled data** (editable FCF)")
            fcf_m = st.number_input("Free Cash Flow ($M)", key=f"{tk}_fcf", step=100.0)
            st.write(f"Shares: {shares_txt}")
            st.write(f"Cash: {fmt_m(d['cash'])}  •  Debt: {fmt_m(d['debt'])}")
            st.write(f"Current price: {price_txt}")
        with right:
            st.markdown("**Your assumptions**")
            g1 = st.number_input("Yr 1–5 growth %", key=f"{tk}_g1", step=0.5)
            g2 = st.number_input("Yr 6–10 growth %", key=f"{tk}_g2", step=0.5)
            gt = st.number_input("Terminal growth %", key=f"{tk}_gt", step=0.25)
            r = st.number_input("Discount rate %", key=f"{tk}_r", step=0.5)

        res = two_stage_dcf(
            fcf0=fcf_m * 1e6, g1=g1/100, g2=g2/100, g_term=gt/100, r=r/100,
            cash=d["cash"], debt=d["debt"], shares=d["shares"],
        )
        if res["error"]:
            st.warning(res["error"])
            continue

        fair = res["fair"]
        price = d["price"]
        m1, m2, m3 = st.columns(3)
        m1.metric("Fair value / share", f"${fair:,.2f}")
        m2.metric("Current price", "—" if price is None else f"${price:,.2f}")
        if price:
            mos = (fair - price) / price * 100
            m3.metric("Margin of safety", f"{mos:+.1f}%",
                      "Undervalued" if mos > 0 else "Overvalued")

        with st.expander("Show projection detail"):
            proj = pd.DataFrame(res["rows"])
            proj["FCF"] = (proj["FCF"] / 1e6).round(0)
            proj["PV of FCF"] = (proj["PV of FCF"] / 1e6).round(0)
            st.dataframe(proj, hide_index=True, use_container_width=True)
            st.caption(
                f"PV of FCFs: {fmt_m(res['pv_fcf_sum'])} • "
                f"PV of terminal value: {fmt_m(res['pv_tv'])} • "
                f"Enterprise value: {fmt_m(res['ev'])} • "
                f"Equity value: {fmt_m(res['equity'])}"
            )

st.divider()
st.caption(
    "Reminder: terminal value usually dominates and is highly sensitive to the "
    "terminal-growth and discount-rate inputs. Treat the output as one scenario, "
    "not a fact. yfinance field names change between versions — if data shows as "
    "missing, the row name in _safe_row() may need updating."
)
