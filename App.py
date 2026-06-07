"""
Two-stage DCF for ~20 large-cap stocks, with AUTOFILLED assumptions.

What changed in this version: the four "Your assumptions" fields are now
pre-populated from data instead of typed from scratch -- but every one stays
editable, because autofilled != correct (see notes at bottom).

Where each assumption comes from (this is the important part):
  * Yr 1-5 growth  -> analyst consensus 5y growth (yfinance), fallback = capped
                      historical FCF CAGR, fallback = your sidebar default.
  * Yr 6-10 growth -> DERIVED, not fetched: fades halfway from g1 to terminal.
  * Terminal       -> a macro convention you set in the sidebar (~2.5%).
  * Discount rate  -> WACC via CAPM, computed live from beta + risk-free + ERP
                      + capital structure, so your ERP/risk-free knobs stay live.

Data via yfinance (unofficial). Verify against 10-K / SEC filings before relying
on any single stock's number. Could not be tested against live Yahoo data in the
build sandbox; yfinance renames fields between versions, so a field showing as
"missing" usually means the row name in _safe_row()/_fetch_one() needs updating.
"""

import datetime as dt

import pandas as pd
import streamlit as st
import yfinance as yf

# --- Configuration -----------------------------------------------------------

TICKERS = [
    "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN",
    "META", "AVGO", "TSLA", "LLY", "WMT",
    "JPM", "V", "ORCL", "XOM", "MA",
    "COST", "HD", "PG", "JNJ", "NFLX",
]

# Fallbacks used ONLY when data can't be fetched.
FALLBACK = {"g1": 10.0, "g2": 6.0}
DEFAULT_TERMINAL = 2.5   # %
DEFAULT_ERP = 4.5        # % equity risk premium (an ASSUMPTION, not a fact)
DEFAULT_RF = 4.3         # % risk-free, used if ^TNX fetch fails
CAGR_CAP = (-10.0, 25.0) # clamp band (%) for the historical-CAGR fallback
CACHE_TTL_SECONDS = 24 * 60 * 60

# --- Data layer (cached for 24h) ---------------------------------------------


def _safe_row(df, *names):
    if df is None or getattr(df, "empty", True):
        return None
    for name in names:
        if name in df.index:
            s = df.loc[name].dropna()
            if not s.empty:
                return float(s.iloc[0])
    return None


def _hist_fcf_cagr(cf):
    """Annualized FCF growth across the available years in the cash-flow
    statement. Returns a decimal (0.12 == 12%) or None."""
    if cf is None or cf.empty:
        return None
    fcf_series = None
    if "Free Cash Flow" in cf.index:
        fcf_series = cf.loc["Free Cash Flow"].dropna()
    else:
        ocf_name = next((n for n in ("Operating Cash Flow",
                                     "Total Cash From Operating Activities")
                         if n in cf.index), None)
        cx_name = next((n for n in ("Capital Expenditure", "Capital Expenditures")
                        if n in cf.index), None)
        if ocf_name and cx_name:
            fcf_series = (cf.loc[ocf_name] + cf.loc[cx_name]).dropna()
    if fcf_series is None or len(fcf_series) < 2:
        return None
    # columns are most-recent-first; oldest is last
    newest, oldest = float(fcf_series.iloc[0]), float(fcf_series.iloc[-1])
    yrs = len(fcf_series) - 1
    if oldest <= 0 or newest <= 0 or yrs < 1:
        return None
    return (newest / oldest) ** (1 / yrs) - 1


def _analyst_5y_growth(t):
    """Best-effort pull of the analyst 'next 5 years per annum' growth estimate.
    yfinance exposes this via .growth_estimates in recent versions, but the
    shape/availability varies -- hence the broad try/except. Returns decimal or None."""
    try:
        ge = t.growth_estimates
        if ge is None or ge.empty:
            return None
        # find a row keyed to the +5y horizon
        idx = next((i for i in ge.index if str(i).lower().replace(" ", "")
                    in ("+5y", "5y", "next5years", "+5years")), None)
        if idx is None:
            return None
        row = ge.loc[idx]
        val = float(row.dropna().iloc[0]) if hasattr(row, "dropna") else float(row)
        if abs(val) > 1.5:   # looks like a percent (e.g. 12.0) not a decimal
            val /= 100.0
        return val
    except Exception:
        return None


def _fetch_one(ticker: str) -> dict:
    out = {"ticker": ticker, "fcf": None, "shares": None, "cash": None,
           "debt": None, "price": None, "beta": None, "market_cap": None,
           "interest_expense": None, "tax_rate": None, "analyst_g5": None,
           "hist_cagr": None, "missing": [], "error": None}
    try:
        t = yf.Ticker(ticker)
        cf = getattr(t, "cashflow", None)
        bs = getattr(t, "balance_sheet", None)
        inc = getattr(t, "income_stmt", None)
        info = {}
        try:
            info = t.info or {}
        except Exception:
            info = {}
        fi = getattr(t, "fast_info", {}) or {}

        # FCF
        fcf = _safe_row(cf, "Free Cash Flow")
        if fcf is None:
            ocf = _safe_row(cf, "Operating Cash Flow",
                            "Total Cash From Operating Activities")
            capex = _safe_row(cf, "Capital Expenditure", "Capital Expenditures")
            if ocf is not None and capex is not None:
                fcf = ocf + capex
        out["fcf"] = fcf

        out["cash"] = _safe_row(
            bs, "Cash And Cash Equivalents",
            "Cash Cash Equivalents And Short Term Investments",
            "Cash And Cash Equivalents And Short Term Investments")

        debt = _safe_row(bs, "Total Debt")
        if debt is None:
            ltd = _safe_row(bs, "Long Term Debt") or 0.0
            std = _safe_row(bs, "Current Debt", "Short Term Debt",
                            "Current Debt And Capital Lease Obligation") or 0.0
            debt = (ltd + std) or None
        out["debt"] = debt

        out["shares"] = fi.get("shares") or info.get("sharesOutstanding")
        out["price"] = fi.get("last_price") or info.get("currentPrice")
        out["market_cap"] = fi.get("market_cap") or info.get("marketCap")
        out["beta"] = info.get("beta")

        out["interest_expense"] = _safe_row(
            inc, "Interest Expense", "Interest Expense Non Operating")

        tax = _safe_row(inc, "Tax Provision", "Income Tax Expense")
        pretax = _safe_row(inc, "Pretax Income", "Income Before Tax")
        if tax is not None and pretax and pretax > 0:
            out["tax_rate"] = min(max(tax / pretax, 0.0), 0.40)
        else:
            out["tax_rate"] = 0.21  # US statutory fallback

        out["analyst_g5"] = _analyst_5y_growth(t)
        out["hist_cagr"] = _hist_fcf_cagr(cf)

    except Exception as e:
        out["error"] = str(e)

    for k in ("fcf", "shares", "cash", "debt", "price"):
        if out[k] is None:
            out["missing"].append(k)
    return out


def _fetch_risk_free():
    """10y Treasury yield via ^TNX, returned as a percent. ^TNX has historically
    been quoted as yield x10 on Yahoo, so normalize defensively."""
    try:
        v = yf.Ticker("^TNX").fast_info.get("last_price")
        if v is None:
            return None
        v = float(v)
        if v > 25:           # implausible as a raw yield -> it's the x10 convention
            v /= 10.0
        return v
    except Exception:
        return None


@st.cache_data(ttl=CACHE_TTL_SECONDS, show_spinner=False)
def fetch_all(tickers: tuple) -> dict:
    return {"as_of": dt.datetime.now(),
            "risk_free": _fetch_risk_free(),
            "stocks": {tk: _fetch_one(tk) for tk in tickers}}


# --- Finance math ------------------------------------------------------------


def compute_wacc(beta, rf, erp, market_cap, debt, interest_expense, tax_rate):
    """All rate args are decimals. Returns WACC as a decimal, or None."""
    if beta is None or not market_cap or market_cap <= 0:
        return None
    cost_equity = rf + beta * erp
    E = market_cap
    D = debt or 0.0
    V = E + D
    if V <= 0:
        return None
    if D > 0 and interest_expense:
        pretax_cd = abs(interest_expense) / D
        after_tax_cd = pretax_cd * (1 - (tax_rate or 0.21))
    else:
        after_tax_cd = 0.0
    return cost_equity * (E / V) + after_tax_cd * (D / V)


def auto_assumptions(s, rf_pct, erp_pct, terminal_pct):
    """Return the four assumptions as PERCENTS for the UI. Each falls back
    gracefully when its data is missing."""
    # Yr 1-5 growth
    if s["analyst_g5"] is not None:
        g1 = s["analyst_g5"] * 100
    elif s["hist_cagr"] is not None:
        g1 = min(max(s["hist_cagr"] * 100, CAGR_CAP[0]), CAGR_CAP[1])
    else:
        g1 = FALLBACK["g1"]
    gt = terminal_pct
    g2 = round((g1 + gt) / 2, 2)          # fade halfway toward terminal
    wacc = compute_wacc(s["beta"], rf_pct / 100, erp_pct / 100,
                        s["market_cap"], s["debt"], s["interest_expense"],
                        s["tax_rate"])
    r = round(wacc * 100, 2) if wacc is not None else None
    return {"g1": round(g1, 2), "g2": g2, "gt": round(gt, 2), "r": r}


def two_stage_dcf(fcf0, g1, g2, g_term, r, cash, debt, shares):
    if r <= g_term:
        return {"error": "Discount rate must exceed terminal growth."}
    if not shares or shares <= 0:
        return {"error": "Shares outstanding unavailable."}
    if fcf0 is None:
        return {"error": "Free cash flow unavailable."}
    rows, pv_sum, fcf = [], 0.0, fcf0
    for year in range(1, 11):
        g = g1 if year <= 5 else g2
        fcf *= (1 + g)
        pv = fcf / (1 + r) ** year
        pv_sum += pv
        rows.append({"Year": year, "FCF": fcf, "PV of FCF": pv})
    tv = fcf * (1 + g_term) / (r - g_term)
    pv_tv = tv / (1 + r) ** 10
    ev = pv_sum + pv_tv
    equity = ev + (cash or 0.0) - (debt or 0.0)
    return {"fair": equity / shares, "ev": ev, "equity": equity,
            "pv_fcf_sum": pv_sum, "pv_tv": pv_tv, "rows": rows, "error": None}


# --- UI ----------------------------------------------------------------------

st.set_page_config(page_title="DCF — Top 20", layout="wide")
st.title("Two-Stage DCF — Top 20 (autofilled, editable)")
st.caption("Assumptions are auto-seeded from data, then yours to override. "
           "Autofilled is a defensible starting point, not 'correct'.")

with st.spinner("Pulling financials (cached for 24h)…"):
    payload = fetch_all(tuple(TICKERS))

as_of = payload["as_of"].strftime("%Y-%m-%d %H:%M")
top = st.columns([3, 1])
top[0].info(f"Data as of **{as_of}** — auto-refreshes after 24h.")
if top[1].button("🔄 Force refresh"):
    fetch_all.clear()
    st.rerun()

st.sidebar.header("Macro knobs (feed WACC)")
rf_pct = st.sidebar.number_input(
    "Risk-free rate % (10y Treasury)",
    value=float(payload["risk_free"] or DEFAULT_RF), step=0.1)
erp_pct = st.sidebar.number_input("Equity risk premium %", value=DEFAULT_ERP, step=0.25)
term_pct = st.sidebar.number_input("Terminal growth %", value=DEFAULT_TERMINAL, step=0.25)
st.sidebar.caption("Changing these updates the WACC autofill — click the button "
                   "below to push fresh values into all 20 stocks.")
do_autofill = st.sidebar.button("↻ Autofill all 20 from data")


def fmt_m(x):
    return "—" if x is None else f"${x/1e6:,.0f}M"


for tk in TICKERS:
    s = payload["stocks"][tk]
    auto = auto_assumptions(s, rf_pct, erp_pct, term_pct)

    # seed once; the button force-overwrites
    for fld in ("g1", "g2", "gt", "r"):
        key = f"{tk}_{fld}"
        default = auto[fld] if auto[fld] is not None else (
            FALLBACK.get(fld, 9.0) if fld != "gt" else term_pct)
        if do_autofill and auto[fld] is not None:
            st.session_state[key] = auto[fld]
        else:
            st.session_state.setdefault(key, default)
    st.session_state.setdefault(
        f"{tk}_fcf", round(s["fcf"] / 1e6, 1) if s["fcf"] is not None else 0.0)

    header = f"**{tk}**"
    if s["error"]:
        header += " — ⚠️ fetch failed"
    elif s["missing"]:
        header += f" — ⚠️ missing: {', '.join(s['missing'])}"
    src = ("analyst" if s["analyst_g5"] is not None
           else "hist-CAGR" if s["hist_cagr"] is not None else "fallback")
    header += f"  ·  g1 src: {src}  ·  WACC: {'auto' if auto['r'] else 'fallback'}"

    with st.expander(header):
        if s["error"]:
            st.error(f"yfinance error: {s['error']}")
            continue

        shares_txt = "—" if s["shares"] is None else f"{s['shares']/1e6:,.0f}M"
        price_txt = "—" if s["price"] is None else f"${s['price']:,.2f}"
        beta_txt = "—" if s["beta"] is None else f"{s['beta']:.2f}"
        left, right = st.columns(2)
        with left:
            st.markdown("**Auto-filled data** (FCF editable)")
            fcf_m = st.number_input("Free Cash Flow ($M)", key=f"{tk}_fcf", step=100.0)
            st.write(f"Shares: {shares_txt}  ·  Beta: {beta_txt}")
            st.write(f"Mkt cap: {fmt_m(s['market_cap'])}")
            st.write(f"Cash: {fmt_m(s['cash'])}  ·  Debt: {fmt_m(s['debt'])}")
            st.write(f"Current price: {price_txt}")
        with right:
            st.markdown("**Your assumptions** (auto-seeded)")
            g1 = st.number_input("Yr 1–5 growth %", key=f"{tk}_g1", step=0.5)
            g2 = st.number_input("Yr 6–10 growth %", key=f"{tk}_g2", step=0.5)
            gt = st.number_input("Terminal growth %", key=f"{tk}_gt", step=0.25)
            r = st.number_input("Discount rate (WACC) %", key=f"{tk}_r", step=0.25)

        res = two_stage_dcf(fcf_m * 1e6, g1/100, g2/100, gt/100, r/100,
                            s["cash"], s["debt"], s["shares"])
        if res["error"]:
            st.warning(res["error"])
            continue

        fair, price = res["fair"], s["price"]
        m = st.columns(3)
        m[0].metric("Fair value / share", f"${fair:,.2f}")
        m[1].metric("Current price", "—" if price is None else f"${price:,.2f}")
        if price:
            mos = (fair - price) / price * 100
            m[2].metric("Margin of safety", f"{mos:+.1f}%",
                        "Undervalued" if mos > 0 else "Overvalued")

        with st.expander("Projection detail"):
            proj = pd.DataFrame(res["rows"])
            proj["FCF"] = (proj["FCF"] / 1e6).round(0)
            proj["PV of FCF"] = (proj["PV of FCF"] / 1e6).round(0)
            st.dataframe(proj, hide_index=True, use_container_width=True)
            st.caption(f"PV of FCFs {fmt_m(res['pv_fcf_sum'])} · "
                       f"PV of TV {fmt_m(res['pv_tv'])} · "
                       f"EV {fmt_m(res['ev'])} · Equity {fmt_m(res['equity'])}")

st.divider()
st.caption("WACC = CAPM cost of equity × E/(D+E) + after-tax cost of debt × D/(D+E). "
           "g1 = analyst 5y estimate (EPS-based proxy) or capped historical FCF CAGR. "
           "g2 fades halfway to terminal. The result is hypersensitive to WACC and "
           "terminal growth — treat it as one scenario and verify FCF/debt/shares "
           "against the latest 10-K.")
