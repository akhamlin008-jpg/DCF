"""
dcf_core.py — pure finance + data-shaping logic for the DCF app.

Deliberately imports NOTHING from streamlit or yfinance, so it is:
  * fast to import,
  * unit-testable offline (no network), and
  * reusable from a batch/CLI scaler later, not just the Streamlit UI.

All rates are decimals (0.0828 == 8.28%) unless a name ends in `_pct`.
Money is in native units (dollars), not millions, until the UI formats it.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field


# --------------------------------------------------------------------------- #
# Growth estimation
# --------------------------------------------------------------------------- #
def robust_cagr(values_oldest_first, cap=(-0.10, 0.25)):
    """Log-linear (least-squares) growth rate across *all* available years,
    not just the two endpoints. Endpoint-to-endpoint CAGR is hostage to a
    single noisy base or final year; a regression through every point is far
    steadier — which matters when you're stamping out hundreds of tickers
    unattended.

    `values_oldest_first`: chronological list/seq of FCF (oldest -> newest).
    Returns a decimal growth rate clamped to `cap`, or None if not derivable
    (need >=2 strictly-positive points).
    """
    vals = [float(v) for v in values_oldest_first if v is not None]
    if len(vals) < 2 or any(v <= 0 for v in vals):
        return None
    n = len(vals)
    xs = list(range(n))
    ys = [math.log(v) for v in vals]
    mx = sum(xs) / n
    my = sum(ys) / n
    denom = sum((x - mx) ** 2 for x in xs)
    if denom == 0:
        return None
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / denom
    g = math.exp(slope) - 1.0
    lo, hi = cap
    return max(lo, min(hi, g))


def normalize_fcf(values_oldest_first, method="latest", n=3):
    """Pick the FCF base year used to launch the projection.

    A single year's FCF can be distorted by a one-off working-capital swing,
    so for accuracy you often want a normalized base.
      * 'latest'  -> most recent year (default; matches the old behaviour)
      * 'mean'    -> mean of the last `n` years
      * 'median'  -> median of the last `n` years (robust to one outlier)
    Returns a float or None.
    """
    vals = [float(v) for v in values_oldest_first if v is not None]
    if not vals:
        return None
    if method == "latest":
        return vals[-1]
    tail = vals[-n:] if n else vals
    if not tail:
        return None
    if method == "mean":
        return sum(tail) / len(tail)
    if method == "median":
        s = sorted(tail)
        m = len(s) // 2
        return s[m] if len(s) % 2 else (s[m - 1] + s[m]) / 2
    return vals[-1]


# --------------------------------------------------------------------------- #
# Discount rate
# --------------------------------------------------------------------------- #
def compute_wacc(beta, rf, erp, market_cap, debt,
                 interest_expense=None, tax_rate=0.21):
    """CAPM cost of equity, capital-structure-weighted with an after-tax cost
    of debt. Returns WACC (decimal) or None when inputs are insufficient.
    """
    if beta is None or not market_cap or market_cap <= 0:
        return None
    cost_equity = rf + beta * erp
    E = float(market_cap)
    D = float(debt or 0.0)
    V = E + D
    if V <= 0:
        return None
    after_tax_cd = 0.0
    if D > 0 and interest_expense:
        pretax_cd = abs(interest_expense) / D
        after_tax_cd = pretax_cd * (1 - (tax_rate if tax_rate is not None else 0.21))
    return cost_equity * (E / V) + after_tax_cd * (D / V)


# --------------------------------------------------------------------------- #
# The model
# --------------------------------------------------------------------------- #
@dataclass
class DCFResult:
    fair: float = None
    ev: float = None
    equity: float = None
    pv_fcf_sum: float = None
    pv_tv: float = None
    tv_fraction: float = None          # PV(TV) / EV  — how much rides on year 11+
    rows: list = field(default_factory=list)
    flags: list = field(default_factory=list)   # accuracy warnings, human-readable
    error: str = None


def two_stage_dcf(fcf0, g1, g2, g_term, r, cash, debt, shares,
                  tv_fraction_warn=0.85):
    """Two-stage (5yr + 5yr) FCF model with a Gordon terminal value.

    Hard guards return an error (model is undefined / meaningless):
      * r <= g_term            -> terminal value blows up / goes negative
      * shares missing/<=0
      * fcf0 is None
      * fcf0 <= 0              -> a growth DCF on negative base FCF is nonsense;
                                  better to refuse than to print a confident
                                  negative fair value.

    Soft flags do NOT stop the calc but travel with the result so the UI can
    surface them (this is the difference between "automated" and "automated
    and trustworthy"):
      * terminal value is an outsized share of enterprise value
      * net cash is large relative to EV (valuation leans on the balance sheet)
    """
    if fcf0 is None:
        return DCFResult(error="Free cash flow unavailable.")
    if not shares or shares <= 0:
        return DCFResult(error="Shares outstanding unavailable.")
    if fcf0 <= 0:
        return DCFResult(error="Base FCF is negative — two-stage DCF not meaningful.")
    if r <= g_term:
        return DCFResult(error="Discount rate must exceed terminal growth.")

    rows, pv_sum, fcf = [], 0.0, float(fcf0)
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

    flags = []
    tv_frac = pv_tv / ev if ev else None
    if tv_frac is not None and tv_frac > tv_fraction_warn:
        flags.append(f"Terminal value is {tv_frac:.0%} of EV — result hinges on yr 11+.")
    net_cash = (cash or 0.0) - (debt or 0.0)
    if ev and abs(net_cash) > 0.25 * ev:
        flags.append("Net cash/debt is large vs EV — balance sheet drives the price.")

    return DCFResult(fair=equity / shares, ev=ev, equity=equity,
                     pv_fcf_sum=pv_sum, pv_tv=pv_tv, tv_fraction=tv_frac,
                     rows=rows, flags=flags)


# --------------------------------------------------------------------------- #
# Assumption seeding + cross-checks
# --------------------------------------------------------------------------- #
def derive_assumptions(stock, rf, erp, terminal, cagr_cap=(-0.10, 0.25),
                       fallback_g1=0.10):
    """Turn a fetched `stock` dict into the four model assumptions (decimals).

    Priority for yr1-5 growth: analyst 5y estimate -> robust historical FCF
    CAGR -> flat fallback. yr6-10 fades halfway to terminal. WACC via CAPM.
    Returns (assumptions_dict, g1_source_str).
    """
    if stock.get("analyst_g5") is not None:
        g1 = stock["analyst_g5"]
        src = "analyst"
    elif stock.get("hist_cagr") is not None:
        lo, hi = cagr_cap
        g1 = max(lo, min(hi, stock["hist_cagr"]))
        src = "hist-CAGR"
    else:
        g1 = fallback_g1
        src = "fallback"
    g2 = (g1 + terminal) / 2.0
    wacc = compute_wacc(stock.get("beta"), rf, erp, stock.get("market_cap"),
                        stock.get("debt"), stock.get("interest_expense"),
                        stock.get("tax_rate"))
    return {"g1": g1, "g2": g2, "gt": terminal, "r": wacc}, src


def data_quality_flags(stock, mktcap_tol=0.05):
    """Catch the data problems that silently corrupt a DCF at scale.

    The headline one: price x shares should ≈ market cap. When a vendor's
    share count is stale relative to a split or buyback (exactly the trap the
    ServiceNow 5:1 split sets), these diverge — and a divergence here means
    your per-share fair value is being divided by the wrong denominator.
    """
    flags = []
    price = stock.get("price")
    shares = stock.get("shares")
    mc = stock.get("market_cap")
    if price and shares and mc:
        implied = price * shares
        if abs(implied - mc) / mc > mktcap_tol:
            flags.append(
                f"price x shares ({implied/1e9:,.1f}B) != market cap "
                f"({mc/1e9:,.1f}B) — possible stale share count or split lag.")
    fcf = stock.get("fcf")
    if fcf is not None and fcf <= 0:
        flags.append("Latest FCF is <= 0.")
    return flags
