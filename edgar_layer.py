"""
edgar_layer.py — fundamentals from SEC EDGAR's free XBRL API.

WHY
---
yfinance scrapes Yahoo and gets throttled/blocked, worst of all on the
datacenter IPs you actually deploy on. SEC EDGAR is the official primary source,
is completely free with no API key, has no daily cap, and does NOT punish cloud
IPs the way Yahoo does. The only requirements are a descriptive User-Agent with
a real contact email and staying under ~10 requests/second.

This module returns the same scalar fundamentals the factor/DCF engine needs
(revenue, net income, assets, equity, cash, debt, FCF series, EPS series),
sourced from each company's most recent annual XBRL facts. It is OPT-IN: set
USE_EDGAR=1 (or pass use_edgar=True in data_layer) to route fundamentals here.

CAVEATS (keep the honesty ledger honest)
-----------------------------------------
* US filers only. Foreign private issuers (TSM, ASML, HSBC, ARM) file 20-F and
  are absent from us-gaap companyfacts — data_layer keeps Yahoo for those.
* Companies tag inconsistently, so every concept has a fallback list. A missing
  tag returns None and the factor engine reweights around it — never imputed.
* This is NOT live-tested in the build sandbox (no network to data.sec.gov).
  Run edgar_smoke_test() on your machine before trusting the numbers.

REQUIRED: set a real contact email in USER_AGENT below or EDGAR returns 403.
"""
from __future__ import annotations

import os
import time

import requests

import cache_layer as kv

# >>> PUT A REAL CONTACT EMAIL HERE <<< — EDGAR 403s generic/blank User-Agents.
USER_AGENT = os.environ.get(
    "EDGAR_USER_AGENT", "DCF-Engine (set EDGAR_USER_AGENT env var) contact@example.com"
)
HEADERS = {"User-Agent": USER_AGENT, "Accept-Encoding": "gzip, deflate"}
BASE = "https://data.sec.gov"
MIN_INTERVAL = 0.12  # ~8 req/s, under the 10 req/s SEC limit
_last_call = [0.0]


def _throttle():
    dt = time.time() - _last_call[0]
    if dt < MIN_INTERVAL:
        time.sleep(MIN_INTERVAL - dt)
    _last_call[0] = time.time()


# --------------------------------------------------------------------------- #
# ticker -> CIK
# --------------------------------------------------------------------------- #
def cik_map() -> dict:
    cached = kv.get("edgar:cik_map", max_age_sec=7 * kv.DAY)
    if cached:
        return cached
    _throttle()
    r = requests.get("https://www.sec.gov/files/company_tickers.json",
                     headers=HEADERS, timeout=30)
    r.raise_for_status()
    m = {}
    for row in r.json().values():
        m[row["ticker"].upper()] = f'{int(row["cik_str"]):010d}'
    kv.put("edgar:cik_map", m)
    return m


# --------------------------------------------------------------------------- #
# companyfacts
# --------------------------------------------------------------------------- #
def company_facts(ticker: str):
    key = f"edgar:facts:{ticker.upper()}"
    cached = kv.get(key)
    if cached is not None:
        return cached
    cik = cik_map().get(ticker.upper())
    if not cik:
        return None
    _throttle()
    r = requests.get(f"{BASE}/api/xbrl/companyfacts/CIK{cik}.json",
                     headers=HEADERS, timeout=30)
    if r.status_code != 200:
        return None
    facts = r.json()
    kv.put(key, facts)
    return facts


# --------------------------------------------------------------------------- #
# fact extraction
# --------------------------------------------------------------------------- #
def _annual_series(facts, *tags, taxonomy="us-gaap", unit="USD"):
    """Oldest->newest list of distinct fiscal-year annual values for the first
    matching tag. Dedupes by fiscal year (keeps the latest filing's value)."""
    if not facts:
        return []
    for tag in tags:
        try:
            units = facts["facts"][taxonomy][tag]["units"][unit]
        except (KeyError, TypeError):
            continue
        annual = {}
        for u in units:
            form = (u.get("form") or "")
            fp = u.get("fp")
            fy = u.get("fy")
            if fy is None:
                continue
            # annual figures: 10-K / 20-F full-year (fp == 'FY')
            if fp == "FY" and ("10-K" in form or "20-F" in form):
                annual[fy] = float(u["val"])
        if annual:
            return [annual[fy] for fy in sorted(annual)]
    return []


def _latest(facts, *tags, **kw):
    s = _annual_series(facts, *tags, **kw)
    return s[-1] if s else None


def fundamentals(ticker: str) -> dict:
    """Return the scalar fundamentals data_layer needs, from EDGAR. Keys mirror
    what fetch_one already produces so the factor/DCF engine is unaffected.
    Missing concepts are None (reweighted downstream, never imputed)."""
    f = company_facts(ticker)
    out = {"ticker": ticker.upper(), "_source_fund": "edgar", "error": None}
    if f is None:
        out["error"] = "edgar: no companyfacts (non-US filer or unknown CIK)"
        return out

    revenue = _latest(
        f, "Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax",
        "RevenueFromContractWithCustomerIncludingAssessedTax", "SalesRevenueNet")
    rev_series = _annual_series(
        f, "Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax",
        "RevenueFromContractWithCustomerIncludingAssessedTax", "SalesRevenueNet")
    net_income = _latest(f, "NetIncomeLoss",
                         "ProfitLoss", "NetIncomeLossAvailableToCommonStockholdersBasic")
    gross = _latest(f, "GrossProfit")
    ebit = _latest(f, "OperatingIncomeLoss")
    assets = _latest(f, "Assets")
    equity = _latest(f, "StockholdersEquity",
                     "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest")
    cash = _latest(f, "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents",
                   "CashAndCashEquivalentsAtCarryingValue")
    debt_lt = _latest(f, "LongTermDebtNoncurrent", "LongTermDebt") or 0.0
    debt_st = _latest(f, "DebtCurrent", "ShortTermBorrowings") or 0.0
    debt = (debt_lt + debt_st) or None
    interest = _latest(f, "InterestExpense", "InterestExpenseDebt")
    cfo = _latest(f, "NetCashProvidedByUsedInOperatingActivities")
    cfo_series = _annual_series(f, "NetCashProvidedByUsedInOperatingActivities")
    capex_series = _annual_series(f, "PaymentsToAcquirePropertyPlantAndEquipment")
    eps_series = _annual_series(f, "EarningsPerShareDiluted", "EarningsPerShareBasic",
                                unit="USD/shares")

    # FCF series = OCF - CapEx, aligned by position (both oldest->newest)
    fcf_series = []
    if cfo_series and capex_series:
        k = min(len(cfo_series), len(capex_series))
        fcf_series = [cfo_series[-k + i] - capex_series[-k + i] for i in range(k)]
    fcf = fcf_series[-1] if fcf_series else None

    out.update({
        "_revenue": revenue, "_net_income": net_income, "_gross": gross,
        "_ebit": ebit, "_assets": assets, "_equity": equity, "_cash": cash,
        "_debt": debt, "_interest": interest, "_cfo": cfo,
        "_fcf": fcf, "fcf_series": fcf_series,
        "rev_series": rev_series, "eps_series": eps_series,
    })
    return out


def edgar_smoke_test(ticker="AAPL"):
    """Run on a networked machine to sanity-check EDGAR access + parsing."""
    print(f"User-Agent: {USER_AGENT}")
    m = cik_map()
    print(f"CIK map entries: {len(m)}; {ticker} -> {m.get(ticker)}")
    fund = fundamentals(ticker)
    for k, v in fund.items():
        print(f"  {k}: {v}")
    return fund


if __name__ == "__main__":
    edgar_smoke_test()
