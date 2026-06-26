"""
pages/2_Factor_Analysis.py — multi-factor heat map and long/short candidate
generation. Standalone page; feeds the Risk Report via st.session_state.

Method note shown to the user, not buried: scores are CROSS-SECTIONAL. A grade
of A means "top of THIS universe right now," not an absolute verdict. Change the
universe and every score moves. The composite is a weighted blend of
standardized factor scores with missing factors reweighted away, then
re-standardized — so the column you rank on is a clean z within the universe.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

import factor_core as fc
import data_layer as dl
from universe import TICKERS, THEME_CSS, NVIDIA_GREEN, hero

st.set_page_config(page_title="Factor Analysis", layout="wide")
st.markdown(THEME_CSS, unsafe_allow_html=True)
st.markdown(hero('Factor <span>Heat Map</span>',
                 'Eight-factor cross-sectional scoring · DCF is one factor, not the verdict · '
                 'longs and shorts fall out of the composite ranking.'),
            unsafe_allow_html=True)


@st.cache_data(ttl=24 * 3600, show_spinner="Pulling financials + prices (cached 24h)…")
def _load(tickers):
    return dl.load_universe(list(tickers))


payload = _load(tuple(TICKERS))
rows = payload["rows"]
avail = payload["availability"]

# ---- sidebar controls ------------------------------------------------------ #
st.sidebar.header("Standardization")
method = st.sidebar.selectbox(
    "Method", ["rank", "z", "zmean"], index=0,
    help="rank = rank-based inverse-normal (outlier-proof, recommended). "
         "z = robust median/MAD z. zmean = classic mean/std z.")
neutralize = st.sidebar.toggle("Sector-neutralize", value=False,
    help="Score each name vs its own sector (intra-sector ranking).")

st.sidebar.header("Factor weights")
st.sidebar.caption("Renormalized over factors actually present per name.")
weights = {}
for g, w0 in fc.DEFAULT_GROUP_WEIGHTS.items():
    cov = avail.get(g, 0.0)
    tag = "cov-good" if cov >= 0.66 else "cov-mid" if cov >= 0.33 else "cov-bad"
    st.sidebar.markdown(
        f"<span class='note'>{g} · coverage "
        f"<span class='{tag}'>{cov*100:.0f}%</span></span>", unsafe_allow_html=True)
    weights[g] = st.sidebar.slider(g, 0.0, 0.40, float(w0), 0.005,
                                   key=f"w_{g}", label_visibility="collapsed")

st.sidebar.header("Selection")
n_long = st.sidebar.slider("# long candidates", 3, 15, 8)
n_short = st.sidebar.slider("# short candidates", 3, 15, 8)
min_groups = st.sidebar.slider("Min factors observed", 2, 8, 3,
    help="Names with fewer observable factor groups are excluded from selection.")

# ---- score the universe ---------------------------------------------------- #
result = fc.score_universe(rows, weights=weights, method=method,
                           neutralize=neutralize)

tickers = [r["ticker"] for r in rows]
sectors = [r.get("sector") or "—" for r in rows]

# ---- data-honesty banner --------------------------------------------------- #
weak = [g for g in ("short_interest", "insider", "institutional")
        if avail.get(g, 0) < 0.5]
if weak:
    st.warning(
        "Thin data on: " + ", ".join(weak) +
        ". These come from Yahoo's scraped short-interest / holders / insider "
        "feeds, which are stale or missing for many large caps. They are "
        "down-weighted automatically per name, but treat their scores as soft.")
st.caption(
    "Geographic / currency factor is intentionally absent: yfinance exposes HQ "
    "country and reporting currency only, not revenue-by-region (that needs "
    "10-K segment data). Domicile/currency are shown in the Risk Report, "
    "labeled as a proxy — never as real geographic revenue exposure.")

# ---- heat map table -------------------------------------------------------- #
group_order = list(fc.GROUP_METRICS.keys())
heat = pd.DataFrame({"Ticker": tickers, "Sector": sectors})
for g in group_order:
    heat[g] = result["group_score"][g]
heat["COMPOSITE"] = result["composite"]
heat["Pct"] = result["percentile"]
heat["Grade"] = result["grade"]
heat = heat.sort_values("COMPOSITE", ascending=False, na_position="last").reset_index(drop=True)

st.subheader("Factor scores (standardized, cross-sectional)")
sty = (heat.style
       .format({g: "{:+.2f}" for g in group_order} |
               {"COMPOSITE": "{:+.2f}", "Pct": "{:.0f}"}, na_rep="·")
       .background_gradient(cmap="RdYlGn", subset=group_order, vmin=-2.0, vmax=2.0)
       .background_gradient(cmap="RdYlGn", subset=["COMPOSITE"], vmin=-2.0, vmax=2.0)
       .background_gradient(cmap="Greens", subset=["Pct"], vmin=0, vmax=100))
st.dataframe(sty, hide_index=True, use_container_width=True, height=560)
st.caption("Cell = standardized factor score (z). Green = attractive for a LONG, "
           "red = unattractive. '·' = factor not observable for that name "
           "(it was reweighted out of that name's composite, not scored 0).")

# ---- long / short candidates ----------------------------------------------- #
longs, shorts = fc.select_candidates(tickers, result, n_long=n_long,
                                     n_short=n_short, min_groups_long=min_groups,
                                     min_groups_short=min_groups)

def _breakdown(tk):
    i = tickers.index(tk)
    d = {g: result["group_score"][g][i] for g in group_order}
    d["sector"] = sectors[i]
    return d

cL, cR = st.columns(2)
with cL:
    st.subheader("🟢 Long candidates")
    dfl = pd.DataFrame([{"Ticker": t, "Composite": c, "Pct": p, **_breakdown(t)}
                        for (t, c, p) in longs])
    if not dfl.empty:
        st.dataframe(dfl.style.format(
            {"Composite": "{:+.2f}", "Pct": "{:.0f}"} |
            {g: "{:+.2f}" for g in group_order}, na_rep="·")
            .background_gradient(cmap="RdYlGn", subset=group_order, vmin=-2, vmax=2),
            hide_index=True, use_container_width=True)
with cR:
    st.subheader("🔴 Short candidates")
    dfs = pd.DataFrame([{"Ticker": t, "Composite": c, "Pct": p, **_breakdown(t)}
                        for (t, c, p) in shorts])
    if not dfs.empty:
        st.dataframe(dfs.style.format(
            {"Composite": "{:+.2f}", "Pct": "{:.0f}"} |
            {g: "{:+.2f}" for g in group_order}, na_rep="·")
            .background_gradient(cmap="RdYlGn_r", subset=group_order, vmin=-2, vmax=2),
            hide_index=True, use_container_width=True)

# ---- hand off to the risk page --------------------------------------------- #
st.session_state["long_candidates"] = [t for (t, _, _) in longs]
st.session_state["short_candidates"] = [t for (t, _, _) in shorts]
st.session_state["factor_result"] = {
    "tickers": tickers,
    "composite": result["composite"].tolist(),
    "grade": list(result["grade"]),
    "sector": sectors,
}

st.divider()
n_l, n_s = len(longs), len(shorts)
st.success(f"Selected {n_l} longs / {n_s} shorts → open **Risk Report** to run "
           "correlation, concentration, stress, VaR, sizing and hedges on this book.")
st.caption("Selection is by composite rank with a minimum-evidence filter. It is "
           "a screen, not a recommendation, and says nothing about position size "
           "— that is the Risk Report's job.")
