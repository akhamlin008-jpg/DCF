"""
pages/3_Risk_Report.py — professional risk report for the long/short book the
Factor page selected. Every number is computed from the return matrix and
statements; assumptions (shock sizes, confidence, horizon, distribution) are
explicit sidebar inputs, never hard-coded magic.

Design choice stated up front: position sizing and risk-parity are LONG-ONLY
clean, so the SIZED book is the long candidates. Shorts and index instruments
appear in the hedging section as overlays. This is a deliberate, disclosed
modeling decision, not an omission.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

import risk_core as rk
import data_layer as dl
from universe import TICKERS, THEME_CSS, hero

st.set_page_config(page_title="Risk Report", layout="wide")
st.markdown(THEME_CSS, unsafe_allow_html=True)
st.markdown(hero('Risk <span>Report</span>',
                 'Correlation · concentration · rate & recession stress · VaR/CVaR · '
                 'risk-parity sizing · hedges · rebalancing. All computed, all assumption-tagged.'),
            unsafe_allow_html=True)

TRADING_DAYS = 252


@st.cache_data(ttl=24 * 3600, show_spinner="Loading universe data (cached 24h)…")
def _load(tickers):
    return dl.load_universe(list(tickers))


payload = _load(tuple(TICKERS))
rows = {r["ticker"]: r for r in payload["rows"]}
rets_all = payload["returns"]
bench = payload["benchmark"]
dyield = payload["dyield"]
adv = payload["adv"]

# ---- book selection -------------------------------------------------------- #
default_long = st.session_state.get("long_candidates")
default_short = st.session_state.get("short_candidates", [])
if not default_long:
    st.info("No candidates handed over from the Factor page — pick a book below "
            "(or run Factor Analysis first).")
    default_long = TICKERS[:8]

avail_syms = [t for t in TICKERS if t in rets_all.columns]
st.sidebar.header("Book")
longs = st.sidebar.multiselect("Long positions", avail_syms,
                               default=[t for t in default_long if t in avail_syms])
shorts = st.sidebar.multiselect("Short candidates (hedging overlay)", avail_syms,
                                default=[t for t in default_short if t in avail_syms])

if len(longs) < 2:
    st.warning("Select at least 2 long positions to compute portfolio risk.")
    st.stop()

st.sidebar.header("Capital & assumptions")
capital = st.sidebar.number_input("Capital ($)", value=1_000_000, step=100_000, format="%d")
sizing = st.sidebar.selectbox("Sizing method",
    ["Equal weight", "Inverse-vol", "Risk parity (ERC)", "Vol target"], index=2)
target_vol = st.sidebar.slider("Target vol (annual, for Vol target) %", 5, 30, 12) / 100
conf = st.sidebar.select_slider("VaR confidence", [0.90, 0.95, 0.975, 0.99], value=0.95)
horizon = st.sidebar.slider("VaR horizon (trading days)", 1, 21, 1)
participation = st.sidebar.slider("Liquidation participation %", 5, 50, 20) / 100
use_lw = st.sidebar.toggle("Ledoit-Wolf covariance shrinkage", value=True)

st.sidebar.header("Recession scenario")
mkt_shock = st.sidebar.slider("Equity shock %", -60, 0, -35) / 100
rate_shock_bp = st.sidebar.slider("10y yield shock (bp)", -300, 100, -150)
rate_shock = rate_shock_bp / 10000.0   # bp -> decimal yield change

# ---- returns matrix for the long book -------------------------------------- #
R = rets_all[longs].dropna(how="any")
n = len(longs)
mkt_ret = rets_all[bench].reindex(R.index)
dy = dyield.reindex(R.index).fillna(0.0)

S_sample = rk.sample_cov(R.values, ddof=1)
if use_lw:
    S, delta = rk.ledoit_wolf_cc(R.values)
else:
    S, delta = S_sample, 0.0
S_ann = rk.annualize_cov(S, TRADING_DAYS)
sig_i_ann = np.sqrt(np.diag(S_ann))

# ---- sizing ---------------------------------------------------------------- #
if sizing == "Equal weight":
    w = np.full(n, 1.0 / n); converged, disp = True, np.nan
elif sizing == "Inverse-vol":
    w = rk.inverse_vol_weights(S); converged, disp = True, np.nan
elif sizing == "Risk parity (ERC)":
    w, converged, _, disp = rk.erc_weights(S)
else:  # vol target -> ERC base then scale gross
    w_base, converged, _, disp = rk.erc_weights(S)
    scale = rk.vol_target_scale(w_base, S, target_vol, TRADING_DAYS)
    w = w_base * (scale if np.isfinite(scale) else 1.0)

w_norm = w / w.sum() if w.sum() else np.full(n, 1.0 / n)   # for % displays
port_vol_ann = rk.port_vol(w_norm, S_ann)
rc_vec, rc_pct = rk.risk_contributions(w_norm, S_ann)
position_dollars = w_norm * capital
port_ret = R.values @ w_norm                                # daily series

st.caption(f"Book: {n} longs · {len(R)} aligned trading days · "
           f"sizing **{sizing}**"
           + (f" (ERC dispersion {disp:.1e}, converged={converged})" if np.isfinite(disp) else "")
           + (f" · LW shrinkage δ={delta:.2f}" if use_lw else " · sample covariance")
           + f" · data as of {payload['as_of']:%Y-%m-%d}")

# =========================================================================== #
# 1. HEAT-MAP SUMMARY TABLE (per position)
# =========================================================================== #
st.header("1 · Position risk heat map")

betas_m, betas_r, r2s = [], [], []
for tk in longs:
    bm, br, r2 = rk.factor_betas(R[tk].values, mkt_ret.values, dy.values)
    betas_m.append(bm); betas_r.append(br * 0.01); r2s.append(r2)   # br per +100bp
betas_m = np.array(betas_m); betas_r = np.array(betas_r)

adv_book = np.array([adv.get(tk) or np.nan for tk in longs])
liq_days = rk.liquidity_days(position_dollars, adv_book, participation)
liq_rate = rk.liquidity_rating(liq_days)

# per-name contribution to a recession scenario
_, stress_contrib = rk.stress_factor_model(w_norm, betas_m, betas_r / 0.01,
                                            mkt_shock, rate_shock)

summary = pd.DataFrame({
    "Ticker": longs,
    "Sector": [rows[t].get("sector") or "—" for t in longs],
    "Weight %": w_norm * 100,
    "Risk %": rc_pct * 100,
    "Vol (ann) %": sig_i_ann * 100,
    "β mkt": betas_m,
    "β rate /100bp %": betas_r * 100,
    "Liq days": liq_days,
    "Liquidity": liq_rate,
    "Stress %": stress_contrib * 100,
})
summary = summary.sort_values("Risk %", ascending=False).reset_index(drop=True)
sty = (summary.style.format({
        "Weight %": "{:.1f}", "Risk %": "{:.1f}", "Vol (ann) %": "{:.1f}",
        "β mkt": "{:.2f}", "β rate /100bp %": "{:+.2f}", "Liq days": "{:.2f}",
        "Stress %": "{:+.1f}"})
       .background_gradient(cmap="Reds", subset=["Risk %"], vmin=0, vmax=max(25, rc_pct.max()*100))
       .background_gradient(cmap="Reds", subset=["Vol (ann) %"])
       .background_gradient(cmap="RdYlGn_r", subset=["Stress %"])
       .background_gradient(cmap="Reds", subset=["Liq days"]))
st.dataframe(sty, hide_index=True, use_container_width=True)
st.caption("Risk % = component contribution to portfolio volatility (Euler "
           "decomposition; sums to 100%). It diverges from Weight % exactly "
           "because of correlation — a moderate-weight name that co-moves with "
           "everything carries more risk than its weight suggests.")

cols = st.columns(4)
cols[0].metric("Portfolio vol (ann)", f"{port_vol_ann*100:.1f}%")
cols[1].metric("Diversification ratio", f"{rk.diversification_ratio(w_norm, S_ann):.2f}")
cols[2].metric("Portfolio β (mkt)", f"{float(w_norm @ betas_m):.2f}")
cols[3].metric("Portfolio β rate /100bp", f"{float(w_norm @ betas_r)*100:+.2f}%")

# =========================================================================== #
# 2. CORRELATION
# =========================================================================== #
st.header("2 · Correlation between holdings")
C = rk.cov_to_corr(S)
cdf = pd.DataFrame(C, index=longs, columns=longs)
st.dataframe(cdf.style.format("{:.2f}")
             .background_gradient(cmap="RdYlGn_r", vmin=-1, vmax=1),
             use_container_width=True)
avg_c = rk.avg_pairwise_corr(C)
pairs = rk.most_correlated_pairs(C, longs, k=5)
st.markdown(f"**Average pairwise correlation: {avg_c:.2f}.** "
            "Most correlated pairs (these are your concentrated bets in disguise):")
st.table(pd.DataFrame(pairs, columns=["A", "B", "Corr"]).style.format({"Corr": "{:.2f}"}))

# =========================================================================== #
# 3. SECTOR CONCENTRATION
# =========================================================================== #
st.header("3 · Sector concentration")
sec_w = {}
for tk, wt in zip(longs, w_norm):
    s = rows[tk].get("sector") or "Unknown"
    sec_w[s] = sec_w.get(s, 0.0) + wt
hhi, eff_n, shares = rk.herfindahl(sec_w)
sec_df = (pd.DataFrame([{"Sector": k, "Weight %": v * 100} for k, v in shares.items()])
          .sort_values("Weight %", ascending=False).reset_index(drop=True))
c1, c2 = st.columns([2, 1])
with c1:
    st.dataframe(sec_df.style.format({"Weight %": "{:.1f}"})
                 .background_gradient(cmap="Reds", subset=["Weight %"], vmin=0, vmax=100),
                 hide_index=True, use_container_width=True)
with c2:
    st.metric("HHI", f"{hhi:.3f}")
    st.metric("Effective # sectors", f"{eff_n:.1f}")
    st.caption("HHI>0.25 or effective sectors < ~3 = concentrated. The S&P 500 "
               "sector HHI sits near 0.15 for reference.")

# =========================================================================== #
# 4. GEOGRAPHIC / CURRENCY (PROXY — disclosed)
# =========================================================================== #
st.header("4 · Domicile & currency exposure (proxy)")
st.warning("This is HQ country and reporting currency, weighted by position — "
           "NOT revenue-by-geography. True geographic revenue exposure requires "
           "10-K segment disclosure, which yfinance does not provide. Read this "
           "as a listing/FX-reporting proxy only.")
geo, cur = {}, {}
for tk, wt in zip(longs, w_norm):
    c = rows[tk].get("_country") or "Unknown"
    ccy = rows[tk].get("_currency") or "Unknown"
    geo[c] = geo.get(c, 0.0) + wt
    cur[ccy] = cur.get(ccy, 0.0) + wt
g1, g2 = st.columns(2)
g1.dataframe(pd.DataFrame([{"Country (HQ)": k, "Weight %": v*100} for k, v in
             sorted(geo.items(), key=lambda x: -x[1])]).style.format({"Weight %": "{:.1f}"}),
             hide_index=True, use_container_width=True)
g2.dataframe(pd.DataFrame([{"Reporting ccy": k, "Weight %": v*100} for k, v in
             sorted(cur.items(), key=lambda x: -x[1])]).style.format({"Weight %": "{:.1f}"}),
             hide_index=True, use_container_width=True)

# =========================================================================== #
# 5. INTEREST-RATE SENSITIVITY
# =========================================================================== #
st.header("5 · Interest-rate sensitivity")
port_brate = float(w_norm @ betas_r)   # already per +100bp
st.markdown(
    f"Empirical 2-factor regression (return ~ market + Δ10y yield) per name. "
    f"**Portfolio rate beta ≈ {port_brate*100:+.2f}% return per +100bp** in the "
    f"10y. Sign and size are regime-dependent — this is the realized "
    f"relationship over the sample window, not a structural constant.")
rate_df = pd.DataFrame({
    "Ticker": longs, "β rate /100bp %": betas_r * 100,
    "β mkt": betas_m, "Regression R²": r2s}).sort_values("β rate /100bp %")
st.dataframe(rate_df.style.format({"β rate /100bp %": "{:+.2f}", "β mkt": "{:.2f}",
             "Regression R²": "{:.2f}"})
             .background_gradient(cmap="RdYlGn", subset=["β rate /100bp %"]),
             hide_index=True, use_container_width=True)

# =========================================================================== #
# 6. RECESSION STRESS TEST
# =========================================================================== #
st.header("6 · Recession stress test")
dP, _ = rk.stress_factor_model(w_norm, betas_m, betas_r / 0.01, mkt_shock, rate_shock)
hist_mdd, pk, tr = rk.max_drawdown(port_ret)
c1, c2, c3 = st.columns(3)
c1.metric(f"Factor-model P&L ({int(mkt_shock*100)}% eq, {rate_shock_bp:+}bp)",
          f"{dP*100:+.1f}%")
c2.metric("Est. $ impact", f"${dP*capital:,.0f}")
c3.metric("Worst historical drawdown (sample)", f"-{hist_mdd*100:.1f}%")
st.caption("Factor-model figure is a FIRST-ORDER, idiosyncratic-zero estimate: "
           "dP ≈ Σ wᵢ(βₘ·eq_shock + β_rate·rate_shock). It ignores convexity and "
           "correlation breakdown, both of which worsen real drawdowns. The "
           "historical drawdown is the actual worst peak-to-trough of THIS sized "
           "book over the available window — a backward-looking complement, not a "
           "forecast. Neither is a probability of recession.")

# =========================================================================== #
# 7. TAIL RISK (VaR / CVaR + distributional scenario probabilities)
# =========================================================================== #
st.header("7 · Tail risk")
h = np.sqrt(horizon)
vh = rk.var_historical(port_ret, conf) * h
vg = rk.var_gaussian(port_ret, conf) * h
vcf = rk.var_cornish_fisher(port_ret, conf) * h
cv = rk.cvar_historical(port_ret, conf) * h
t1, t2, t3, t4 = st.columns(4)
t1.metric(f"VaR {conf:.0%} ({horizon}d) hist", f"-{vh*100:.1f}%")
t2.metric("VaR Gaussian", f"-{vg*100:.1f}%")
t3.metric("VaR Cornish-Fisher", f"-{vcf*100:.1f}%")
t4.metric(f"CVaR {conf:.0%} (expected shortfall)", f"-{cv*100:.1f}%")
st.caption("Cornish-Fisher corrects the Gaussian quantile for the book's actual "
           "skew and kurtosis; when it exceeds the Gaussian VaR, your tail is "
           "fatter than normal. Horizon scaled by √t (iid assumption). "
           f"≈ ${vcf*capital:,.0f} at risk at {conf:.0%} over {horizon} day(s) "
           "on the Cornish-Fisher measure.")

# rolling-horizon empirical scenario probabilities (no fabricated macro odds)
roll = pd.Series(port_ret).rolling(horizon).sum().dropna().values
st.markdown("**Scenario probabilities** — frequency of a loss at least this large "
            f"over any {horizon}-day window in the sample, vs the Gaussian fit:")
scen = []
for thr in (0.05, 0.10, 0.20):
    p = rk.tail_prob(roll, thr)
    scen.append({"Loss ≥": f"{int(thr*100)}%",
                 "Empirical P": p["empirical"], "Gaussian P": p["gaussian"]})
st.table(pd.DataFrame(scen).style.format({"Empirical P": "{:.1%}", "Gaussian P": "{:.1%}"}))
st.caption("These are DISTRIBUTIONAL tail probabilities under the sample / "
           "Gaussian assumption — not forecasts of macro events. Empirical P is "
           "limited by how many independent windows the sample contains.")

# =========================================================================== #
# 8. TOP-3 RISKS + HEDGES
# =========================================================================== #
st.header("8 · Top-3 risks & hedges")
port_beta = float(w_norm @ betas_m)
top_name_i = int(np.argmax(rc_pct))
top_sector = max(shares, key=shares.get) if shares else None
risk_scores = {
    "Equity directional (market beta)": abs(port_beta),
    f"Single-name concentration ({longs[top_name_i]})": rc_pct[top_name_i],
    f"Sector concentration ({top_sector})": shares.get(top_sector, 0) if top_sector else 0,
    "Rate sensitivity": abs(port_brate) * 5,   # scale to compare on similar footing
    "Tail / drawdown": hist_mdd,
}
ranked = sorted(risk_scores.items(), key=lambda x: x[1], reverse=True)[:3]

hedges = []
for name, _ in ranked:
    if name.startswith("Equity"):
        notional = port_beta * capital
        hedges.append((name,
            f"Short ≈ ${notional:,.0f} of {bench} (β≈1) to neutralize market beta "
            f"of {port_beta:.2f}; a partial hedge of half that cuts beta to "
            f"≈{port_beta/2:.2f}. Index puts are the convex alternative if you "
            f"want to keep upside."))
    elif name.startswith("Single-name"):
        tk = longs[top_name_i]
        hedges.append((name,
            f"{tk} carries {rc_pct[top_name_i]*100:.0f}% of portfolio risk on "
            f"{w_norm[top_name_i]*100:.0f}% weight. Trimming it toward its "
            f"risk-parity weight, or a single-name collar, is the direct lever."))
    elif name.startswith("Sector"):
        sw = shares.get(top_sector, 0)
        hedges.append((name,
            f"{top_sector} is {sw*100:.0f}% of the book. A short in the matching "
            f"sector ETF sized to ≈${sw*capital:,.0f} offsets the concentrated "
            f"factor while keeping your single-name selection."))
    elif name.startswith("Rate"):
        hedges.append((name,
            f"Portfolio loses ≈{port_brate*100:+.2f}% per +100bp. If long-duration "
            f"exposure is unwanted, pair with rate-hedged or short-duration "
            f"instruments sized to flatten the {port_brate*100:+.2f}% sensitivity."))
    else:
        hedges.append((name,
            f"Historical drawdown reached -{hist_mdd*100:.0f}%. A standing "
            f"out-of-the-money index put overlay, budgeted as a fixed % of "
            f"capital per quarter, caps the left tail measured in section 7."))
st.table(pd.DataFrame(hedges, columns=["Risk", "Hedge (computed sizing)"]))
st.caption("Hedge notionals are mechanical translations of the measured "
           "exposures above. They are analytical, not investment advice, and "
           "ignore financing/borrow cost, basis risk, and option premium.")

# =========================================================================== #
# 9. REBALANCING
# =========================================================================== #
st.header("9 · Rebalancing to target")
w_current = np.full(n, 1.0 / n)            # assume you hold equal-weight today
rebal = pd.DataFrame({
    "Ticker": longs,
    "Current %": w_current * 100,
    "Target %": w_norm * 100,
    "Δ %": (w_norm - w_current) * 100,
    "Δ $": (w_norm - w_current) * capital,
    "Target risk %": rc_pct * 100,
}).sort_values("Δ %", ascending=False).reset_index(drop=True)
st.dataframe(rebal.style.format({
    "Current %": "{:.1f}", "Target %": "{:.1f}", "Δ %": "{:+.1f}",
    "Δ $": "${:+,.0f}", "Target risk %": "{:.1f}"})
    .background_gradient(cmap="RdYlGn", subset=["Δ %"]),
    hide_index=True, use_container_width=True)
gross = float(np.abs(w).sum())
st.caption(f"Target weights are the **{sizing}** solution. Current assumed "
           f"equal-weight; replace with your live book to get real trade sizes. "
           + (f"Gross exposure {gross*100:.0f}% (vol-target leverage applied). "
              if sizing == "Vol target" else ""))

# ---- shorts overlay note --------------------------------------------------- #
if shorts:
    st.divider()
    st.subheader("Short candidates (overlay, not sized above)")
    st.write(", ".join(shorts))
    st.caption("Held separately because long-only risk parity is clean; fold "
               "these in as signed weights only if you want a true long/short "
               "covariance — at which point ERC needs a long/short-aware solver.")

st.divider()
st.caption("FULL DISCLOSURE: outputs are model estimates on unofficial scraped "
           "data over one historical window. Betas, correlations, and tail "
           "measures are regime-dependent and will shift out of sample. This is "
           "analytical tooling, not investment advice or a price target. Verify "
           "fundamentals against SEC EDGAR before acting.")
