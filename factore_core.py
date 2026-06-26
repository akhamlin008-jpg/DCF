"""
factor_core.py — pure cross-sectional factor math. No streamlit / yfinance /
network, so it unit-tests offline exactly like dcf_core.py.

Pipeline (per raw metric, ACROSS the universe at one point in time):
    raw x_i  ->  winsorize  ->  standardize (z or rank-normal)  ->  sign by
    direction  ->  (optional) sector-neutralize  ->  combine within group
    ->  weight groups into a composite  ->  re-standardize  ->  percentile/grade

Why rank-normal is the DEFAULT, not classic z:
    Financial cross-sections are heavy-tailed and contain sign-flipping ratios
    (a near-zero denominator makes EV/EBIT explode). A mean/std z-score lets one
    such outlier dominate the whole factor. The rank-based inverse-normal
    transform (a.k.a. "normal scores") depends only on ordering, so it is
    immune to that, at the cost of discarding magnitude. Both are provided;
    pick per use-case.

Missing data is handled by REWEIGHTING, never by imputing 0. Imputing the
cross-sectional mean (z=0) silently asserts "this stock is exactly average on
the factor we couldn't measure" — a real, directional lie. Reweighting the
factors we DID measure is the honest choice and is what `composite()` does.

All public functions are nan-safe. Inputs are 1-D array-likes aligned to a
fixed ticker order the caller controls.
"""
from __future__ import annotations

import numpy as np
from math import erf, sqrt

# --------------------------------------------------------------------------- #
# Standardization primitives
# --------------------------------------------------------------------------- #

def _as_float(x):
    a = np.asarray(x, dtype="float64")
    return a

def winsorize(x, lo=0.02, hi=0.98):
    """Clip to the [lo, hi] empirical quantiles (nan-ignoring). Caps the
    influence of extreme observations BEFORE standardizing so a single bad
    data point can't set the scale for everyone."""
    a = _as_float(x).copy()
    m = np.isfinite(a)
    if m.sum() < 3:
        return a
    ql, qh = np.nanquantile(a[m], [lo, hi])
    a[m] = np.clip(a[m], ql, qh)
    return a

def zscore(x, robust=True):
    """Cross-sectional standardization, nan-preserving.

    robust=True : (x - median) / (1.4826 * MAD)  — 1.4826 makes MAD a
                  consistent estimator of sigma under normality, so the unit
                  matches a classic z while resisting outliers.
    robust=False: (x - mean) / std (population, ddof=0).
    Returns all-zeros (not nan) for a degenerate (zero-spread) factor so it
    contributes nothing rather than poisoning the composite with nan.
    """
    a = _as_float(x)
    m = np.isfinite(a)
    out = np.full_like(a, np.nan)
    if m.sum() < 2:
        return out
    v = a[m]
    if robust:
        med = np.median(v)
        mad = np.median(np.abs(v - med))
        scale = 1.4826 * mad
        if scale <= 0:                      # all identical (or near) -> no signal
            out[m] = 0.0
            return out
        out[m] = (v - med) / scale
    else:
        mu = v.mean()
        sd = v.std(ddof=0)
        if sd <= 0:
            out[m] = 0.0
            return out
        out[m] = (v - mu) / sd
    return out

def rank_to_normal(x):
    """Rank-based inverse-normal scores. Ranks the finite values to
    p_i = (rank - 0.5)/n (Blom-style, plotting-position), then applies the
    inverse standard-normal CDF. Output is ~N(0,1), order-preserving, and
    completely outlier-insensitive. nan stays nan."""
    a = _as_float(x)
    m = np.isfinite(a)
    out = np.full_like(a, np.nan)
    n = int(m.sum())
    if n < 2:
        return out
    v = a[m]
    order = np.argsort(np.argsort(v))            # 0..n-1 ranks, ties broken stably
    p = (order + 0.5) / n
    out[m] = np.array([_norm_ppf(pi) for pi in p])
    return out

def standardize(x, method="rank", winsor=(0.02, 0.98)):
    """Dispatch: 'rank' -> rank_to_normal, 'z' -> robust z, 'zmean' -> mean/std z.
    Winsorization is applied before the mean/std variants (it is a no-op for
    rank, which already ignores magnitude)."""
    if method == "rank":
        return rank_to_normal(x)
    xw = winsorize(x, *winsor) if winsor else x
    if method == "z":
        return zscore(xw, robust=True)
    if method == "zmean":
        return zscore(xw, robust=False)
    raise ValueError(f"unknown standardize method {method!r}")

def sector_neutralize(z, sectors):
    """Subtract the sector mean from each standardized score so the factor
    expresses INTRA-sector ranking only (a cheap stock vs cheap sector are
    different bets). Sectors with <2 members are left unchanged (no reliable
    group mean). nan-safe."""
    z = _as_float(z).copy()
    sectors = np.asarray(sectors, dtype=object)
    for s in set(sectors.tolist()):
        idx = np.where(sectors == s)[0]
        vals = z[idx]
        fin = np.isfinite(vals)
        if fin.sum() >= 2:
            z[idx[fin]] = vals[fin] - vals[fin].mean()
    return z

# --------------------------------------------------------------------------- #
# Group + composite assembly
# --------------------------------------------------------------------------- #

# Each group lists (metric_key, direction). direction = +1 if higher raw value
# is "better" (more attractive for a LONG), -1 if lower is better. The DCF
# margin-of-safety enters here as just another factor ("dcf"), so the existing
# engine is one input among several rather than the whole verdict.
GROUP_METRICS = {
    "value":          [("fcf_yield", +1), ("ebit_ev", +1),
                       ("earnings_yield", +1), ("sales_ev", +1)],
    "quality":        [("gross_margin", +1), ("op_margin", +1),
                       ("roe", +1), ("roa", +1),
                       ("debt_to_equity", -1), ("interest_coverage", +1),
                       ("accruals", -1)],
    "growth":         [("rev_cagr", +1), ("fcf_cagr", +1), ("eps_cagr", +1)],
    "momentum":       [("mom_12_1", +1), ("mom_6_1", +1)],
    "short_interest": [("short_pct_float", -1), ("days_to_cover", -1)],
    "insider":        [("insider_net_ratio", +1)],
    "institutional":  [("inst_own_pct", +1)],
    "dcf":            [("mos", +1)],
}

DEFAULT_GROUP_WEIGHTS = {
    "value": 0.20, "quality": 0.20, "growth": 0.125, "momentum": 0.175,
    "dcf": 0.15, "short_interest": 0.05, "insider": 0.05, "institutional": 0.025,
}

def standardize_group(rows, group, method="rank", winsor=(0.02, 0.98),
                      sectors=None, neutralize=False):
    """Standardize every sub-metric of one group across the universe, sign it
    by direction, optionally sector-neutralize, then average the available
    sub-metrics per name (nan-safe). Returns (group_score[N], coverage[N]) where
    coverage is the fraction of the group's sub-metrics that were observable for
    that name — propagated so the caller can flag thin evidence."""
    specs = GROUP_METRICS[group]
    N = len(rows)
    stacked, signs = [], []
    for key, direction in specs:
        raw = np.array([_get(r, key) for r in rows], dtype="float64")
        if np.isfinite(raw).sum() < 2:           # metric unusable in this universe
            continue
        z = standardize(raw, method=method, winsor=winsor) * direction
        if neutralize and sectors is not None:
            z = sector_neutralize(z, sectors)
        stacked.append(z)
        signs.append(direction)
    if not stacked:
        return np.full(N, np.nan), np.zeros(N)
    M = np.vstack(stacked)                         # (k_used, N)
    score = np.nanmean(M, axis=0)
    coverage = np.isfinite(M).mean(axis=0)
    return score, coverage

def composite(group_scores, weights):
    """Weight standardized group scores into one composite, REWEIGHTING over
    the groups actually present for each name (missing groups dropped, surviving
    weights renormalized to sum to 1). Names with zero observable groups -> nan.
    `group_scores`: dict group -> array[N]. `weights`: dict group -> float."""
    groups = [g for g in group_scores if g in weights and weights[g] > 0]
    if not groups:
        N = len(next(iter(group_scores.values())))
        return np.full(N, np.nan)
    M = np.vstack([group_scores[g] for g in groups])      # (G, N)
    w = np.array([weights[g] for g in groups], dtype="float64")[:, None]
    present = np.isfinite(M)
    wmat = np.where(present, w, 0.0)
    wsum = wmat.sum(axis=0)
    num = np.nansum(np.where(present, M, 0.0) * wmat, axis=0)
    out = np.where(wsum > 0, num / np.where(wsum > 0, wsum, 1.0), np.nan)
    return out

def percentile_and_grade(score):
    """Cross-sectional percentile (0-100) of the composite and a letter grade.
    Percentile is rank-based so it is well-defined even if the composite is
    skewed. nan -> (nan, 'NR')."""
    a = _as_float(score)
    m = np.isfinite(a)
    pct = np.full_like(a, np.nan)
    n = int(m.sum())
    if n >= 1:
        order = np.argsort(np.argsort(a[m]))
        pct[m] = (order + 0.5) / n * 100.0
    grades = np.array(["NR"] * len(a), dtype=object)
    bands = [(90, "A+"), (80, "A"), (65, "B"), (45, "C"), (25, "D"), (-1, "F")]
    for i in range(len(a)):
        if not np.isfinite(pct[i]):
            continue
        for thr, g in bands:
            if pct[i] >= thr:
                grades[i] = g
                break
    return pct, grades

def score_universe(rows, weights=None, method="rank", winsor=(0.02, 0.98),
                   neutralize=False, sector_key="sector"):
    """End-to-end. `rows` is a list of per-ticker dicts holding the RAW metric
    keys referenced in GROUP_METRICS (computed upstream in the data layer).
    Returns a dict of arrays aligned to `rows` order:
        per-group scores, per-group coverage, composite (re-standardized),
        percentile, grade.
    The composite is re-standardized with the SAME `method` so it reads on a
    clean cross-sectional scale regardless of how many groups survived."""
    weights = weights or DEFAULT_GROUP_WEIGHTS
    sectors = [r.get(sector_key) for r in rows] if neutralize else None
    out = {"group_score": {}, "group_cov": {}}
    for g in GROUP_METRICS:
        s, cov = standardize_group(rows, g, method=method, winsor=winsor,
                                   sectors=sectors, neutralize=neutralize)
        out["group_score"][g] = s
        out["group_cov"][g] = cov
    comp_raw = composite(out["group_score"], weights)
    comp = standardize(comp_raw, method=method, winsor=winsor)
    pct, grade = percentile_and_grade(comp)
    out["composite_raw"] = comp_raw
    out["composite"] = comp
    out["percentile"] = pct
    out["grade"] = grade
    return out

# --------------------------------------------------------------------------- #
# Long / short selection
# --------------------------------------------------------------------------- #

def select_candidates(tickers, result, n_long=10, n_short=10,
                      min_groups_long=3, min_groups_short=3, group_cov=None):
    """Top names by composite -> longs, bottom -> shorts, but only if enough
    groups were actually observed for that name (thin-evidence names are
    excluded rather than ranked on one lucky factor). Returns (longs, shorts)
    as lists of (ticker, composite, percentile) sorted by conviction."""
    comp = result["composite"]
    pct = result["percentile"]
    if group_cov is None:
        group_cov = result["group_cov"]
    # count groups with a finite score per name
    gmat = np.vstack([result["group_score"][g] for g in GROUP_METRICS])
    n_groups = np.isfinite(gmat).sum(axis=0)
    rec = []
    for i, tk in enumerate(tickers):
        if not np.isfinite(comp[i]):
            continue
        rec.append((tk, float(comp[i]), float(pct[i]), int(n_groups[i])))
    longs = sorted([r for r in rec if r[3] >= min_groups_long],
                   key=lambda r: r[1], reverse=True)[:n_long]
    shorts = sorted([r for r in rec if r[3] >= min_groups_short],
                    key=lambda r: r[1])[:n_short]
    fmt = lambda lst: [(t, c, p) for (t, c, p, _) in lst]
    return fmt(longs), fmt(shorts)

# --------------------------------------------------------------------------- #
# small helpers
# --------------------------------------------------------------------------- #

def _get(row, key):
    v = row.get(key)
    if v is None:
        return np.nan
    try:
        f = float(v)
        return f if np.isfinite(f) else np.nan
    except (TypeError, ValueError):
        return np.nan

def _norm_cdf(x):
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))

def _norm_ppf(p):
    """Acklam's rational approximation to the inverse normal CDF.
    |error| < 1.15e-9 over (0,1). Avoids a scipy dependency in the math core."""
    if p <= 0.0:
        return -np.inf
    if p >= 1.0:
        return np.inf
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = sqrt(-2 * np.log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
               ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    if p > phigh:
        q = sqrt(-2 * np.log(1 - p))
        return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
                ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    q = p - 0.5
    r = q * q
    return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / \
           (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)
