"""
risk_core.py — pure portfolio-risk math. numpy only; no streamlit / yfinance /
network, so every function is unit-testable offline against synthetic inputs
whose answer is known in closed form (see test_factor_risk.py).

Conventions
-----------
* R is a (T, N) matrix of PERIODIC simple returns, columns aligned to `tickers`.
* w is a length-N weight vector. Long book: w >= 0, sum 1. Long/short: signed.
* Covariances are returned at the INPUT frequency; annualize() scales by the
  periods-per-year you pass (252 daily, 52 weekly, 12 monthly).
* Nothing here fabricates a number. Where an estimate rests on an assumption
  (a shock size, a distribution), the assumption is an explicit argument.
"""
from __future__ import annotations

import numpy as np

SQRT = np.sqrt

# --------------------------------------------------------------------------- #
# Covariance / correlation
# --------------------------------------------------------------------------- #

def sample_cov(R, ddof=1):
    R = np.asarray(R, dtype="float64")
    return np.cov(R, rowvar=False, ddof=ddof)

def cov_to_corr(S):
    d = SQRT(np.diag(S))
    outer = np.outer(d, d)
    with np.errstate(divide="ignore", invalid="ignore"):
        C = np.where(outer > 0, S / outer, 0.0)
    np.fill_diagonal(C, 1.0)
    return C

def ledoit_wolf_cc(R):
    """Ledoit & Wolf (2004) shrinkage toward the CONSTANT-CORRELATION target.

    Sample covariance is unbiased but noisy; with N positions and limited T its
    smallest eigenvalues are badly underestimated, which makes any optimizer
    (risk parity, min-var) chase phantom arbitrage. Shrinkage pulls S toward a
    structured target F (every pair shares the average correlation) by the
    analytically optimal intensity delta in [0,1]. This is the estimator, not a
    hand-tuned fudge.

    Returns (Sigma_star, delta). Implements the closed form for pi, rho, gamma
    from the paper (honor-code: 1/T scaling on demeaned returns).
    """
    X = np.asarray(R, dtype="float64")
    T, N = X.shape
    Xc = X - X.mean(axis=0, keepdims=True)
    S = (Xc.T @ Xc) / T                              # 1/T as in LW derivation
    var = np.diag(S).copy()
    std = SQRT(var)
    outer_std = np.outer(std, std)
    with np.errstate(divide="ignore", invalid="ignore"):
        Corr = np.where(outer_std > 0, S / outer_std, 0.0)
    # average off-diagonal correlation
    iu = np.triu_indices(N, k=1)
    rbar = Corr[iu].mean() if N > 1 else 0.0
    F = rbar * outer_std
    np.fill_diagonal(F, var)

    # pi : sum of asymptotic variances of the entries of sqrt(T)*S
    Y = Xc ** 2
    pi_mat = (Y.T @ Y) / T - S ** 2
    pi_hat = pi_mat.sum()

    # rho : sum of asymptotic covariances of F's entries with S's entries
    # diagonal part
    rho_diag = np.diag(pi_mat).sum()
    # off-diagonal part (constant-correlation target term)
    term = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            # theta_ii,ij and theta_jj,ij
            t_iiij = ((Xc[:, i] ** 2) * (Xc[:, i] * Xc[:, j])).mean() - var[i] * S[i, j]
            t_jjij = ((Xc[:, j] ** 2) * (Xc[:, i] * Xc[:, j])).mean() - var[j] * S[i, j]
            if std[i] > 0 and std[j] > 0:
                term[i, j] = (std[j] / std[i]) * t_iiij + (std[i] / std[j]) * t_jjij
    rho_off = (rbar / 2.0) * term.sum()
    rho_hat = rho_diag + rho_off

    # gamma : squared Frobenius distance between S and F
    gamma_hat = ((S - F) ** 2).sum()

    if gamma_hat <= 0:
        delta = 0.0
    else:
        kappa = (pi_hat - rho_hat) / gamma_hat
        delta = max(0.0, min(1.0, kappa / T))
    Sigma = delta * F + (1.0 - delta) * S
    # symmetrize against fp drift
    Sigma = 0.5 * (Sigma + Sigma.T)
    return Sigma, float(delta)

def annualize_cov(S, periods=252):
    return np.asarray(S) * periods

# --------------------------------------------------------------------------- #
# Portfolio risk + contributions
# --------------------------------------------------------------------------- #

def port_vol(w, S):
    w = np.asarray(w, dtype="float64")
    return float(SQRT(w @ S @ w))

def risk_contributions(w, S):
    """Euler decomposition of portfolio volatility.
        MCR_i = (S w)_i / sigma_p           (marginal contribution to risk)
        RC_i  = w_i * MCR_i                  (component contribution)
        sum_i RC_i == sigma_p                (exact, by Euler's theorem)
    Returns (rc[N], pct[N]). pct sums to 1 and is the rigorous "how much of
    portfolio risk does this single name actually carry" — which is NOT the
    same as its weight once correlations bite."""
    w = np.asarray(w, dtype="float64")
    sig = port_vol(w, S)
    if sig <= 0:
        return np.zeros_like(w), np.zeros_like(w)
    mcr = (S @ w) / sig
    rc = w * mcr
    return rc, rc / sig

def diversification_ratio(w, S):
    """(sum_i |w_i| sigma_i) / sigma_p. =1 means no diversification (perfectly
    correlated); higher means the book's vol is below the weighted average of
    its parts. A compact one-number read on diversification quality."""
    w = np.asarray(w, dtype="float64")
    sig_i = SQRT(np.diag(S))
    num = np.sum(np.abs(w) * sig_i)
    den = port_vol(w, S)
    return float(num / den) if den > 0 else np.nan

# --------------------------------------------------------------------------- #
# Position sizing
# --------------------------------------------------------------------------- #

def inverse_vol_weights(S):
    """w_i proportional to 1/sigma_i, normalized. Closed-form, always valid,
    ignores correlation. A clean risk-based default."""
    sig = SQRT(np.diag(S))
    inv = np.where(sig > 0, 1.0 / sig, 0.0)
    tot = inv.sum()
    return inv / tot if tot > 0 else np.full_like(sig, 1.0 / len(sig))

def erc_weights(S, budget=None, max_iter=200, tol=1e-10):
    """Equal-Risk-Contribution (a.k.a. risk parity) weights via Newton on the
    convex problem
        min_{w>0}  0.5 w'Sw  -  sum_i b_i log(w_i)
    whose stationarity (S w)_i = b_i / w_i forces RC_i proportional to b_i.
    Then renormalize to sum 1. Default budget b = equal. Returns
    (w, converged, iters, max_rc_dispersion). Correlation-aware, unlike
    inverse-vol. Falls back to inverse-vol if Newton stalls."""
    S = np.asarray(S, dtype="float64")
    N = S.shape[0]
    b = np.full(N, 1.0 / N) if budget is None else np.asarray(budget, dtype="float64")
    b = b / b.sum()
    w = inverse_vol_weights(S).copy()
    w = np.maximum(w, 1e-8)
    converged = False
    it = 0
    for it in range(1, max_iter + 1):
        Sw = S @ w
        grad = Sw - b / w
        H = S + np.diag(b / w**2)
        try:
            step = np.linalg.solve(H, grad)
        except np.linalg.LinAlgError:
            break
        # backtracking to keep w strictly positive
        alpha = 1.0
        while alpha > 1e-12:
            w_new = w - alpha * step
            if np.all(w_new > 0):
                break
            alpha *= 0.5
        if alpha <= 1e-12:
            break
        w = w_new
        if np.linalg.norm(grad, ord=np.inf) < tol:
            converged = True
            break
    w = np.maximum(w, 0.0)
    w = w / w.sum() if w.sum() > 0 else inverse_vol_weights(S)
    _, pct = risk_contributions(w, S)
    disp = float(pct.max() - pct.min()) if len(pct) else np.nan
    if not converged and disp > 0.05:                # honest fallback
        w = inverse_vol_weights(S)
        _, pct = risk_contributions(w, S)
        disp = float(pct.max() - pct.min())
    return w, converged, it, disp

def vol_target_scale(w, S, target_vol, periods=252):
    """Gross leverage multiplier that scales the book's annualized vol to
    `target_vol`. >1 means lever up to hit target; <1 means hold cash."""
    ann = port_vol(w, S) * SQRT(periods)
    return float(target_vol / ann) if ann > 0 else np.nan

# --------------------------------------------------------------------------- #
# Factor exposures: market beta + interest-rate beta
# --------------------------------------------------------------------------- #

def ols(y, X):
    """OLS with intercept handled by caller. Returns (coef, resid_std, r2)."""
    y = np.asarray(y, dtype="float64")
    X = np.asarray(X, dtype="float64")
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ coef
    dof = max(len(y) - X.shape[1], 1)
    rss = float(resid @ resid)
    tss = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - rss / tss if tss > 0 else np.nan
    return coef, SQRT(rss / dof), r2

def factor_betas(stock_rets, mkt_rets, dyield):
    """Two-factor regression per name:
        r_i = a + beta_mkt * r_mkt + beta_rate * d(10y yield) + eps
    beta_rate is the empirical return response to a +1.0 (i.e. +100 percentage-
    point) move in the 10y yield over the sample frequency; multiply by 0.01 for
    a +1bp read, or report per +100bp by leaving as-is and labeling. Returns
    (beta_mkt, beta_rate, r2). Empirical and regime-dependent — state both."""
    y = np.asarray(stock_rets, dtype="float64")
    X = np.column_stack([np.ones_like(y), mkt_rets, dyield])
    coef, _, r2 = ols(y, X)
    return float(coef[1]), float(coef[2]), float(r2)

# --------------------------------------------------------------------------- #
# Tail risk: VaR / CVaR (historical, Gaussian, Cornish-Fisher)
# --------------------------------------------------------------------------- #

def _z_alpha(alpha):
    # left-tail quantile of N(0,1) at prob (1-alpha); alpha is confidence e.g .95
    from math import sqrt as _s
    # inverse-normal via Acklam (reuse a compact copy)
    p = 1.0 - alpha
    return _acklam(p)

def var_historical(rets, alpha=0.95):
    """Empirical VaR: the (1-alpha) quantile of realized losses. No distribution
    assumed. Reported as a POSITIVE loss fraction."""
    r = np.asarray(rets, dtype="float64")
    q = np.nanquantile(r, 1.0 - alpha)
    return float(-q)

def cvar_historical(rets, alpha=0.95):
    """Expected shortfall: mean loss in the worst (1-alpha) tail. Positive."""
    r = np.asarray(rets, dtype="float64")
    thr = np.nanquantile(r, 1.0 - alpha)
    tail = r[r <= thr]
    return float(-tail.mean()) if tail.size else np.nan

def var_gaussian(rets, alpha=0.95):
    r = np.asarray(rets, dtype="float64")
    mu, sd = np.nanmean(r), np.nanstd(r, ddof=1)
    return float(-(mu + _z_alpha(alpha) * sd))

def var_cornish_fisher(rets, alpha=0.95):
    """Modified VaR that corrects the Gaussian quantile for skewness (S) and
    excess kurtosis (K) via the Cornish-Fisher expansion:
        z* = z + (z^2-1)S/6 + (z^3-3z)K/24 - (2z^3-5z)S^2/36
    Fat left tails (negative skew, high kurtosis) push VaR out beyond the naive
    Gaussian number, which matters precisely in the scenarios you care about."""
    r = np.asarray(rets, dtype="float64")
    mu, sd = np.nanmean(r), np.nanstd(r, ddof=1)
    n = np.isfinite(r).sum()
    rc = r[np.isfinite(r)]
    if n < 4 or sd == 0:
        return var_gaussian(rets, alpha)
    S = float(((rc - mu) ** 3).mean() / sd ** 3)
    K = float(((rc - mu) ** 4).mean() / sd ** 4 - 3.0)
    z = _z_alpha(alpha)
    zcf = (z + (z**2 - 1) * S / 6.0 + (z**3 - 3*z) * K / 24.0
           - (2*z**3 - 5*z) * S**2 / 36.0)
    return float(-(mu + zcf * sd))

def tail_prob(rets, loss_threshold):
    """Distributional probability of a single-period loss worse than
    `loss_threshold` (a positive fraction), estimated two ways:
    empirical frequency and Gaussian. Returned as a dict. This is a tail
    probability UNDER THE SAMPLE/ASSUMPTION — not a macro forecast."""
    r = np.asarray(rets, dtype="float64")
    r = r[np.isfinite(r)]
    emp = float((r <= -loss_threshold).mean()) if r.size else np.nan
    mu, sd = r.mean(), r.std(ddof=1)
    gauss = float(_norm_cdf((-loss_threshold - mu) / sd)) if sd > 0 else np.nan
    return {"empirical": emp, "gaussian": gauss}

# --------------------------------------------------------------------------- #
# Drawdown + factor-model stress
# --------------------------------------------------------------------------- #

def max_drawdown(rets):
    """Worst peak-to-trough on the cumulative-return path of the series.
    Returns (mdd_positive, peak_idx, trough_idx)."""
    r = np.asarray(rets, dtype="float64")
    cum = np.cumprod(1.0 + np.nan_to_num(r))
    peak = np.maximum.accumulate(cum)
    dd = cum / peak - 1.0
    trough = int(np.argmin(dd))
    peak_idx = int(np.argmax(cum[:trough + 1])) if trough > 0 else 0
    return float(-dd.min()), peak_idx, trough

def stress_factor_model(w, betas_mkt, betas_rate, mkt_shock, rate_shock):
    """First-order P&L of the book under a named macro scenario:
        dP ~ sum_i w_i ( beta_mkt,i * mkt_shock + beta_rate,i * rate_shock )
    e.g. mkt_shock=-0.35 (equities -35%), rate_shock=-0.015 (10y -150bp). This
    is a LINEAR, idiosyncratic-zero approximation — it captures factor exposure,
    not convexity or correlation breakdown, and should be read as a directional
    estimate beside the historical drawdown, not a guarantee."""
    w = np.asarray(w, dtype="float64")
    bm = np.asarray(betas_mkt, dtype="float64")
    br = np.asarray(betas_rate, dtype="float64")
    contrib = w * (bm * mkt_shock + br * rate_shock)
    return float(contrib.sum()), contrib

# --------------------------------------------------------------------------- #
# Liquidity, concentration, correlation summaries
# --------------------------------------------------------------------------- #

def liquidity_days(position_dollars, adv_dollars, participation=0.20):
    """Days to exit at `participation` of average daily $ volume. Standard
    desk rule of thumb: never be more than ~20% of a day's volume."""
    pos = np.asarray(position_dollars, dtype="float64")
    adv = np.asarray(adv_dollars, dtype="float64")
    cap = participation * adv
    with np.errstate(divide="ignore", invalid="ignore"):
        days = np.where(cap > 0, pos / cap, np.inf)
    return days

def liquidity_rating(days):
    d = np.asarray(days, dtype="float64")
    out = np.empty(d.shape, dtype=object)
    for i, x in np.ndenumerate(d):
        out[i] = ("High" if x <= 0.5 else "Good" if x <= 1 else
                  "Moderate" if x <= 3 else "Low" if x <= 7 else "Illiquid")
    return out

def herfindahl(weights_by_group):
    """HHI = sum s_k^2 over group weight shares; effective number = 1/HHI.
    Pass a dict group -> summed |weight|. Returns (hhi, eff_n, shares_dict)."""
    tot = sum(abs(v) for v in weights_by_group.values())
    if tot <= 0:
        return np.nan, np.nan, {}
    shares = {k: abs(v) / tot for k, v in weights_by_group.items()}
    hhi = sum(s**2 for s in shares.values())
    return float(hhi), float(1.0 / hhi), shares

def avg_pairwise_corr(C):
    N = C.shape[0]
    if N < 2:
        return np.nan
    iu = np.triu_indices(N, k=1)
    return float(C[iu].mean())

def most_correlated_pairs(C, tickers, k=5):
    N = C.shape[0]
    pairs = []
    for i in range(N):
        for j in range(i + 1, N):
            pairs.append((tickers[i], tickers[j], float(C[i, j])))
    pairs.sort(key=lambda p: abs(p[2]), reverse=True)
    return pairs[:k]

# --------------------------------------------------------------------------- #
# inverse-normal + normal cdf (compact, scipy-free)
# --------------------------------------------------------------------------- #

def _norm_cdf(x):
    from math import erf, sqrt
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))

def _acklam(p):
    from math import sqrt, log
    if p <= 0:
        return -np.inf
    if p >= 1:
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
        q = sqrt(-2 * log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
               ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    if p > phigh:
        q = sqrt(-2 * log(1 - p))
        return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
                ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    q = p - 0.5
    r = q * q
    return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / \
           (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)
