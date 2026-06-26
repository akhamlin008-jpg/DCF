"""
test_factor_risk.py — offline correctness proofs (no network). Every test
checks a function against a case whose answer is known in closed form or by
construction. Run: python test_factor_risk.py
"""
import numpy as np
import factor_core as fc
import risk_core as rc

rng = np.random.default_rng(7)
FAILS = []

def check(name, cond, detail=""):
    status = "PASS" if cond else "FAIL"
    print(f"[{status}] {name}" + (f" :: {detail}" if detail else ""))
    if not cond:
        FAILS.append(name)

# --- factor_core ----------------------------------------------------------- #

# rank-to-normal yields ~N(0,1) and is strictly order-preserving
x = rng.uniform(-100, 100, size=5000)
z = fc.rank_to_normal(x)
check("rank_to_normal ~ N(0,1) mean", abs(np.nanmean(z)) < 0.02, f"mean={np.nanmean(z):.4f}")
check("rank_to_normal ~ N(0,1) std", abs(np.nanstd(z) - 1) < 0.02, f"std={np.nanstd(z):.4f}")
check("rank_to_normal order-preserving",
      np.all(np.argsort(x) == np.argsort(z)))

# robust z: identical values -> zeros (no signal), not nan
check("zscore degenerate -> 0", np.allclose(fc.zscore(np.ones(10)), 0.0))

# robust z on a clean symmetric set matches (x-median)/(1.4826*MAD)
v = np.array([1, 2, 3, 4, 5], dtype=float)
zexp = (v - 3) / (1.4826 * 1.0)
check("zscore robust formula", np.allclose(fc.zscore(v), zexp, atol=1e-9))

# composite reweighting: missing a group must renormalize surviving weights
gs = {"value": np.array([1.0, np.nan]), "quality": np.array([0.0, 2.0])}
wts = {"value": 0.5, "quality": 0.5}
comp = fc.composite(gs, wts)
# name0: both present -> 0.5*1 + 0.5*0 = 0.5 ; name1: only quality -> 2.0
check("composite reweights on missing", np.allclose(comp, [0.5, 2.0]))

# winsorize caps the tails
w = fc.winsorize(np.array([-1e9, 0, 1, 2, 1e9]), 0.2, 0.8)
check("winsorize caps extremes", w.max() < 1e9 and w.min() > -1e9)

# --- risk_core: contributions sum to vol (Euler) --------------------------- #
N = 6
A = rng.normal(size=(N, N))
S = A @ A.T / N + np.eye(N) * 0.01           # SPD covariance
w = rng.uniform(0.05, 0.3, size=N); w /= w.sum()
rc_vec, pct = rc.risk_contributions(w, S)
check("risk contributions sum to vol",
      abs(rc_vec.sum() - rc.port_vol(w, S)) < 1e-12,
      f"sum={rc_vec.sum():.6f} vol={rc.port_vol(w,S):.6f}")
check("risk pct sums to 1", abs(pct.sum() - 1) < 1e-12)

# inverse-vol: w_i proportional to 1/sigma_i
wiv = rc.inverse_vol_weights(S)
sig = np.sqrt(np.diag(S))
ratio = wiv * sig
check("inverse-vol proportionality", np.allclose(ratio, ratio[0], atol=1e-9))

# ERC: equal risk contributions (dispersion ~ 0)
werc, conv, iters, disp = rc.erc_weights(S)
check("ERC equalizes risk contributions", disp < 1e-6,
      f"converged={conv} iters={iters} dispersion={disp:.2e}")

# Ledoit-Wolf: delta in [0,1], Sigma* symmetric PSD, shrinks with more data
T_small = rng.normal(size=(40, N))
T_big = rng.normal(size=(4000, N))
Ss, ds = rc.ledoit_wolf_cc(T_small)
Sb, db = rc.ledoit_wolf_cc(T_big)
eig = np.linalg.eigvalsh(Ss)
check("LW delta in [0,1]", 0.0 <= ds <= 1.0, f"delta_small={ds:.3f}")
check("LW Sigma* symmetric", np.allclose(Ss, Ss.T))
check("LW Sigma* PSD", eig.min() > -1e-10, f"min_eig={eig.min():.2e}")
check("LW shrinks with more data", db <= ds + 1e-9,
      f"delta_small={ds:.3f} delta_big={db:.3f}")

# factor_betas: recover planted coefficients
T = 2000
mkt = rng.normal(0, 0.01, T)
dy = rng.normal(0, 0.0005, T)
true_bm, true_br = 1.3, -8.0
y = 0.0001 + true_bm * mkt + true_br * dy + rng.normal(0, 0.002, T)
bm, br, r2 = rc.factor_betas(y, mkt, dy)
check("factor_betas recovers market beta", abs(bm - true_bm) < 0.05, f"bm={bm:.3f}")
check("factor_betas recovers rate beta", abs(br - true_br) < 1.5, f"br={br:.3f}")

# VaR: for normal data historical ~ gaussian; CF > gaussian when left-fat-tailed
norm_r = rng.normal(0, 0.01, 100000)
vh = rc.var_historical(norm_r, 0.95)
vg = rc.var_gaussian(norm_r, 0.95)
check("VaR hist ~ gaussian on normal data", abs(vh - vg) < 0.0005,
      f"hist={vh:.4f} gauss={vg:.4f}")
# negatively skewed, fat-tailed mixture
fat = np.concatenate([rng.normal(0.001, 0.008, 95000),
                      rng.normal(-0.05, 0.03, 5000)])
vcf = rc.var_cornish_fisher(fat, 0.99)
vgf = rc.var_gaussian(fat, 0.99)
check("Cornish-Fisher widens tail vs Gaussian", vcf > vgf,
      f"CF={vcf:.4f} gauss={vgf:.4f}")

# max drawdown on a known path: +10%,+10%, then -50% -> peak 1.21 trough 0.605
path = np.array([0.10, 0.10, -0.50])
mdd, pk, tr = rc.max_drawdown(path)
check("max_drawdown known path", abs(mdd - 0.50) < 1e-9, f"mdd={mdd:.4f}")

# stress factor model linearity
betas_m = np.array([1.0, 1.5]); betas_r = np.array([-5.0, 2.0])
ww = np.array([0.5, 0.5])
dp, contrib = rc.stress_factor_model(ww, betas_m, betas_r, -0.30, -0.01)
expected = 0.5*(1.0*-0.30 + -5.0*-0.01) + 0.5*(1.5*-0.30 + 2.0*-0.01)
check("stress factor model linearity", abs(dp - expected) < 1e-12)

# diversification ratio = 1 for identical perfectly-correlated assets
ones = np.ones((3, 3)) * 0.04
wd = np.array([1/3, 1/3, 1/3])
check("div ratio = 1 when perfectly correlated",
      abs(rc.diversification_ratio(wd, ones) - 1.0) < 1e-9)

# herfindahl on known shares: 50/30/20 -> 0.38, eff_n ~ 2.63
hhi, effn, shares = rc.herfindahl({"a": 50, "b": 30, "c": 20})
check("herfindahl value", abs(hhi - 0.38) < 1e-9, f"hhi={hhi:.4f} eff_n={effn:.3f}")

# liquidity days: $10M position, $50M ADV, 20% participation -> 1.0 day
days = rc.liquidity_days(np.array([10e6]), np.array([50e6]), 0.20)
check("liquidity days", abs(days[0] - 1.0) < 1e-9, f"days={days[0]:.3f}")

print("\n" + ("ALL PASS" if not FAILS else f"FAILURES: {FAILS}"))
