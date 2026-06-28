import dcf_core as c

def approx(a, b, tol=0.01):
    return abs(a - b) <= tol * max(1, abs(b))

# 1) Reproduce the NOW DCF we computed by hand earlier ($292.71, ~70% TV)
res = c.two_stage_dcf(
    fcf0=4533e6, g1=0.25, g2=0.1375, g_term=0.025, r=0.0828,
    cash=3726e6, debt=2403e6, shares=1031e6)
assert res.error is None, res.error
print(f"NOW fair value : ${res.fair:,.2f}   (expected ~292.71)")
print(f"NOW TV fraction: {res.tv_fraction:.1%}  (expected ~70%)")
assert approx(res.fair, 292.71, 0.005)
assert approx(res.tv_fraction, 0.70, 0.03)
print("NOW flags:", res.flags)

# 2) Hard guards must fire, not silently print garbage
assert c.two_stage_dcf(-100e6, .2,.1,.025,.09, 0,0, 1e6).error  # negative base FCF
assert c.two_stage_dcf(100e6, .2,.1,.10,.08, 0,0, 1e6).error    # r <= g_term
assert c.two_stage_dcf(100e6, .2,.1,.025,.09, 0,0, 0).error     # no shares
print("Guards: negative FCF / r<=g / no-shares all correctly refused.")

# 3) robust_cagr vs naive endpoint CAGR on a noisy series
#    NOW-like FCF path (oldest->newest), with a noisy final-year bump.
fcf_path = [1792, 2173, 2704, 3375, 4533]   # $M, 2021->2025
naive = (fcf_path[-1]/fcf_path[0])**(1/(len(fcf_path)-1)) - 1
robust = c.robust_cagr(fcf_path)
print(f"\nFCF CAGR  naive endpoint: {naive:.1%}   robust regression: {robust:.1%}")
# regression should be close but less jumpy; both land in mid-20s here
assert 0.20 < robust < 0.27

# 4) normalize_fcf options
path = [1792, 2173, 2704, 3375, 4533]
print("base latest :", c.normalize_fcf(path, "latest"))
print("base mean3  :", round(c.normalize_fcf(path, "mean", 3)))
print("base median3:", c.normalize_fcf(path, "median", 3))
assert c.normalize_fcf(path, "latest") == 4533

# 5) WACC sanity: NOW-ish, ~all equity
w = c.compute_wacc(beta=0.93, rf=0.045, erp=0.05,
                   market_cap=95.9e9, debt=2.4e9,
                   interest_expense=60e6, tax_rate=0.21)
print(f"\nWACC (beta .93, rf 4.5%, erp 5%): {w:.2%}  (expect ~9%)")
assert 0.085 < w < 0.095

# 6) data_quality_flags catches a stale/pre-split share count
stale = {"price": 93.0, "shares": 207e6, "market_cap": 95.9e9}  # pre-split shares
ok    = {"price": 93.0, "shares": 1031e6, "market_cap": 95.9e9} # post-split
print("\nstale-shares flags:", c.data_quality_flags(stale))
print("clean-shares flags:", c.data_quality_flags(ok))
assert c.data_quality_flags(stale)      # should warn
assert not c.data_quality_flags(ok)     # should be clean

print("\nALL TESTS PASSED")
