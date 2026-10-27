"""
15_check_comparison.py -- Stage 6 checks: information criteria, forecast
scores, the stiff ODE solver and the rolling-origin forecasts.

Run:  python src\\15_check_comparison.py      (about two minutes)
Every check prints PASS or FAIL with the numbers behind it.
"""

import importlib
import math
import sys
import warnings

import numpy as np
import pandas as pd
from scipy.stats import nbinom

import config
import fitting as F
import outbreaks as O
import scoring as SC
import synthetic as Y

warnings.simplefilter("ignore")
fit16 = importlib.import_module("16_compare_forecast")
results = []


def check(name, ok, detail):
    results.append(ok)
    print(f"[{'PASS' if ok else 'FAIL'}] {len(results)}. {name}  ({detail})")


# ---------------------------------------------------------------------------
# A. Information criteria and scores
# ---------------------------------------------------------------------------
w = SC.akaike_weights([100.0, 102.0, 110.0])
check("AIC, AICc and Akaike weights",
      SC.aic(-50, 4) == 108 and abs(SC.aicc(-50, 4, 34) - (108 + 40 / 29)) < 1e-12
      and abs(w.sum() - 1) < 1e-12 and abs(w[0] / w[1] - math.e) < 1e-12,
      f"AIC 108, AICc {SC.aicc(-50, 4, 34):.3f}, weights {np.round(w, 4).tolist()}")

mu, kd = np.array([3.0, 40.0, 900.0]), 2.5
y = np.array([0, 55, 700])
ls_ref = [nbinom.logpmf(yy, kd, kd / (kd + m)) for yy, m in zip(y, mu)]
check("negative binomial log score equals scipy's log pmf with mean mu, variance mu + mu^2/k",
      np.allclose(SC.nb_logscore(y, mu, kd), ls_ref),
      f"{np.round(SC.nb_logscore(y, mu, kd), 4).tolist()}")

rng = np.random.default_rng(3)
crps = SC.nb_crps(y, mu, kd)
mc = []
for yy, m in zip(y, mu):
    x = rng.negative_binomial(kd, kd / (kd + m), size=(2, 400000))
    mc.append(np.mean(np.abs(x[0] - yy)) - 0.5 * np.mean(np.abs(x[0] - x[1])))
check("CRPS sum formula equals E|X - y| - E|X - X'|/2 (Monte Carlo)",
      np.allclose(crps, mc, rtol=0.01, atol=0.02),
      f"formula {np.round(crps, 3).tolist()}, Monte Carlo {np.round(mc, 3).tolist()}")

mu_true, kd = 60.0, 3.0
ys = rng.negative_binomial(kd, kd / (kd + mu_true), size=20000)
cov = {lev: np.mean([(lo <= v <= hi) for v in ys]) for lev in (0.5, 0.9)
       for lo, hi in [SC.nb_interval(mu_true, kd, lev)]}
check("central forecast intervals cover at least their level (discrete counts)",
      0.5 <= cov[0.5] < 0.56 and 0.9 <= cov[0.9] < 0.93,
      f"50%: {cov[0.5]:.3f}, 90%: {cov[0.9]:.3f}")

sc_right = np.mean(SC.nb_logscore(ys[:5000], np.full(5000, mu_true), kd))
sc_wrong = np.mean(SC.nb_logscore(ys[:5000], np.full(5000, 1.3 * mu_true), kd))
crps_right = np.mean(SC.nb_crps(ys[:1000], np.full(1000, mu_true), kd))
crps_wrong = np.mean(SC.nb_crps(ys[:1000], np.full(1000, 1.3 * mu_true), kd))
check("both scores prefer the true forecast distribution on average (proper scores)",
      sc_right > sc_wrong and crps_right < crps_wrong,
      f"log score {sc_right:.3f} vs {sc_wrong:.3f}; CRPS {crps_right:.2f} vs {crps_wrong:.2f}")

# ---------------------------------------------------------------------------
# B. Solvers and fits
# ---------------------------------------------------------------------------
s, g = O.rates("mpox")
cases = {"seir": {"beta": 3.0, "e0": 1e-4}, "gfree": {"beta": 1.5, "e0": 1e-4, "g": 0.4},
         "decline": {"beta": 4.0, "e0": 1e-4, "rho": 0.3, "kappa": 0.2},
         "hetero": {"beta": 80.0, "e0": 1e-4, "q": 0.0026, "c": 1e4}}
errs = {}
for m, p in cases.items():
    a = F.alt_weekly(m, p, 33, s, g, "DOP853")
    b = F.alt_weekly(m, p, 33, s, g, "LSODA")
    errs[m] = np.max(np.abs(a - b)) / np.max(a)
check("LSODA (Stage 6) and DOP853 (Stage 5) give the same weekly increments",
      max(errs.values()) < 1e-6, ", ".join(f"{m} {e:.1e}" for m, e in errs.items()))

y = O.load_series("mpox-US")["y"]
a5 = pd.read_csv(config.RESULTS_DIR / "stage5_alt_fits.csv")
ll_new = F.AltFitter(y, s, g, "hetero", ode_method="LSODA").fit()["loglik"]
ll_old = a5[(a5.series == "mpox-US") & (a5.model == "hetero")].loglik.iloc[0]
check("refitting with LSODA reproduces the Stage 5 heterogeneous-recovery fit to mpox",
      abs(ll_new - ll_old) < 0.01, f"{ll_new:.3f} vs {ll_old:.3f}")

# ---------------------------------------------------------------------------
# C. Rolling-origin forecasts
# ---------------------------------------------------------------------------
y = O.load_series("measles-TX")["y"]
mu1, kd1, _, _ = fit16.fit_forecast(y[:12], "measles-TX", "seir", 4)
y_changed = y.copy()
y_changed[12:] = 999.0                       # change every week AFTER the origin
mu2, kd2, _, _ = fit16.fit_forecast(y_changed[:12], "measles-TX", "seir", 4)
r1 = fit16.forecast_task(("measles-TX", "seir", 12))
check("no leakage: a forecast does not change when the weeks after the origin change",
      np.allclose(mu1, mu2) and kd1 == kd2 and np.allclose([r["mean"] for r in r1], mu1),
      f"means {np.round(mu1, 2).tolist()}")

fitter = F.AltFitter(y[:12], *O.rates("measles"), "seir", ode_method="LSODA")
out = fitter.fit()
in_sample = fitter.fitted_mean(np.array(out["z"]), out["A"])
m_long = F.alt_weekly("seir", F.alt_params("seir", np.array(out["z"])), 16, *O.rates("measles"), "LSODA")
check("the forecast continues the fitted curve: the first 12 weeks equal the in-sample fit",
      np.allclose(out["A"] * np.maximum(m_long[:12], F.M_FLOOR), in_sample, rtol=1e-6),
      f"week 12 fitted {in_sample[-1]:.2f}, continued {out['A'] * m_long[11]:.2f}, "
      f"forecast week 13 {out['A'] * m_long[12]:.2f}")

orig = {s: fit16.origins(s) for s in O.SHORT_NAMES}
n_or = sum(len(v) for v in orig.values())
check("forecast origins every 2 weeks from week 8, leaving at least one week to forecast",
      all(v[0] == 8 and v[-1] < len(O.load_series(s)["y"]) and np.all(np.diff(v) == 2)
          for s, v in orig.items()),
      f"{n_or} origins; mpox-US {orig['mpox-US'][0]}..{orig['mpox-US'][-1]}")

# ---------------------------------------------------------------------------
# D. Synthetic data: the memoryless SEIR fit equals the Stage 4 Caputo fit at alpha = 1
# ---------------------------------------------------------------------------
d = pd.read_csv(config.PROJECT_DIR / "data" / "synthetic" / "replicates.csv")
cap = pd.read_csv(config.RESULTS_DIR / "stage4_replicate_fits.csv").drop_duplicates(["truth", "k_noise", "rep"])
diffs = []
for t in ("T0", "T2"):
    ys = d[(d.truth == t) & (d.k_noise == 10) & (d["rep"] == 0)].sort_values("week").cases.to_numpy(float)
    ll = F.AltFitter(ys, Y.SIGMA, Y.GAMMA, "seir", ode_method="LSODA").fit()["loglik"]
    c = cap[(cap.truth == t) & (cap.k_noise == 10) & (cap["rep"] == 0)].loglik_alpha1.iloc[0]
    diffs.append(ll - c)
check("on synthetic data, the ODE SEIR fit matches the Stage 4 Caputo fit at alpha = 1",
      all(x > -0.05 for x in diffs),
      f"differences {np.round(diffs, 3).tolist()} (the ODE fit may be slightly better, never worse)")

print()
n_fail = results.count(False)
print("All %d checks passed." % len(results) if n_fail == 0 else f"{n_fail} check(s) FAILED.")
sys.exit(1 if n_fail else 0)
