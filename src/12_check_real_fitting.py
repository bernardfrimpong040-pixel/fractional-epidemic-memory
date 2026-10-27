"""
12_check_real_fitting.py -- Stage 5 checks: the new model (AHM-SEIR), its
solver, the memoryless alternatives' fitter and the real-data handling.

Run:  python src\\12_check_real_fitting.py      (about two minutes)
Every check prints PASS or FAIL with the numbers behind it.
"""

import math
import sys
import warnings

import numpy as np
import pandas as pd
from scipy.integrate import quad
from scipy.stats import kstest

import fitting as F
import models as M
import outbreaks as O
import solvers as S

warnings.simplefilter("ignore")
results = []


def check(name, ok, detail):
    results.append(ok)
    print(f"[{'PASS' if ok else 'FAIL'}] {len(results)}. {name}  ({detail})")


def rel_err(a, b):
    return float(np.max(np.abs(np.asarray(a) - np.asarray(b))) / np.max(np.abs(b)))


# ---------------------------------------------------------------------------
# A. Data and disease parameters
# ---------------------------------------------------------------------------
gi = {d: v["latent_days"] + v["infectious_days"] for d, v in O.DISEASES.items()}
check("mean generation interval = 1/sigma + 1/gamma matches the cited serial intervals",
      abs(gi["mpox"] - 8.5) < 1e-9 and abs(gi["ebola"] - 15.3) < 1e-9,
      f"mpox {gi['mpox']:.1f} d (8.5), Ebola {gi['ebola']:.1f} d (15.3), measles {gi['measles']:.0f} d")

summ = pd.read_csv(O.config.RESULTS_DIR / "stage1_data_summary.csv")
lens, ok, worst = {}, True, 0.0
raw = pd.read_csv(O.config.PROCESSED_DIR / "all_series_weekly.csv")
for disease, name, short in O.SERIES:
    d = O.load_series(short)
    row = summ[(summ.dataset == disease) & (summ.series == name)].iloc[0]
    lens[short] = len(d["y"])
    ok &= len(d["y"]) == row.n_weeks - row.takeoff_week
    ok &= bool(np.all(d["y"] >= 0) and np.all(d["y"] == np.round(d["y"])))
    r = raw[(raw.dataset == disease) & (raw.series == name) & (raw.week >= row.takeoff_week)]
    worst = max(worst, abs(d["y"].sum() - r.cases.sum()) / len(r))
    ok &= worst <= 0.5
check("series start at take-off, counts are whole and non-negative, rounding is small",
      ok, f"weeks {lens}; largest rounding change of the total = {worst:.2f} per week")

d = O.load_series("measles-SC", mask_flagged=True)
check("masking sets exactly the flagged weeks to NaN",
      np.array_equal(np.isnan(d["y"]), d["flag"]) and d["flag"].sum() == 2,
      f"{int(d['flag'].sum())} flagged weeks masked in measles-SC")

# ---------------------------------------------------------------------------
# B. The AHM-SEIR model and its solver
# ---------------------------------------------------------------------------
sigma, gamma = O.rates("mpox")
K = 30
beta = 2.0 * gamma
_, ode = S.simulate_erlang_seir(beta, sigma, gamma, 1.0, 1, 1, 1e-4, K, np.arange(K + 1.0))
_, ahm = S.solve_ahm_seir(beta, sigma, gamma, 1.0, 1e-4, K, 1.0, 64)
e = rel_err(np.diff(ahm["cumulative"]), np.diff(ode["cumulative"]))
check("AHM-SEIR at alpha = 1 equals the ordinary SEIR (independent ODE solver)",
      e < 1e-3, f"largest weekly difference {e:.1e} of the peak")

tau, gc = M.ahm_seir_tau_gc(gamma)
means = [quad(lambda t: S.ahm_survival(np.array([t]), tau, a, gc)[0], 0, np.inf, limit=400)[0] * gamma
         for a in (0.4, 0.7, 0.95)]
check("mean infectious period = 1/gamma for every alpha (tau = 2/gamma, gamma_c = gamma/2)",
      max(abs(m - 1) for m in means) < 1e-6, f"gamma x mean = {np.round(means, 8).tolist()}")

errs = {}
for a in (0.4, 0.9):
    b = M.ahm_seir_beta_for_growth_rate(1.0, sigma, gamma, a)
    ref = np.diff(S.solve_ahm_seir(b, sigma, gamma, a, 1e-3, K, 1.0, 256)[1]["cumulative"])
    e1 = rel_err(np.diff(S.solve_ahm_seir(b, sigma, gamma, a, 1e-3, K, 1.0, 16)[1]["cumulative"]), ref)
    e2 = rel_err(np.diff(S.solve_ahm_seir(b, sigma, gamma, a, 1e-3, K, 1.0, 32)[1]["cumulative"]), ref)
    errs[a] = (e1, e2, math.log2(e1 / e2))
check("AHM-SEIR solver converges with order 2 at small and large alpha (product integration)",
      all(1.8 < v[2] < 2.2 for v in errs.values()),
      "; ".join(f"alpha {a}: errors {v[0]:.1e} -> {v[1]:.1e}, order {v[2]:.2f}" for a, v in errs.items()))

W, B = S._ahm_seir_weights(float(gamma), 0.6, 1.0 / 32, 320)
n = 320
exact = quad(lambda t: S.ahm_survival(np.array([t]), tau, 0.6, gc)[0], 0, n / 32, limit=400, points=[1e-6, 1e-3])[0]
approx = B[n] + W[1:n].sum() + W[0]
check("product-integration weights integrate a constant exactly (sum = int_0^t Phi)",
      abs(approx - exact) / exact < 1e-8, f"{approx:.12f} vs {exact:.12f}")

a, r_true = 0.6, 0.5
b = M.ahm_seir_beta_for_growth_rate(r_true, sigma, gamma, a)
r_back = M.ahm_seir_growth_rate(b, sigma, gamma, a)
# tiny seed and a late window: the heavy-tailed infectious period makes the
# approach to steady exponential growth slow, and S must still be close to 1
_, o = S.solve_ahm_seir(b, sigma, gamma, a, 1e-12, 41, 1.0, 32)
inc = np.diff(o["cumulative"])
r_sim = math.log(inc[40] / inc[35]) / 5.0
check("AHM-SEIR growth rate: Euler-Lotka formula, its inverse, and the simulated growth agree",
      abs(r_back - r_true) < 1e-10 and abs(r_sim - r_true) < 2e-3,
      f"r = {r_true}, inverse {r_back:.12f}, simulated {r_sim:.4f}")

b = 1.8 * gamma
_, o = S.solve_ahm_seir(b, sigma, gamma, 0.5, 1e-4, 400, 1.0, 16)
s_inf, s0 = o["S"][-1], 1 - 1e-4
lhs, rhs = math.log(s0 / s_inf), (b / gamma) * (1 - s_inf)
check("final-size relation ln(S0/S_inf) = R0 (1 - S_inf) with R0 = beta/gamma (alpha = 0.5)",
      abs(lhs - rhs) / rhs < 2e-3, f"{lhs:.5f} vs {rhs:.5f}")

# ---------------------------------------------------------------------------
# C. Fitting code
# ---------------------------------------------------------------------------
y = O.load_series("measles-SC")["y"]
sg = O.rates("measles")
tab = [{"alpha": 0.7, "method": "pece", "k": 8}]
f_all = F.ProfileFitter(np.where(O.load_series("measles-SC")["flag"], np.nan, y),
                        {"sigma": sg[0], "gamma": sg[1], "substeps": tab})
ll_mask, extra = f_all.loglik(0.7, math.log(3.0), math.log(1e-4))
keep = ~O.load_series("measles-SC")["flag"]
ll_drop = F.best_scale_dispersion(y[keep], extra[2][keep])[0]
check("a masked (NaN) week is simply left out of the likelihood",
      abs(ll_mask - ll_drop) < 1e-8, f"{ll_mask:.6f} vs {ll_drop:.6f}")

gs, qs = F.hetero_classes(gamma, 0.2, 9.0)
check("heterogeneous recovery keeps the mean infectious period at 1/gamma",
      abs(np.sum(qs / gs) - 1 / gamma) < 1e-12 and np.allclose(F.hetero_classes(gamma, 0.3, 1.0)[0], gamma),
      f"mean {np.sum(qs / gs):.6f}, 1/gamma {1 / gamma:.6f}; c = 1 gives rate gamma for both classes")

y = O.load_series("mpox-US")["y"]
sg = O.rates("mpox")
seir = F.AltFitter(y, *sg, "seir").fit()
cap = F.ProfileFitter(y, {"sigma": sg[0], "gamma": sg[1],
                          "substeps": [{"alpha": 1.0, "method": "pece", "k": 64}]})
ll_c, x_c = cap.optimise(1.0, np.array(seir["z"]))
check("ordinary SEIR fit (ODE solver) = Caputo fit at alpha = 1 (PECE solver), mpox data",
      abs(seir["loglik"] - ll_c) < 0.05, f"{seir['loglik']:.3f} vs {ll_c:.3f}")

# synthetic data from the declining-transmission model: the fitter must find
# a likelihood at least as high as that of the true parameters
sg = O.rates("ebola")
true_p = {"beta": 2.6 * sg[1], "e0": 1e-4, "rho": 0.3, "kappa": 0.15}
m_true = F.alt_weekly("decline", true_p, 40, *sg)
mu = 3000 * m_true / m_true.max()
rng = np.random.default_rng(7)
ysim = rng.negative_binomial(50, 50 / (50 + mu)).astype(float)
fd = F.AltFitter(ysim, *sg, "decline")
z_true = [math.log(true_p["beta"]), math.log(true_p["e0"]), math.log(0.3 / 0.7), math.log(0.15)]
ll_true = fd.loglik(z_true)[0]
fit_d = fd.fit()
check("declining-transmission fitter reaches at least the likelihood of the truth",
      fit_d["loglik"] >= ll_true - 0.01 and abs(fit_d["rho"] - 0.3) < 0.1,
      f"fit {fit_d['loglik']:.2f} >= truth {ll_true:.2f}; rho {fit_d['rho']:.3f} (0.3), "
      f"kappa {fit_d['kappa']:.3f} (0.15)")

true_h = {"beta": 2.0 * sg[1], "e0": 1e-4, "q": 0.2, "c": 9.0}
m_true = F.alt_weekly("hetero", true_h, 40, *sg)
mu = 3000 * m_true / m_true.max()
ysim = rng.negative_binomial(50, 50 / (50 + mu)).astype(float)
fh = F.AltFitter(ysim, *sg, "hetero")
z_true = [math.log(true_h["beta"]), math.log(1e-4), math.log(0.2 / 0.8), math.log(8.0)]
ll_true = fh.loglik(z_true)[0]
fit_h = fh.fit()
check("heterogeneous-recovery fitter reaches at least the likelihood of the truth",
      fit_h["loglik"] >= ll_true - 0.01,
      f"fit {fit_h['loglik']:.2f} >= truth {ll_true:.2f}; q {fit_h['q']:.3f} (0.2), c {fit_h['c']:.2f} (9)")

r_formula = F.alt_growth_rate("hetero", true_h, *sg)
b_back = M.beta_for_growth_rate_hetero(r_formula, sg[0], *F.hetero_classes(sg[1], 0.2, 9.0))
check("growth rate of the heterogeneous model inverts Stage 4's beta_for_growth_rate_hetero",
      abs(b_back - true_h["beta"]) < 1e-9, f"beta {true_h['beta']:.6f} -> r {r_formula:.4f} -> {b_back:.6f}")

# ---------------------------------------------------------------------------
# D. Bootstrap and residual tools
# ---------------------------------------------------------------------------
sys.path.insert(0, str(O.config.PROJECT_DIR / "src"))
import importlib
fit13 = importlib.import_module("13_fit_outbreaks")
mu = np.array([5.0, 50.0, 500.0])
kd = 4.0
rng = np.random.default_rng(1)
draws = rng.negative_binomial(kd, kd / (kd + mu), size=(40000, 3))
check("simulated negative binomial counts have mean mu and variance mu + mu^2/k",
      rel_err(draws.mean(0), mu) < 0.02 and rel_err(draws.var(0), mu + mu ** 2 / kd) < 0.05,
      f"means {np.round(draws.mean(0), 1).tolist()}, variances {np.round(draws.var(0)).tolist()}")

mu = np.linspace(2, 300, 40)
ysim = rng.negative_binomial(3.0, 3.0 / (3.0 + mu), size=(400, 40)).astype(float)
q = np.concatenate([fit13.quantile_residuals(row, mu, 3.0, rng) for row in ysim])
ks = kstest(q, "norm")
check("randomised quantile residuals are standard normal when the model is right",
      ks.pvalue > 0.01 and abs(q.mean()) < 0.05, f"mean {q.mean():.3f}, sd {q.std():.3f}, KS p = {ks.pvalue:.2f}")

check("variants: 'masked' is used only for series with flagged weeks",
      "masked" not in fit13.variants_for("mpox-US") and "masked" in fit13.variants_for("ebola-LR"),
      f"mpox-US {fit13.variants_for('mpox-US')}")

d = O.load_series("measles-SC")
y_masked = np.where(d["flag"], np.nan, d["y"])
k_true = int(np.nanargmax(y_masked))
yt = O.truncate_after_peak(d["y"], 4, flag=d["flag"])
check("'peak + 4 weeks' ignores flagged spikes when finding the peak",
      len(yt) == k_true + 5 and int(np.argmax(d["y"])) != k_true,
      f"measles-SC: highest week {int(np.argmax(d['y']))} is a spike; epidemic peak week {k_true}; "
      f"{len(yt)} weeks kept")

# Section 6.2: the West Texas likelihood drops sharply just below alpha = 1
y = O.load_series("measles-TX")["y"]
sg = O.rates("measles")
tx = F.AltFitter(y, *sg, "seir").fit()
drops = {}
for meth in ("pece", "l1"):
    ll = [F.best_scale_dispersion(y, F.caputo_weekly(a, tx["beta"], tx["e0"], len(y), *sg, meth, 16))[0]
          for a in (1.0, 0.99)]
    drops[meth] = ll[0] - ll[1]
check("West Texas: at the ordinary fit's parameters, alpha = 0.99 is about 21 units worse "
      "than alpha = 1 (both solvers)",
      all(18 < d < 24 for d in drops.values()) and abs(drops["pece"] - drops["l1"]) < 2,
      f"drop {drops['pece']:.1f} (PECE), {drops['l1']:.1f} (L1)")

print()
n_fail = results.count(False)
print("All %d checks passed." % len(results) if n_fail == 0 else f"{n_fail} check(s) FAILED.")
sys.exit(1 if n_fail else 0)
