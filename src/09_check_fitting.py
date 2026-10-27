"""
09_check_fitting.py -- checks of the fitting machinery (Stage 4).

Each check tests one piece of fitting.py or synthetic.py by an independent
route: scipy's own distributions, brute-force search, finite differences,
exact answers, or Monte Carlo. Prints PASS or FAIL; the last line should read
"All 19 checks passed." Takes about two minutes (it builds a small solver
table and runs two short profile fits).
"""

import math
import sys
import warnings

import numpy as np
from scipy.stats import chi2, nbinom

import fitting as F
import models as M
import solvers as S
import synthetic as Y

warnings.simplefilter("ignore")
rng = np.random.default_rng(2026)
results = []


def check(name, condition, detail=""):
    results.append(bool(condition))
    print(f"[{'PASS' if condition else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))


# ===========================================================================
# A. Likelihood pieces
# ===========================================================================
# 1. Negative binomial log-likelihood equals scipy's
worst = 0.0
for _ in range(200):
    mu = rng.uniform(0.1, 2000, size=20)
    kd = rng.uniform(0.05, 500)
    y = rng.negative_binomial(kd, kd / (kd + mu)).astype(float)
    ref = nbinom.logpmf(y, kd, kd / (kd + mu)).sum()
    worst = max(worst, abs(F.nb_loglik(y, mu, kd) - ref) / abs(ref))
check("1. NB log-likelihood equals scipy.stats.nbinom", worst < 1e-10, f"max relative error {worst:.1e}")

# 2. Inner maximisation over (A, k_disp) matches a brute-force grid search
ok, det = True, []
for kd_true in (3.0, 30.0):
    m = Y.expected_counts("T0") / 1e4                     # a model curve (any positive shape)
    y = rng.negative_binomial(kd_true, kd_true / (kd_true + 1e4 * m)).astype(float)
    ll, A, kd = F.best_scale_dispersion(y, m)
    As = A * np.exp(np.linspace(-0.2, 0.2, 201))
    ks = kd * np.exp(np.linspace(-1.0, 1.0, 201))
    grid = max(F.nb_loglik(y, a * m, k) for a in As for k in ks)
    ok &= ll >= grid - 1e-6
    det.append(f"k={kd_true}: optimiser {ll:.3f} vs grid {grid:.3f}")
check("2. inner (A, k_disp) optimiser is at least as good as a fine grid", ok, "; ".join(det))

# 3. Poisson quasi-likelihood: the closed-form scale is the maximiser
m = Y.expected_counts("T2") / 7.0
y = Y.expected_counts("T2")
ll, A, _ = F.poisson_quasi_loglik(y, m)
num = [float(np.sum(y * np.log(a * m) - a * m)) for a in (A * 0.999, A * 1.001)]
check("3. Poisson scale A = sum(y)/sum(m) is the maximum", all(ll >= v for v in num) and abs(A - 7.0) < 1e-9,
      f"A = {A:.6f} (true 7)")

# ===========================================================================
# B. Profile summaries and the test of alpha = 1
# ===========================================================================
# 4. A known quadratic profile: exact alpha-hat, LRT and (nearly exact) interval
grid = [0.4, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]
prof = [{"alpha": a, "loglik": -50.0 * (a - 0.73) ** 2} for a in grid]
s = F.summarise_profile(prof)
half = math.sqrt(F.CI_CRIT / 100.0)                       # 2*50*(a-0.73)^2 = 3.841
lrt_exact = 100 * 0.27 ** 2                               # 2 * 50 * (1 - 0.73)^2
ok = (abs(s["alpha_hat"] - 0.73) < 1e-9
      and lrt_exact - 0.1 < s["lrt"] <= lrt_exact            # conservative: never above the exact value
      and abs(s["ci_low"] - (0.73 - half)) < 0.01 and abs(s["ci_high"] - (0.73 + half)) < 0.01)
check("4. quadratic profile: alpha-hat exact; LRT conservative; 95% interval recovered", ok,
      f"alpha-hat {s['alpha_hat']:.4f}, LRT {s['lrt']:.4f} (exact {lrt_exact:.4f}), "
      f"CI [{s['ci_low']:.3f}, {s['ci_high']:.3f}] vs [{0.73 - half:.3f}, {0.73 + half:.3f}]")

# 5. Maximum at the boundary alpha = 1: alpha-hat = 1, LRT = 0, p = 1
prof = [{"alpha": a, "loglik": 10.0 * a} for a in grid]
s = F.summarise_profile(prof)
check("5. profile rising to alpha = 1: alpha-hat 1, LRT 0, not rejected",
      s["alpha_hat"] == 1.0 and s["lrt"] == 0.0 and not s["reject_alpha1"] and s["p_value"] == 1.0)

# 6. Boundary test: critical value 2.706 gives p = 0.05 under the 50:50 mixture
p = 0.5 * chi2.sf(F.LRT_CRIT_BOUNDARY, 1)
check("6. boundary LRT critical value 2.706 has p = 0.05", abs(p - 0.05) < 1e-4, f"p = {p:.5f}")

# ===========================================================================
# C. Optimiser and data generation
# ===========================================================================
# 7. Growth-rate-preserving starts (Proposition 10): mapped beta has the same r
settings_dummy = {"sigma": 1.0, "gamma": 1.0, "substeps": [{"alpha": 1.0, "method": "pece", "k": 8}]}
fit = F.ProfileFitter(np.ones(5), settings_dummy)
worst = 0.0
for _ in range(200):
    a1, a2 = rng.uniform(0.4, 1.0, 2)
    r = rng.uniform(0.05, 0.9)
    z = np.array([math.log(M.beta_for_growth_rate(r, 1.0, 1.0, a1)), -7.0])
    z2 = fit._mapped_start(z, a1, a2)
    p2 = {"beta": math.exp(z2[0]), "sigma": 1.0, "gamma": 1.0, "N": 1.0, "mu": 0.0}
    worst = max(worst, abs(M.growth_rate_caputo(p2, a2) / r - 1))
check("7. mapped starting values keep the growth rate (Proposition 10)", worst < 1e-10,
      f"max relative error {worst:.1e}")

# 8. Noisy data have the right mean and variance (Monte Carlo, 4000 draws)
mu = Y.expected_counts("T5")
draws = np.array([Y.replicate("T5", 10, r) for r in range(4000)])
mean_err = np.max(np.abs(draws.mean(axis=0) / mu - 1))
var_err = np.max(np.abs(draws.var(axis=0) / (mu + mu ** 2 / 10) - 1))
check("8. NB replicates: mean mu and variance mu + mu^2/k", mean_err < 0.03 and var_err < 0.12,
      f"max relative error: mean {mean_err:.3f}, variance {var_err:.3f}")

# 9. Same seed -> same data; data identical to the saved file if it exists
a, b = Y.replicate("T3", 100, 7), Y.replicate("T3", 100, 7)
check("9. replicates are reproducible", np.array_equal(a, b) and not np.array_equal(a, Y.replicate("T3", 100, 8)))

# 10. Every truth (except declining transmission) is set up to grow at r0.
#     Checked through a route independent of the formulas used to choose the
#     parameters: the dominant eigenvalue of the linearised system (ODE truths),
#     lambda1^(1/alpha) (Caputo), and the AHM growth-rate root.
r0 = Y.R0_GROWTH
got = {}
got["T0"] = M.growth_rate_caputo({"beta": Y.BETA_SEIR, "sigma": 1.0, "gamma": 1.0, "N": 1.0, "mu": 0.0}, 1.0)
mE, mI = Y.ERLANG_STAGES
bE = M.r0_from_growth_rate_erlang(r0, 1.0, 1.0, mE, mI)
got["T1"] = max(np.linalg.eigvals(M.erlang_early_matrix(bE, 1.0, 1.0, mE, mI)).real)
bC = M.beta_for_growth_rate(r0, 1.0, 1.0, 0.8)
got["T4"] = M.growth_rate_caputo({"beta": bC, "sigma": 1.0, "gamma": 1.0, "N": 1.0, "mu": 0.0}, 0.8)
ap = Y.AHM_PARAMS
wS0 = M.ahm_transmission_for_growth_rate(r0, ap["tau"], ap["alpha"], ap["gamma_c"])
got["T6"] = M.ahm_growth_rate(wS0, ap["tau"], ap["alpha"], ap["gamma_c"])
g, q = np.array(Y.HETERO_PARAMS["gammas"]), np.array(Y.HETERO_PARAMS["fractions"])
bH = M.beta_for_growth_rate_hetero(r0, 1.0, g, q)
Ah = np.array([[-1.0, bH, bH], [q[0], -g[0], 0.0], [q[1], 0.0, -g[1]]])   # (E, I1, I2)
got["T7"] = max(np.linalg.eigvals(Ah).real)
worst = max(abs(v / r0 - 1) for v in got.values())
check("10. truths are set up to grow at r0 = 0.3416 per week", worst < 1e-8,
      ", ".join(f"{k}: {v:.6f}" for k, v in got.items()))

# ===========================================================================
# D. Fitting on known answers (small solver table, two alphas)
# ===========================================================================
print("    building a small solver table for checks 11-13 ...")
table = F.substep_table([0.8, 1.0], Y.SIGMA, Y.GAMMA, Y.K_WEEKS)
settings = {"sigma": Y.SIGMA, "gamma": Y.GAMMA, "substeps": table}

# 11. The table's settings are accurate for random parameters (not just the two used to build it)
worst = 0.0
for row in table:
    for _ in range(3):
        r = rng.uniform(0.05, F.R_MAX)
        beta = M.beta_for_growth_rate(r, 1.0, 1.0, row["alpha"])
        e0 = 10 ** rng.uniform(-5, -2)
        ref = F.caputo_weekly(row["alpha"], beta, e0, Y.K_WEEKS, 1.0, 1.0, "pece", 512)
        m = F.caputo_weekly(row["alpha"], beta, e0, Y.K_WEEKS, 1.0, 1.0, row["method"], row["k"])
        worst = max(worst, np.max(np.abs(m - ref)) / np.max(ref))
check("11. solver settings reach 0.2% accuracy for random parameters", worst < 2e-3,
      f"max error {worst:.1e} of the peak")

# 12-13. Exact recovery from noise-free data at the true alpha
for idx, (tid, a, beta_true) in enumerate([("T0", 1.0, Y.BETA_SEIR),
                                           ("T4", 0.8, M.beta_for_growth_rate(Y.R0_GROWTH, 1, 1, 0.8))]):
    fit = F.ProfileFitter(Y.expected_counts(tid), settings, likelihood="poisson_quasi")
    start = np.array([math.log(beta_true * 1.3), math.log(3e-3)])
    ll, x = fit.optimise(a, start)
    b_err = abs(math.exp(x[0]) / beta_true - 1)
    e_err = abs(math.exp(x[1]) / (Y.E0_SEED / Y.N_POP) - 1)
    check(f"{12 + idx}. noise-free {tid}: fit at alpha = {a} recovers beta and e0", b_err < 2e-3 and e_err < 2e-2,
          f"beta error {b_err:.1e}, e0 error {e_err:.1e}")

# 14. The Caputo SEIR's cumulative infections can FALL (Section 5): both
#     methods agree that S rises after its minimum at alpha = 0.9
a = 0.9
p = {"beta": M.beta_for_growth_rate(0.5, 1.0, 1.0, a), "sigma": 1.0, "gamma": 1.0, "N": 1e5, "mu": 0.0}
rises = []
for method in ("pece", "l1"):
    _, Ysol = S.simulate_caputo_seir(p, a, [1e5 - 10, 10, 0, 0], 150.0, 0.25, method)
    Sw = Ysol[::4, 0]
    rises.append((Sw[-1] - Sw.min()) / 1e5)
check("14. Caputo SEIR (alpha = 0.9): S rises after its minimum, both methods agree",
      min(rises) > 0.1 and abs(rises[0] - rises[1]) < 0.01,
      f"rise in S by week 150: {rises[0]:.4f} N (PECE), {rises[1]:.4f} N (L1)")

# 15-16. Proposition 11: the AHM model's "memory" is heterogeneity.
#   15: the Mittag-Leffler survival function is a mixture of exponentials,
#       E_a(-(t/tau)^a) = int_0^inf exp(-lam t) g(lam) d lam, with g a density;
#   16: an ordinary SIR model with many memoryless infectious classes, whose
#       recovery rates are spread according to g, reproduces the AHM model.
from scipy.integrate import quad
from mittag_leffler import ml


def g_density(lam, a, tau):
    r = tau * lam
    return tau * math.sin(a * math.pi) * r ** (a - 1) / (math.pi * (r ** (2 * a) + 2 * r ** a * math.cos(a * math.pi) + 1))


worst, mass_err = 0.0, 0.0
for a, tau in [(0.5, 1.0), (0.7, 2.0), (0.9, 0.5)]:
    split = lambda f: quad(f, 0, 1, limit=400)[0] + quad(f, 1, np.inf, limit=400)[0]
    mass_err = max(mass_err, abs(split(lambda l: g_density(l, a, tau)) - 1))
    for t in (0.1, 1.0, 5.0, 30.0):
        mix = split(lambda l: math.exp(-l * t) * g_density(l, a, tau))
        worst = max(worst, abs(mix / ml(-(t / tau) ** a, a) - 1))
check("15. Mittag-Leffler survival = mixture of exponential survivals (g is a density)",
      worst < 1e-8 and mass_err < 1e-9, f"max relative error {worst:.1e}; |int g - 1| = {mass_err:.1e}")

a, tau, gc = Y.AHM_PARAMS["alpha"], Y.AHM_PARAMS["tau"], Y.AHM_PARAMS["gamma_c"]
om = M.ahm_transmission_for_growth_rate(Y.R0_GROWTH, tau, a, gc) / Y.N_POP
x, w = np.polynomial.legendre.leggauss(300)                 # classes spread in log(rate)
lo, hi = math.log(1e-8), math.log(1e5)
u = 0.5 * (hi - lo) * x + 0.5 * (hi + lo)
lam = np.exp(u)
wt = np.array([g_density(l, a, tau) for l in lam]) * lam * 0.5 * (hi - lo) * w
# (rates above 1e5 per week, mass 7e-5, are infectious for seconds and are left out)


def mixture_rhs(t, yv):
    inf = om * yv[0] * yv[1:].sum()
    return np.concatenate([[-inf], wt * inf - (lam + gc) * yv[1:]])


I0 = Y.E0_SEED
_, Ymix = S.solve_ode(mixture_rhs, np.concatenate([[Y.N_POP - I0], wt * I0]), 36.0, np.arange(0, 37.0))
I_mix = Ymix[:, 1:].sum(axis=1)
_, o = S._solve_ahm_grid(om, tau, a, gc, Y.N_POP - I0, I0, 36.0, 1.0 / 320)
diff = np.max(np.abs(I_mix - o["I"][::320])) / I_mix.max()
check("16. memoryless SIR with 300 recovery-rate classes reproduces the AHM model", diff < 2e-4,
      f"max difference {diff:.1e} of the peak (AHM step 1/320 week)")

# 17. No "inverse crime": the Caputo truths T4 and T5 are generated with the
#     predictor-corrector; regenerating them with the other method (implicit
#     L1, 1024 sub-steps/week) gives the same curves.
worst = 0.0
for a in (0.8, 0.6):
    p = {"beta": M.beta_for_growth_rate(Y.R0_GROWTH, 1.0, 1.0, a), "sigma": 1.0, "gamma": 1.0,
         "N": Y.N_POP, "mu": 0.0}
    _, Yl = S.simulate_caputo_seir(p, a, [Y.N_POP - Y.E0_SEED, Y.E0_SEED, 0, 0], float(Y.K_WEEKS),
                                   1.0, "l1", substeps=1024)
    w_l1 = np.diff(Yl[:, 2] + Yl[:, 3])
    w_pc = Y.weekly_entries("T4" if a == 0.8 else "T5")
    worst = max(worst, np.max(np.abs(w_l1 - w_pc)) / w_pc.max())
check("17. Caputo truths: independent L1 solution matches the PECE-generated data", worst < 1e-3,
      f"max difference {worst:.1e} of the peak")

# 18. Solver setting for an alpha BETWEEN table values (the next-lower row is used)
row = {"alpha": 0.80, "method": table[0]["method"], "k": table[0]["k"]}
worst = 0.0
for a in (0.83, 0.87):
    beta = M.beta_for_growth_rate(F.R_MAX, 1.0, 1.0, a)
    ref = F.caputo_weekly(a, beta, 1e-3, Y.K_WEEKS, 1.0, 1.0, "pece", 512)
    m = F.caputo_weekly(a, beta, 1e-3, Y.K_WEEKS, 1.0, 1.0, row["method"], row["k"])
    worst = max(worst, np.max(np.abs(m - ref)) / np.max(ref))
check("18. alphas between table rows: the lower row's setting is accurate", worst < 2e-3,
      f"max error {worst:.1e} of the peak at alpha = 0.83, 0.87")

# 19. Expected NB log-likelihood (closed form) against Monte Carlo
mu = Y.expected_counts("T7")
m1, m2 = mu / 5.0, Y.expected_counts("T0") / 5.0           # the truth's shape, and a wrong shape
ll1, A1, _ = F.expected_nb_loglik(mu, m1, 10.0)
ll2, A2, _ = F.expected_nb_loglik(mu, m2, 10.0)
draws = [Y.replicate("T7", 10, r) for r in range(3000)]
mc = np.mean([F.nb_loglik(y, A1 * m1, 10.0) - F.nb_loglik(y, A2 * m2, 10.0) for y in draws])
ok = abs(A1 - 5.0) < 1e-6 and abs((ll1 - ll2) - mc) < 0.05 * abs(ll1 - ll2)
check("19. expected NB log-likelihood: right maximiser and matches Monte Carlo", ok,
      f"A = {A1:.6f} (true 5); difference {ll1 - ll2:.2f} (closed form) vs {mc:.2f} (Monte Carlo)")

print()
if all(results):
    print(f"All {len(results)} checks passed.")
else:
    print(f"{results.count(False)} of {len(results)} checks FAILED.")
    sys.exit(1)
