"""
07_check_solvers.py -- validate the numerical solvers (Section 4 of the paper).

Every solver is tested against something it was NOT built from: exact
Mittag-Leffler solutions, the ordinary-equation case alpha = 1 solved by an
independent high-accuracy ODE solver, the other Caputo method, conservation
laws, and the growth rates proved in Section 3.

Writes results/stage3_convergence.csv (convergence table and figure in Section 4.5).
Each check prints PASS or FAIL; the last line should read
"All 21 checks passed." Takes about a minute.
"""

import sys
import warnings

import numpy as np
import pandas as pd
from scipy.integrate import IntegrationWarning

import config
import models as M
import solvers as S
from mittag_leffler import ml, ml_neg_fast

warnings.filterwarnings("error", category=RuntimeWarning)
warnings.filterwarnings("error", category=IntegrationWarning)
rng = np.random.default_rng(2026)
results = []


def check(name, condition, detail=""):
    results.append(bool(condition))
    print(f"[{'PASS' if condition else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))


def orders(errors):
    """Observed orders log2(e_h / e_{h/2}) for errors at h, h/2, h/4, ..."""
    e = np.asarray(errors)
    return np.log2(e[:-1] / e[1:])


def local_slope(t, y, T, ratio=1.1):
    """d log y / d log t between T/ratio and T (y and t on the same grid)."""
    i = np.searchsorted(t, T)
    j = np.searchsorted(t, T / ratio)
    return (np.log(y[i]) - np.log(y[j])) / (np.log(t[i]) - np.log(t[j]))


ALPHAS = [0.3, 0.5, 0.7, 0.9, 1.0]
STEPS = [0.1 / 2 ** k for k in range(5)]          # 0.1, 0.05, ..., 0.00625

# ===========================================================================
# A. Building blocks
# ===========================================================================
# 1. Fast Mittag-Leffler (used by the AHM solver) against the reference version
worst = 0.0
for a in [0.2, 0.4, 0.6, 0.8, 0.95, 0.99, 0.999, 0.9999]:
    x = np.concatenate([np.linspace(0, 0.5, 6), np.linspace(0.6, 5, 12), np.logspace(0.8, 3.5, 12)])
    worst = max(worst, np.max(np.abs(ml_neg_fast(x, a) - ml(-x, a)) / ml(-x, a)))
check("1. fast E_a(-x) agrees with the high-precision version (alpha up to 0.9999)", worst < 1e-9,
      f"max relative error {worst:.1e}")

# 2-3. Convergence on D^a y = -y, exact solution E_a(-t^a), errors at T = 2
rows = []
ok_p, ok_l, detail_p, detail_l = True, True, [], []
for a in ALPHAS:
    exact = ml(-(2.0 ** a), a)
    for method in ["pece", "l1"]:
        errs = []
        for h in STEPS:
            _, Y = S.solve_caputo(lambda t, y: -y, [1.0], a, 2.0, h, method=method,
                                  jac=lambda t, y: np.array([[-1.0]]))
            errs.append(abs(Y[-1, 0] - exact))
            rows.append({"method": method, "alpha": a, "h": h, "error": errs[-1]})
        q = orders(errs)[-1]
        if method == "pece":
            ok_p &= bool(np.all(np.diff(errs) < 0)) and (1 + a - 0.25 <= q <= 2.1)
            detail_p.append(f"{a}:{q:.2f}")
        else:
            ok_l &= bool(np.all(np.diff(errs) < 0)) and (0.85 <= q <= 1.15)
            detail_l.append(f"{a}:{q:.2f}")
conv = pd.DataFrame(rows)
config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
conv.to_csv(config.RESULTS_DIR / "stage3_convergence.csv", index=False)
check("2. PECE converges to E_a(-t^a) with order about 1+alpha", ok_p, "alpha:order " + " ".join(detail_p))
check("3. L1 converges to E_a(-t^a) with order about 1", ok_l, "alpha:order " + " ".join(detail_l))

# 4. Linear early-phase system against its exact solution (Proposition 6)
par = {"beta": 2.0, "sigma": 1.5, "gamma": 1.0, "N": 1.0, "mu": 0.0}
ok, det = True, []
for a in [0.6, 0.9]:
    A = M.early_matrix(par, a)
    ex = M.early_solution(par, a, [5.0], x0=(1.0, 0.0))[0]
    for method in ["pece", "l1"]:
        errs = []
        for h in [0.05, 0.025, 0.0125]:
            _, Y = S.solve_caputo(lambda t, y: A @ y, [1.0, 0.0], a, 5.0, h, method=method,
                                  jac=lambda t, y: A)
            errs.append(np.max(np.abs(Y[-1] - ex)) / np.max(np.abs(ex)))
        ok &= bool(np.all(np.diff(errs) < 0)) and errs[-1] < (2e-4 if method == "pece" else 5e-3)
        det.append(f"{method} a={a}: {errs[-1]:.1e}")
check("4. both methods converge to the exact early-phase solution", ok, "; ".join(det))

# ===========================================================================
# B. Caputo SEIR
# ===========================================================================
P = {"beta": 2.0, "sigma": 1.0, "gamma": 1.0, "N": 1e6, "mu": 0.0}
y0 = [1e6 - 10.0, 10.0, 0.0, 0.0]

# 5. alpha = 1: both methods against an independent ODE solver
ode_rhs = lambda t, y: M.caputo_seir_rhs(y, P, 1.0)
T5 = 60.0
_, ref = S.solve_ode(ode_rhs, y0, T5, [T5])
ok, det = True, []
for method, target in [("pece", 2.0), ("l1", 1.0)]:
    errs = []
    for h in [0.05, 0.025, 0.0125]:
        _, Y = S.simulate_caputo_seir(P, 1.0, y0, T5, h, method)
        errs.append(np.max(np.abs(Y[-1] - ref[-1])) / P["N"])
    q = orders(errs)[-1]
    ok &= abs(q - target) < 0.2
    det.append(f"{method} order {q:.2f}")
check("5. alpha = 1: PECE is 2nd order and L1 1st order against the ODE solution", ok, ", ".join(det))

# 6. alpha < 1: both methods converge to the same nonlinear solution.
#    Reference: PECE with a very small step. Errors are the largest difference
#    in I over [0, 60], divided by the peak of I.
ok, det = True, []
for a in [0.6, 0.85]:
    tr, Yr = S.simulate_caputo_seir(P, a, y0, 60.0, 0.003125, "pece")
    peak = Yr[:, 2].max()
    for method, tol in [("pece", 5e-5), ("l1", 1e-3)]:
        errs = []
        for h in [0.025, 0.0125, 0.00625]:
            k = int(round(h / 0.003125))
            _, Y = S.simulate_caputo_seir(P, a, y0, 60.0, h, method)
            errs.append(np.max(np.abs(Y[:, 2] - Yr[::k, 2])) / peak)
        ok &= bool(np.all(np.diff(errs) < 0)) and errs[-1] < tol
        det.append(f"{method} a={a}: " + "/".join(f"{e:.1e}" for e in errs))
check("6. PECE and L1 converge to the same nonlinear solution (error / peak I)", ok, "; ".join(det))

# 7-8. Conservation and positivity on random parameter sets, including
#      large transmission rates and coarse output steps (daily, half-weekly,
#      weekly), so that the automatic step rules are really tested.
ok_c, ok_p = True, True
worst_c, worst_p, n_runs = 0.0, 0.0, 0
for _ in range(30):
    p = {"beta": rng.uniform(0.5, 20), "sigma": rng.uniform(0.3, 5), "gamma": rng.uniform(0.3, 3),
         "N": 10 ** rng.uniform(3, 7), "mu": 0.0}
    a = rng.choice([0.3, 0.5, 0.7, 0.9, 1.0])
    yy = [p["N"] - 5.0, 5.0, 0.0, 0.0]
    for h in [1.0 / 7.0, 0.5, 1.0]:
        T = 28.0 if h > 0.2 else 4.0 * 7.0
        for method in ["pece", "l1"]:
            _, Y = S.simulate_caputo_seir(p, a, yy, T, h, method)
            worst_c = max(worst_c, np.max(np.abs(Y.sum(axis=1) - p["N"])) / p["N"])
            worst_p = min(worst_p, Y.min() / p["N"])
            n_runs += 1
check("7. S+E+I+R = N is conserved by both methods", worst_c < 1e-10, f"max error {worst_c:.1e}")
check("8. no compartment goes negative (beta up to 20, steps up to 1 week)", worst_p > -1e-9,
      f"{n_runs} runs, most negative value / N = {worst_p:.1e}")

# 9. Simulated early growth rate equals r = lambda1^(1/alpha) (Proposition 6)
ok, det = True, []
for a in [0.5, 0.8, 1.0]:
    p = {"beta": 2.0, "sigma": 1.0, "gamma": 1.0, "N": 1e12, "mu": 0.0}
    r = M.growth_rate_caputo(p, a)
    # run until e^(r t) ~ 1e6 (S still about N = 1e12)
    T9 = round(14.0 / r / 0.01) * 0.01              # a whole number of steps
    t, Y = S.simulate_caputo_seir(p, a, [1e12 - 1, 1, 0, 0], T9, 0.01, "pece")
    # slope of log I over the last quarter (exponential phase, S still ~ N)
    i0, i1 = int(0.75 * len(t)), len(t) - 1
    r_sim = (np.log(Y[i1, 2]) - np.log(Y[i0, 2])) / (t[i1] - t[i0])
    ok &= abs(r_sim / r - 1) < 0.01
    det.append(f"a={a}: {r_sim / r - 1:+.1e}")
check("9. simulated growth rate matches r = lambda1^(1/alpha)", ok, "relative error " + ", ".join(det))

# 10. PECE stability limit on D^a y = -lam y: stable for h lam^(1/a) <= 1.5
ok = True
for a in ALPHAS:
    lam = 1.0
    _, Y = S.solve_caputo(lambda t, y: -lam * y, [1.0], a, 600.0, 1.5, method="pece")
    ok &= bool(np.all(np.isfinite(Y)) and np.all(np.abs(Y) <= 1.0))
check("10. PECE is stable for h * lam^(1/alpha) <= 1.5 (all alpha)", ok)

# 11-12. Long-run decline: algebraic, slope -> -alpha; exponential at alpha = 1
ok, det = True, []
for a in [0.6, 0.8]:
    t, Y = S.simulate_caputo_seir(P, a, y0, 6000.0, 0.5, "l1")
    s = local_slope(t, Y[:, 2], 6000.0)
    ok &= abs(s + a) < 0.03
    det.append(f"a={a}: {s:.3f}")
t2, Y2 = S.simulate_caputo_seir(P, 0.8, y0, 6000.0, 0.25, "l1")      # halve the step: same slope?
s2 = local_slope(t2, Y2[:, 2], 6000.0)
t5, Y5 = S.simulate_caputo_seir(P, 0.8, y0, 6000.0, 0.5, "l1")
ok &= abs(s2 - local_slope(t5, Y5[:, 2], 6000.0)) < 0.005
det.append(f"a=0.8, h=0.25: {s2:.3f}")
check("11. Caputo SEIR decline is algebraic with slope -> -alpha", ok, "slope at t=6000: " + ", ".join(det))
ok, det = True, []
for a in [0.6, 0.8]:                 # the transitional 'shoulder' is confirmed by both methods
    tp, Yp = S.simulate_caputo_seir(P, a, y0, 1000.0, 0.1, "pece")
    tl, Yl = S.simulate_caputo_seir(P, a, y0, 1000.0, 0.5, "l1")
    for T in [200.0, 450.0, 1000.0]:
        sp, sl = local_slope(tp, Yp[:, 2], T), local_slope(tl, Yl[:, 2], T)
        ok &= abs(sp - sl) < 0.02
        det.append(f"{a}@{T:.0f}:{sp:.2f}")
t1, Y1 = S.simulate_caputo_seir(P, 1.0, y0, 200.0, 0.05, "pece")
ok &= Y1[-1, 2] < 1e-6 * Y1[:, 2].max()       # alpha = 1: exponential collapse
check("12. intermediate tail slopes agree between methods; alpha=1 decays exponentially", ok,
      " ".join(det))

# ===========================================================================
# C. AHM model
# ===========================================================================
om, tau, N = 2.5e-6, 1.0, 1e6
S0, I0 = N - 10.0, 10.0

# 13. alpha = 1: equals SIR with removal rate 1/tau + gamma_c (2nd order)
gc = 0.2
rhs = lambda t, y: [-om * y[0] * y[1], om * y[0] * y[1] - (1 / tau + gc) * y[1]]
grid = np.linspace(0, 40, 401)
_, Yr = S.solve_ode(rhs, [S0, I0], 40, grid)
errs = []
for h in [0.1, 0.05, 0.025]:
    _, o = S._solve_ahm_grid(om, tau, 1.0, gc, S0, I0, 40, h)     # raw scheme, no auto step
    errs.append(np.max(np.abs(o["I"][np.round(grid / h).astype(int)] - Yr[:, 1])) / Yr[:, 1].max())
q = orders(errs)[-1]
check("13. AHM at alpha = 1 equals the SIR ODE, 2nd-order convergence", abs(q - 2) < 0.2, f"order {q:.2f}")

# 14. alpha < 1: self-convergence order about 1+alpha; conservation; I >= 0
ok, det = True, []
for a in [0.6, 0.9]:
    vals = []
    for h in [0.2, 0.1, 0.05, 0.025]:
        _, o = S._solve_ahm_grid(om, tau, a, 0.0, S0, I0, 60, h)  # raw scheme, no auto step
        vals.append(o["I"][-1])
        ok &= np.max(np.abs(o["S"] + o["I"] + o["R"] - N)) / N < 1e-12 and o["I"].min() >= 0
    q = np.log2(abs(vals[1] - vals[2]) / abs(vals[2] - vals[3]))
    ok &= abs(q - (1 + a)) < 0.3
    det.append(f"a={a}: order {q:.2f}")
check("14. AHM: order about 1+alpha, conservation, I >= 0", ok, "; ".join(det))

# 15. AHM early growth rate equals the root of omega S0 phi_hat(gamma_c + r) = 1 (Prop 8)
ok, det = True, []
for a, gcc in [(0.6, 0.3), (0.8, 0.0), (0.9, 0.5)]:
    om2, N2 = 3.0e-12, 1e12
    r = M.ahm_growth_rate(om2 * N2, tau, a, gcc)
    T15 = round(14.0 / r / 0.02) * 0.02             # e^(r t) ~ 1e6 << N
    t, o = S.solve_ahm(om2, tau, a, gcc, N2 - 1, 1.0, T15, 0.02)
    i0, i1 = int(0.75 * len(t)), len(t) - 1
    r_sim = (np.log(o["I"][i1]) - np.log(o["I"][i0])) / (t[i1] - t[i0])
    ok &= abs(r_sim / r - 1) < 0.01
    det.append(f"a={a},gc={gcc}: {r_sim / r - 1:+.1e}")
check("15. AHM simulated growth rate matches Proposition 8", ok, "; ".join(det))

# 16. AHM with gamma_c = 0: everyone eventually infected; tail slope -> -alpha
ok, det = True, []
for a in [0.6, 0.9]:
    t, o = S.solve_ahm(om, tau, a, 0.0, S0, I0, 2000, 0.5)
    s = local_slope(t, o["I"], 2000.0)
    ok &= abs(s + a) < 0.03 and o["S"][-1] < 0.01 * N
    det.append(f"a={a}: slope {s:.3f}, S/N {o['S'][-1] / N:.1e}")
check("16. AHM (gamma_c = 0): S -> 0 and tail slope -> -alpha", ok, "; ".join(det))

# ===========================================================================
# D. Integer-order models
# ===========================================================================
# 17. Erlang SEIR: m = 1 equals the alpha = 1 SEIR; growth rate matches Prop 9
tg = np.linspace(0, 60, 61)
_, o1 = S.simulate_erlang_seir(2.0, 1.0, 1.0, 1e6, 1, 1, 10.0, 60, tg)
_, Yc = S.solve_ode(ode_rhs, y0, 60, tg)
ok = np.max(np.abs(o1["I"] - Yc[:, 2])) / 1e6 < 1e-8
det = []
for mE, mI in [(2, 3), (4, 1)]:
    r = 0.4
    beta = 1.0 * M.r0_from_growth_rate_erlang(r, 1.0, 1.0, mE, mI)
    tt = np.linspace(0, 60, 601)
    _, o = S.simulate_erlang_seir(beta, 1.0, 1.0, 1e14, mE, mI, 1.0, 60, tt)
    r_sim = (np.log(o["I"][-1]) - np.log(o["I"][450])) / (tt[-1] - tt[450])
    ok &= abs(r_sim / r - 1) < 1e-3
    det.append(f"({mE},{mI}): {r_sim / r - 1:+.1e}")
check("17. Erlang SEIR: matches SEIR when m=1; growth rate matches Prop 9", ok, "; ".join(det))

# 18. Declining transmission: rho = 1 gives constant beta; rho < 1 slows growth
tt = np.linspace(0, 30, 301)
_, a1 = S.simulate_seir_beta_t(2.0, 1.0, 0.3, 1.0, 1.0, 1e12, 1.0, 30, tt)
_, a2 = S.simulate_erlang_seir(2.0, 1.0, 1.0, 1e12, 1, 1, 1.0, 30, tt)
_, a3 = S.simulate_seir_beta_t(2.0, 0.5, 0.3, 1.0, 1.0, 1e12, 1.0, 30, tt)
g = np.diff(np.log(a3["I"][50:]))                 # log-growth per step after the start
ok = np.max(np.abs(a1["I"] - a2["I"]) / a2["I"].max()) < 1e-8 and np.all(np.diff(g) < 0)
check("18. beta(t) model: rho = 1 is ordinary SEIR; rho < 1 bends the log curve down", ok)

# 19. Reporting delay: weights sum to 1 with the requested mean; growth rate unchanged
h = M.gamma_delay_weights(2.0, 3.0, 60)
c = np.exp(0.3 * np.arange(60))
y = S.apply_reporting_delay(c, 2.0, 3.0, rho_rep=0.5)
r_after = np.log(y[-1] / y[-2])
mean_disc = np.sum(np.arange(60) * h) + 0.5     # weekly bins: add half a week
ok = abs(h.sum() - 1) < 1e-12 and abs(mean_disc - 2.0) < 0.05 and abs(r_after - 0.3) < 1e-6
check("19. reporting delay: weights valid; exponential growth rate unchanged", ok,
      f"mean {mean_disc:.3f}, growth {r_after:.6f}")

# 20. AHM solver with the automatic step: (a) stays non-negative even at very
#     large transmission (omega N = 15 per week) with weekly output; (b) at a
#     realistic growth rate (0.4 per week) weekly output is within 0.5% of a
#     fine-step solution (relative to the peak)
ok, det = True, []
_, o = S.solve_ahm(15.0 / N, tau, 0.8, 0.5, S0, I0, 20.0, 1.0)
ok &= o["I"].min() >= 0
det.append(f"omega N = 15: min I {o['I'].min():.1f}")
for a, gcc, T20 in [(0.8, 0.5, 28.0), (0.6, 0.0, 52.0)]:
    omN = M.ahm_transmission_for_growth_rate(0.4, tau, a, gcc)
    _, fine = S._solve_ahm_grid(omN / N, tau, a, gcc, S0, I0, T20, 0.002)
    _, o = S.solve_ahm(omN / N, tau, a, gcc, S0, I0, T20, 1.0)
    err = np.max(np.abs(o["I"] - fine["I"][::500])) / fine["I"].max()
    ok &= err < 5e-3
    det.append(f"a={a}, gc={gcc}: error {err:.1e}")
check("20. AHM automatic step: non-negative at large transmission, accurate at weekly output",
      ok, "; ".join(det))

# 21. Integer-order models decline exponentially: the log-log slope keeps
#     falling (for a power law it would level off)
tt = np.linspace(0, 100, 1001)
_, e1 = S.simulate_erlang_seir(2.0, 1.0, 1.0, 1e6, 2, 3, 10.0, 100, tt)
_, e2 = S.simulate_seir_beta_t(3.0, 0.2, 0.2, 1.0, 1.0, 1e6, 10.0, 100, tt)   # R drops below 1
ok, det = True, []
for name, o in [("Erlang", e1), ("beta(t)", e2)]:
    s40, s60 = local_slope(tt, o["I"], 40.0), local_slope(tt, o["I"], 60.0)
    ok &= s60 < s40 < -5
    det.append(f"{name}: {s40:.1f} then {s60:.1f}")
check("21. integer-order declines are exponential (log-log slope keeps falling)", ok, "; ".join(det))

print()
if all(results):
    print(f"All {len(results)} checks passed.")
else:
    print(f"{results.count(False)} of {len(results)} checks FAILED.")
    sys.exit(1)
