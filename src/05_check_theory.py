"""
05_check_theory.py -- numerical checks of every formula in Section 3.

A proof is only as good as its algebra, so each claim in the paper is tested
here by an INDEPENDENT route (numerical integration, finite differences,
eigenvalues, or a separate high-precision calculation). Each check prints
PASS or FAIL; the last line should read "All 21 checks passed."
Takes about a minute.
"""

import math
import sys
import warnings

import mpmath as mp
import numpy as np
from scipy.integrate import quad
from scipy.special import erfc, gamma as Gamma

import models as M
from mittag_leffler import ml

# Numerical warnings (overflow, failed integration) count as failures
from scipy.integrate import IntegrationWarning
warnings.filterwarnings("error", category=RuntimeWarning)
warnings.filterwarnings("error", category=IntegrationWarning)
rng = np.random.default_rng(2026)  # fixed seed: the same random tests every run
results = []


def check(name, condition, detail=""):
    results.append(bool(condition))
    print(f"[{'PASS' if condition else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))


def random_params(mu_positive):
    """Random parameters on wide ranges (per week)."""
    return {"beta": rng.uniform(0.2, 10), "sigma": rng.uniform(0.2, 5),
            "gamma": rng.uniform(0.2, 5), "N": rng.uniform(1e3, 1e7),
            "mu": rng.uniform(1e-4, 0.05) if mu_positive else 0.0}


ALPHAS = [0.3, 0.5, 0.7, 0.9, 1.0]

# ===========================================================================
# A. Fractional calculus facts (Section 3.1)
# ===========================================================================

# 1. Caputo derivative of t^p  =  Gamma(p+1)/Gamma(p+1-a) t^(p-a)
#    Direct definition: (1/Gamma(1-a)) int_0^t (t-s)^(-a) p s^(p-1) ds.
#    quad's weight "alg" handles both end-point singularities exactly.
worst = 0.0
for a in [0.2, 0.5, 0.8, 0.95]:
    for p in [0.5, 1.0, 2.0, 3.5]:
        for t in [0.5, 2.0]:
            integral, _ = quad(lambda s: p, 0, t, weight="alg", wvar=(p - 1.0, -a))
            numeric = integral / Gamma(1 - a)
            formula = Gamma(p + 1) / Gamma(p + 1 - a) * t ** (p - a)
            worst = max(worst, abs(numeric - formula) / abs(formula))
check("1. Caputo derivative of t^p matches the Gamma formula", worst < 1e-10,
      f"max relative error {worst:.1e}")

# 2. Mittag-Leffler special cases
v = [abs(ml(1.7, 1.0) - math.exp(1.7)),
     max(abs(ml(-x, 0.5) - math.exp(x * x) * erfc(x)) / (math.exp(x * x) * erfc(x)) for x in [0.3, 2.0, 8.0]),
     abs(ml(-4.0, 2.0) - math.cos(2.0))]
check("2. E_1(z)=e^z, E_1/2(-x)=e^(x^2)erfc(x), E_2(-x^2)=cos(x)", max(v) < 1e-12,
      f"max error {max(v):.1e}")

# 3. y(t) = E_a(-lam t^a) solves D^a y = -lam y, y(0) = 1.
#    Checked through the equivalent Volterra form  y = 1 - lam I^a y.
worst = 0.0
lam = 1.3
for a in [0.4, 0.7, 0.9]:
    y = lambda s, a=a: ml(-lam * s ** a, a)
    for t in [0.5, 1.5, 4.0]:
        Iay, _ = quad(y, 0, t, weight="alg", wvar=(0.0, a - 1.0), limit=200)
        rhs = 1.0 - lam * Iay / Gamma(a)
        worst = max(worst, abs(y(t) - rhs))
check("3. E_a(-lam t^a) solves the relaxation equation (Volterra form)", worst < 1e-8,
      f"max error {worst:.1e}")

# 4. d/dt E_a(-k t^a) = -k t^(a-1) E_{a,a}(-k t^a)   (used in Lemma 2)
worst, h, k = 0.0, 1e-5, 2.0
for a in [0.3, 0.6, 0.9]:
    for t in [0.3, 1.0, 5.0, 40.0]:
        fd = (ml(-k * (t + h) ** a, a) - ml(-k * (t - h) ** a, a)) / (2 * h)
        exact = -k * t ** (a - 1) * ml(-k * t ** a, a, a)
        worst = max(worst, abs(fd - exact) / abs(exact))
check("4. derivative identity for E_a(-k t^a)", worst < 1e-6, f"max relative error {worst:.1e}")

# 5. Signs used in the positivity proof: E_a(-x) > 0 and decreasing; E_{a,a}(-x) >= 0
ok = True
for a in [0.1, 0.3, 0.5, 0.7, 0.9, 0.99]:
    x = np.concatenate([np.linspace(0, 5, 26), np.linspace(6, 300, 30)])
    e1 = ml(-x, a)
    e2 = ml(-x, a, a)
    ok &= bool(np.all(e1 > 0) and np.all(np.diff(e1) < 0) and np.all(e2 >= 0))
    ok &= bool(np.all(np.diff(e2) < 0) and np.all(e2 <= 1.0 / Gamma(a) + 1e-14))
check("5. E_a(-x) > 0, decreasing; 0 <= E_{a,a}(-x) <= 1/Gamma(a), decreasing", ok)

# 6. Large positive argument:  E_a(z) = (1/a) exp(z^(1/a)) - 1/(z Gamma(1-a)) + O(z^-2)
#    Computed in 120-digit arithmetic so the huge exponential can be subtracted.
worst = 0.0
for a in [0.5, 0.7, 0.9]:
    for z in [10.0, 20.0]:
        with mp.workdps(int(z ** (1 / a) / 2.3) + 60):
            series = mp.nsum(lambda kk: mp.mpf(z) ** kk * mp.rgamma(mp.mpf(a) * kk + 1), [0, mp.inf])
            remainder = series - mp.exp(mp.mpf(z) ** (1 / mp.mpf(a))) / a
            # rgamma = 1/Gamma (zero at the poles, e.g. Gamma(0) when a = 1/2)
            predicted = -mp.rgamma(1 - mp.mpf(a)) / z - mp.rgamma(1 - 2 * mp.mpf(a)) / z ** 2
            worst = max(worst, float(abs(remainder - predicted) / abs(predicted)))
check("6. asymptotic expansion of E_a(z) for large z > 0", worst < 0.05,
      f"max relative error {worst:.1e} (next term is O(z^-3))")

# 7. Large negative argument: E_a(-x) ~ 1/(x Gamma(1-a))  (algebraic, not exponential, decay)
worst = 0.0
for a in [0.3, 0.6, 0.9]:
    x = 1e4
    worst = max(worst, abs(ml(-x, a) * x * Gamma(1 - a) - 1.0))
check("7. algebraic tail E_a(-x) ~ 1/(x Gamma(1-a))", worst < 1e-3, f"max relative error {worst:.1e}")

# ===========================================================================
# B. Caputo SEIR: well-posedness (Theorem 3)
# ===========================================================================

def project_onto_omega(x, N):
    """Closest point of Omega = {x >= 0, sum(x) <= N} to x (Euclidean projection)."""
    y = np.maximum(x, 0.0)
    if y.sum() <= N:
        return y
    u = np.sort(x)[::-1]                 # project onto {x >= 0, sum = N}
    css = np.cumsum(u) - N
    idx = np.arange(1, len(x) + 1)
    rho = idx[u - css / idx > 0][-1]
    tau = css[rho - 1] / rho
    return np.maximum(x - tau, 0.0)


# 8. Key inequality of the proof: for x >= 0 and p = proj(x),
#    p <= x componentwise, and G(x) = f(p) + k x >= 0 with k = b + s + g + m.
ok, n_tests = True, 20000
for _ in range(n_tests):
    par = random_params(mu_positive=rng.random() < 0.5)
    a = rng.choice(ALPHAS)
    N = par["N"]
    x = rng.uniform(0, 1.5 * N, size=4) * (rng.random(4) < 0.8)   # some exact zeros
    pr = project_onto_omega(x, N)
    b, s, g, m = M.rates(par, a)
    G = M.caputo_seir_rhs(pr, par, a) + (b + s + g + m) * x
    ok &= bool(np.all(pr <= x + 1e-9 * N) and np.all(G >= -1e-9 * N))
check("8. Theorem 3 key step: p <= x and f(p) + kx >= 0 for x >= 0", ok, f"{n_tests:,} random cases")

# 9. Conservation: f_S + f_E + f_I + f_R = mu^a (N - (S+E+I+R))
worst = 0.0
for _ in range(2000):
    par = random_params(True)
    a = rng.choice(ALPHAS)
    x = rng.uniform(0, par["N"], 4)
    lhs = M.caputo_seir_rhs(x, par, a).sum()
    rhs = par["mu"] ** a * (par["N"] - x.sum())
    worst = max(worst, abs(lhs - rhs) / par["N"])
check("9. compartments sum: D^a(S+E+I+R) = mu^a (N - total)", worst < 1e-12, f"max error {worst:.1e}")

# ===========================================================================
# C. Reproduction number, equilibria, stability (Proposition 4, Theorem 5)
# ===========================================================================

# 10. R0 = spectral radius of F V^(-1) (next-generation matrix)
worst = 0.0
for _ in range(2000):
    par = random_params(rng.random() < 0.5)
    a = rng.choice(ALPHAS)
    b, s, g, m = M.rates(par, a)
    F = np.array([[0.0, b], [0.0, 0.0]])
    V = np.array([[s + m, 0.0], [-s, g + m]])
    rho = max(abs(np.linalg.eigvals(F @ np.linalg.inv(V))))
    worst = max(worst, abs(rho - M.r0_caputo_seir(par, a)) / rho)
check("10. R0 formula equals the next-generation spectral radius", worst < 1e-12, f"max error {worst:.1e}")

# 11. With mu = 0, R0 = (beta/gamma)^alpha
worst = 0.0
for _ in range(500):
    par = random_params(False)
    a = rng.choice(ALPHAS)
    worst = max(worst, abs(M.r0_caputo_seir(par, a) - (par["beta"] / par["gamma"]) ** a))
check("11. outbreak model (mu = 0): R0 = (beta/gamma)^alpha", worst < 1e-12)

# 12. Disease-free equilibrium: Matignon-stable exactly when R0 < 1, for every alpha
ok, n = True, 0
for _ in range(3000):
    par = random_params(True)
    a = rng.choice(ALPHAS)
    R0 = M.r0_caputo_seir(par, a)
    if abs(R0 - 1) < 1e-6:
        continue
    eig = np.linalg.eigvals(M.caputo_seir_jacobian(M.disease_free_equilibrium(par), par, a))
    ok &= M.matignon_stable(eig, a) == (R0 < 1)
    n += 1
check("12. DFE stable iff R0 < 1 (Matignon), all alpha", ok, f"{n:,} random cases")

# 13-15. Endemic equilibrium (R0 > 1, mu > 0)
worst_f, ok_poly, ok_stab, n = 0.0, True, True, 0
while n < 3000:
    par = random_params(True)
    a = rng.choice(ALPHAS)
    if M.r0_caputo_seir(par, a) <= 1.0:
        continue
    n += 1
    xs = M.endemic_equilibrium(par, a)
    worst_f = max(worst_f, np.abs(M.caputo_seir_rhs(xs, par, a)).max() / par["N"])
    J = M.caputo_seir_jacobian(xs, par, a)
    # characteristic polynomial of the (S, E, I) block: l^3 + A1 l^2 + A2 l + A3
    b, s, g, m = M.rates(par, a)
    iota = b * xs[2] / par["N"]                      # force of infection at equilibrium
    P, Q, Rr = iota + m, s + m, g + m
    A1, A2, A3 = P + Q + Rr, P * (Q + Rr), Q * Rr * iota
    coeffs = np.poly(J[:3, :3])                      # [1, A1, A2, A3] numerically
    ok_poly &= np.allclose(coeffs, [1.0, A1, A2, A3], rtol=1e-8, atol=1e-12 * max(1.0, A1 * A2))
    ok_stab &= (A1 > 0) and (A3 > 0) and (A1 * A2 > A3)
    eig = np.linalg.eigvals(J)
    ok_stab &= bool(np.all(eig.real < 0)) and all(M.matignon_stable(eig, al) for al in ALPHAS)
check("13. endemic equilibrium formula solves f(x*) = 0", worst_f < 1e-10, f"max |f|/N {worst_f:.1e}")
check("14. characteristic polynomial coefficients A1, A2, A3 are correct", ok_poly, f"{n:,} random cases")
check("15. Routh-Hurwitz holds, so endemic equilibrium is stable for all alpha", ok_stab)

# ===========================================================================
# D. Early phase and identifiability (Propositions 6 and 10)
# ===========================================================================

# 16. Exact linear solution: (E, I)(t) satisfies the Volterra form x = x0 + I^a(A x)
worst = 0.0
par = {"beta": 3.0, "sigma": 1.5, "gamma": 1.0, "N": 1e5, "mu": 0.0}
for a in [0.6, 0.9]:
    A = M.early_matrix(par, a)
    for t in [0.7, 2.0]:
        sol_t = M.early_solution(par, a, [t])[0]
        integrand = lambda s, i: (A @ M.early_solution(par, a, [s])[0])[i]
        Ix = np.array([quad(integrand, 0, t, args=(i,), weight="alg", wvar=(0.0, a - 1.0), limit=200)[0]
                       for i in range(2)]) / Gamma(a)
        worst = max(worst, np.abs(sol_t - (np.array([1.0, 0.0]) + Ix)).max())
check("16. Proposition 6 solution satisfies the fractional system", worst < 1e-8, f"max error {worst:.1e}")

# 17. Proposition 10(a): beta(alpha) gives growth rate r; R0 formula; for every alpha
worst = 0.0
for _ in range(3000):
    r, sig, gam = rng.uniform(0.01, 3), rng.uniform(0.2, 5), rng.uniform(0.2, 5)
    a = rng.uniform(0.1, 1.0)
    bet = M.beta_for_growth_rate(r, sig, gam, a)
    p_ = {"beta": bet, "sigma": sig, "gamma": gam, "N": 1.0, "mu": 0.0}
    worst = max(worst, abs(M.growth_rate_caputo(p_, a) - r) / r,
                abs(M.r0_caputo_seir(p_, a) - M.r0_from_growth_rate_caputo(r, sig, gam, a))
                / M.r0_caputo_seir(p_, a))
check("17. Prop 10(a): one beta per alpha reproduces the same growth rate r", worst < 1e-10,
      f"max error {worst:.1e}")

# 18. Prop 10(a): R0 decreases in alpha when r < min(sigma, gamma); increases when r > max
ok = True
for _ in range(2000):
    sig, gam = rng.uniform(0.2, 5), rng.uniform(0.2, 5)
    grid = np.linspace(0.05, 1.0, 60)
    r_lo = rng.uniform(0.01, 0.99) * min(sig, gam)
    r_hi = rng.uniform(1.01, 5) * max(sig, gam)
    ok &= bool(np.all(np.diff(M.r0_from_growth_rate_caputo(r_lo, sig, gam, grid)) < 0))
    ok &= bool(np.all(np.diff(M.r0_from_growth_rate_caputo(r_hi, sig, gam, grid)) > 0))
check("18. Prop 10(a): implied R0 falls as alpha rises when r < min(sigma, gamma)", ok)

# 19. Integer-order Erlang SEIR: beta = gamma R0(r) makes r the dominant eigenvalue
worst = 0.0
for _ in range(500):
    r, sig, gam = rng.uniform(0.01, 2), rng.uniform(0.2, 5), rng.uniform(0.2, 5)
    mE, mI = int(rng.integers(1, 6)), int(rng.integers(1, 6))
    bet = gam * M.r0_from_growth_rate_erlang(r, sig, gam, mE, mI)
    lead = max(np.linalg.eigvals(M.erlang_early_matrix(bet, sig, gam, mE, mI)).real)
    worst = max(worst, abs(lead - r) / r)
check("19. Erlang SEIR growth-rate formula (Euler-Lotka)", worst < 1e-8, f"max error {worst:.1e}")

# 20. AHM model: Laplace transform of phi, R0, growth rate, vanishing threshold
ok, worst = True, 0.0
for a, tau, s in [(0.6, 1.0, 0.7), (0.85, 2.0, 0.3)]:
    num, _ = quad(lambda t: math.exp(-s * t) * ml(-(t / tau) ** a, a), 0, np.inf, limit=400)
    worst = max(worst, abs(num - M.ahm_phi_hat(s, tau, a)) / num)
ok &= worst < 1e-7
for _ in range(500):
    a, tau, gc = rng.uniform(0.2, 0.99), rng.uniform(0.3, 3), rng.uniform(0.01, 1)
    wS0 = rng.uniform(0.05, 5)
    r = M.ahm_growth_rate(wS0, tau, a, gc)
    q = gc + r
    if q > 1e-300:
        ok &= abs(wS0 * M.ahm_phi_hat(q, tau, a) - 1.0) < 1e-9          # Euler-Lotka holds
    else:
        ok &= M.ahm_r0(wS0, tau, a, gc) < 1   # root below 1e-300: dies out at rate gamma_c
    ok &= (r > 0) == (M.ahm_r0(wS0, tau, a, gc) > 1)                       # threshold is R0 = 1
    ok &= abs(M.ahm_transmission_for_growth_rate(r, tau, a, gc) - wS0) < 1e-8 * wS0 if r > 0 else True
for a in [0.3, 0.6, 0.9]:
    ok &= M.ahm_growth_rate(1e-3, 1.0, a, 0.0) > 0                          # tiny transmission still grows
check("20. AHM: Laplace transform, R0 threshold, growth rate; no threshold if gamma_c = 0",
      ok, f"Laplace error {worst:.1e}")

# 21. Remark 3: in the Caputo model, people entering I later are retained by
#     k_I(u) = u^(a-1) E_{a,a}(-(gamma u)^a), whose integral is 1/gamma^a (finite),
#     unlike the AHM survival function, whose integral is infinite.
worst = 0.0
for a, gam in [(0.5, 1.0), (0.7, 0.4), (0.9, 2.5)]:
    k_I = lambda u: ml(-(gam * u) ** a, a, a)
    head, _ = quad(k_I, 0, 1.0, weight="alg", wvar=(a - 1.0, 0.0), limit=200)
    tail, _ = quad(lambda u: u ** (a - 1.0) * k_I(u), 1.0, np.inf, limit=400)
    worst = max(worst, abs((head + tail) * gam ** a - 1.0))
check("21. Caputo retention kernel k_I integrates to 1/gamma^alpha", worst < 1e-6,
      f"max relative error {worst:.1e}")

print()
if all(results):
    print(f"All {len(results)} checks passed.")
else:
    print(f"{results.count(False)} of {len(results)} checks FAILED.")
    sys.exit(1)
