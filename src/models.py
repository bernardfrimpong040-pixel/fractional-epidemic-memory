"""
models.py -- the models of the paper and the formulas proved in Section 3.

Nothing runs here; other scripts import these functions. Every formula below
is checked numerically in 05_check_theory.py.

Notation (all rates are per unit time, e.g. per week)
    beta   transmission rate          sigma  1 / mean latent period
    gamma  1 / mean infectious period mu     birth = death rate (0 for an outbreak)
    N      population size            alpha  fractional order, 0 < alpha <= 1

DIMENSIONAL CONSISTENCY
    A Caputo derivative of order alpha has units  (quantity) / time^alpha.
    So every rate in a Caputo model must have units 1/time^alpha. We write
    each one as (ordinary rate)^alpha: beta^alpha, sigma^alpha, ... . Then
    alpha = 1 gives back the ordinary model with the SAME beta, sigma, gamma.
"""

import math

import numpy as np
from scipy.optimize import brentq

# ===========================================================================
# 1. Caputo fractional SEIR  (Section 3.2)
# ===========================================================================
#   D^a S = mu^a N - beta^a S I / N - mu^a S
#   D^a E = beta^a S I / N - (sigma^a + mu^a) E
#   D^a I = sigma^a E - (gamma^a + mu^a) I
#   D^a R = gamma^a I - mu^a R


def rates(p, alpha):
    """Return (b, s, g, m) = (beta^a, sigma^a, gamma^a, mu^a)."""
    return (p["beta"] ** alpha, p["sigma"] ** alpha,
            p["gamma"] ** alpha, p.get("mu", 0.0) ** alpha)


def caputo_seir_rhs(x, p, alpha):
    """Right-hand side f(x) of the Caputo SEIR system, x = (S, E, I, R)."""
    S, E, I, R = x
    b, s, g, m = rates(p, alpha)
    N = p["N"]
    infection = b * S * I / N
    return np.array([
        m * N - infection - m * S,
        infection - (s + m) * E,
        s * E - (g + m) * I,
        g * I - m * R,
    ])


def caputo_seir_jacobian(x, p, alpha):
    """Jacobian matrix of f at x (4 x 4)."""
    S, E, I, R = x
    b, s, g, m = rates(p, alpha)
    N = p["N"]
    return np.array([
        [-b * I / N - m, 0.0, -b * S / N, 0.0],
        [b * I / N, -(s + m), b * S / N, 0.0],
        [0.0, s, -(g + m), 0.0],
        [0.0, 0.0, g, -m],
    ])


def r0_caputo_seir(p, alpha):
    """Basic reproduction number  R0 = beta^a sigma^a / ((sigma^a+mu^a)(gamma^a+mu^a)).

    With mu = 0 this is (beta/gamma)^alpha.
    """
    b, s, g, m = rates(p, alpha)
    return b * s / ((s + m) * (g + m))


def disease_free_equilibrium(p):
    return np.array([p["N"], 0.0, 0.0, 0.0])


def endemic_equilibrium(p, alpha):
    """Endemic equilibrium (exists when mu > 0 and R0 > 1), Proposition 4."""
    b, s, g, m = rates(p, alpha)
    N = p["N"]
    R0 = r0_caputo_seir(p, alpha)
    S = N / R0
    I = m * N * (R0 - 1.0) / b
    E = (g + m) * I / s
    R = g * I / m
    return np.array([S, E, I, R])


def matignon_stable(eigenvalues, alpha):
    """Matignon's condition: every eigenvalue satisfies |arg(lambda)| > alpha*pi/2."""
    return bool(np.all(np.abs(np.angle(eigenvalues)) > alpha * math.pi / 2))


# ===========================================================================
# 2. Early (linear) phase of the outbreak model, mu = 0  (Section 3.4)
# ===========================================================================
#   While S is close to N:  D^a (E, I) = A (E, I)  with
#   A = [[-s, b], [s, -g]]   (b = beta^a, s = sigma^a, g = gamma^a)


def early_matrix(p, alpha):
    b, s, g, _ = rates(p, alpha)
    return np.array([[-s, b], [s, -g]])


def early_eigenvalues(p, alpha):
    """Eigenvalues lambda1 > lambda2 of A (always real: b*s > 0)."""
    b, s, g, _ = rates(p, alpha)
    disc = math.sqrt((s - g) ** 2 + 4.0 * b * s)
    return (-(s + g) + disc) / 2.0, (-(s + g) - disc) / 2.0


def growth_rate_caputo(p, alpha):
    """Long-run exponential growth rate r = lambda1^(1/alpha) (needs lambda1 > 0)."""
    lam1, _ = early_eigenvalues(p, alpha)
    if lam1 <= 0:
        raise ValueError("no growth: R0 <= 1")
    return lam1 ** (1.0 / alpha)


def early_solution(p, alpha, t, x0=(1.0, 0.0)):
    """Exact solution of the linear early phase, Proposition 6:
        (E, I)(t) = c1 v1 E_a(lambda1 t^a) + c2 v2 E_a(lambda2 t^a),
    where v1, v2 are eigenvectors of A and c = V^(-1) x0.
    Returns an array of shape (len(t), 2) with columns E and I.
    """
    from mittag_leffler import ml
    A = early_matrix(p, alpha)
    lam, V = np.linalg.eig(A)
    order = np.argsort(-lam.real)              # lambda1 (largest) first
    lam, V = lam.real[order], V.real[:, order]
    c = np.linalg.solve(V, np.asarray(x0, dtype=float))
    t = np.asarray(t, dtype=float)
    out = np.zeros((len(t), 2))
    for i in range(2):
        out += np.outer(ml(lam[i] * t ** alpha, alpha), c[i] * V[:, i])
    return out


def beta_for_growth_rate(r, sigma, gamma, alpha):
    """Proposition 10(a): the unique beta giving growth rate r, for this alpha.

    From  b s = (lambda + s)(lambda + g)  with lambda = r^a:
        beta = [ (r^a + sigma^a)(r^a + gamma^a) / sigma^a ]^(1/a).
    """
    ra, sa, ga = r ** alpha, sigma ** alpha, gamma ** alpha
    return ((ra + sa) * (ra + ga) / sa) ** (1.0 / alpha)


def r0_from_growth_rate_caputo(r, sigma, gamma, alpha):
    """Proposition 10(a): R0 implied by growth rate r:  (1 + (r/sigma)^a)(1 + (r/gamma)^a)."""
    return (1.0 + (r / sigma) ** alpha) * (1.0 + (r / gamma) ** alpha)


# ===========================================================================
# 3. Integer-order alternatives  (Section 3.6)
# ===========================================================================

def r0_from_growth_rate_erlang(r, sigma, gamma, m_E, m_I):
    """SEIR with Erlang latent (m_E stages) and infectious (m_I stages) periods.

    Euler-Lotka equation (proof in Section 3.6):
        R0 = (r/gamma) (1 + r/(m_E sigma))^m_E / (1 - (1 + r/(m_I gamma))^(-m_I)).
    m_E = m_I = 1 gives the exponential case (1 + r/sigma)(1 + r/gamma).
    """
    latent = (1.0 + r / (m_E * sigma)) ** m_E
    infectious = 1.0 - (1.0 + r / (m_I * gamma)) ** (-m_I)
    return (r / gamma) * latent / infectious


def erlang_early_matrix(beta, sigma, gamma, m_E, m_I):
    """Linear early-phase matrix of the Erlang SEIR (stages E_1..E_mE, I_1..I_mI).

    New infections = beta * (I_1 + ... + I_mI), entering E_1.
    Each latent stage is left at rate m_E*sigma, each infectious stage at m_I*gamma.
    """
    n = m_E + m_I
    A = np.zeros((n, n))
    for j in range(m_E):
        A[j, j] = -m_E * sigma
        if j > 0:
            A[j, j - 1] = m_E * sigma
    A[m_E, m_E - 1] = m_E * sigma            # last latent stage -> first infectious stage
    for j in range(m_E, n):
        A[j, j] = -m_I * gamma
        if j > m_E:
            A[j, j - 1] = m_I * gamma
    A[0, m_E:] += beta                        # infection from all infectious stages
    return A


def beta_for_growth_rate_hetero(r, sigma, gammas, fractions):
    """SEIR with heterogeneous recovery (fraction q_j recovers at rate g_j):
    the beta giving growth rate r.

    Euler-Lotka (same argument as Proposition 9): with an exponential latent
    period and a mixture of exponential infectious periods,
        1 = beta * sigma/(sigma + r) * sum_j q_j / (g_j + r),
    so  beta = (sigma + r) / sigma / sum_j q_j / (g_j + r).
    R0 = beta * (mean infectious period) = beta * sum_j q_j / g_j.
    """
    g = np.asarray(gammas, dtype=float)
    q = np.asarray(fractions, dtype=float)
    return (sigma + r) / sigma / np.sum(q / (g + r))


def beta_declining(t, beta0, rho, kappa):
    """Time-varying transmission: beta(t) = beta0 * (rho + (1 - rho) e^(-kappa t))."""
    return beta0 * (rho + (1.0 - rho) * np.exp(-kappa * np.asarray(t)))


def gamma_delay_weights(mean, shape, n):
    """Weekly reporting-delay probabilities h_0, ..., h_{n-1} from a gamma distribution.

    h_j = P(j <= delay < j+1) for a gamma delay with this mean and shape;
    the last weight absorbs the remaining tail so the weights sum to 1.
    """
    from scipy.stats import gamma as gamma_dist
    scale = mean / shape
    edges = gamma_dist.cdf(np.arange(n + 1), a=shape, scale=scale)
    h = np.diff(edges)
    h[-1] += 1.0 - edges[-1]
    return h


# ===========================================================================
# 4. Angstmann-Henry-McGann (AHM) fractional-recovery SIR  (Section 3.5)
# ===========================================================================
# Infectious periods have survival function  phi(t) = E_a(-(t/tau)^a)
# (Mittag-Leffler distribution). An optional competing removal at rate
# gamma_c (isolation or death) multiplies survival by exp(-gamma_c t).
# Laplace transform:  phi_hat(s) = s^(a-1) / (s^a + tau^(-a)).


def ahm_phi_hat(s, tau, alpha):
    return s ** (alpha - 1.0) / (s ** alpha + tau ** (-alpha))


def ahm_mean_infectious_period(tau, alpha, gamma_c):
    """Mean time infectious = phi_hat(gamma_c) = 1/(gamma_c + tau^-a gamma_c^(1-a)).

    Infinite when gamma_c = 0 and alpha < 1 (heavy tail).
    """
    if gamma_c == 0.0:
        return tau if alpha == 1.0 else math.inf
    return ahm_phi_hat(gamma_c, tau, alpha)


def ahm_r0(omega_S0, tau, alpha, gamma_c):
    """R0 = omega S0 x mean infectious period  (Proposition 8)."""
    return omega_S0 * ahm_mean_infectious_period(tau, alpha, gamma_c)


def ahm_growth_rate(omega_S0, tau, alpha, gamma_c=0.0):
    """Growth rate r solving  1 = omega S0 phi_hat(gamma_c + r)  (Proposition 8).

    Writing q = gamma_c + r: omega S0 q^(a-1) = q^a + tau^(-a). The left side
    falls and the right side rises in q, so there is exactly one root q*.
    Returns r = q* - gamma_c (negative means the outbreak dies out).

    The root can be astronomically small (alpha near 1, weak transmission),
    so we solve for y = ln q, using the log of both sides:
        h(y) = ln(omega S0) + (a-1) y - ln(e^(a y) + tau^(-a)),
    which decreases from +infinity to -infinity.
    """
    if alpha == 1.0:
        return omega_S0 - 1.0 / tau - gamma_c
    log_rhs = lambda y: np.logaddexp(alpha * y, -alpha * math.log(tau))
    h = lambda y: math.log(omega_S0) + (alpha - 1.0) * y - log_rhs(y)
    lo, hi = -1.0, 1.0
    while h(lo) <= 0:
        lo *= 2.0
    while h(hi) >= 0:
        hi *= 2.0
    y = brentq(h, lo, hi, xtol=1e-14, rtol=1e-15, maxiter=1000)
    return math.exp(y) - gamma_c


def ahm_transmission_for_growth_rate(r, tau, alpha, gamma_c=0.0):
    """Proposition 10(b): omega S0 = 1/phi_hat(gamma_c + r) = q + tau^-a q^(1-a), q = gamma_c + r."""
    q = gamma_c + r
    return q + tau ** (-alpha) * q ** (1.0 - alpha)
