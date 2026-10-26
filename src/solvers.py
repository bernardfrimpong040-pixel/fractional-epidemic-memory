"""
solvers.py -- numerical solvers for the models of Section 3.

Caputo systems   D^a y = f(t, y),  y(0) = y0,  0 < a <= 1, on a uniform grid t_n = n h:
    solve_caputo(..., method="pece")  Adams-type predictor-corrector
                                      (Diethelm, Ford & Freed 2002; Garrappa 2018)
    solve_caputo(..., method="l1")    implicit L1 finite-difference scheme
AHM model (renewal form)                solve_ahm(...)
Integer-order models (ordinary ODEs)    solve_ode(...) with scipy's solve_ivp

Convenience wrappers for the paper's models:
    simulate_caputo_seir, simulate_erlang_seir, simulate_seir_beta_t

WHY TWO CAPUTO METHODS?
    They discretise the problem in different ways (one the integral form, one
    the derivative itself). If both give the same answer as the step h shrinks,
    that answer is very unlikely to be a numerical artefact.

COST
    Memory means every step looks back over the whole history, so N steps cost
    about N^2 operations. With daily steps over a year (N = 365) this takes
    milliseconds.
"""

import math

import numpy as np
from scipy.integrate import solve_ivp
from scipy.special import gamma as Gamma

import models as M
from mittag_leffler import ml_neg_fast


# ===========================================================================
# 1. Caputo solvers
# ===========================================================================

def solve_caputo(f, y0, alpha, T, h, method="pece", jac=None, corrector_iters=1,
                 newton_tol=1e-12, newton_max=30):
    """Solve D^alpha y = f(t, y), y(0) = y0 on [0, T] with step h.

    f(t, y) returns an array like y; jac(t, y) its Jacobian (L1 method only;
    a finite-difference Jacobian is used if jac is None).
    Returns (t, Y) with Y[n] the approximation of y(t_n).
    """
    N = _n_steps(T, h)
    t = h * np.arange(N + 1)
    y0 = np.atleast_1d(np.asarray(y0, dtype=float))
    Y = np.zeros((N + 1, y0.size))
    Y[0] = y0
    if method == "pece":
        _pece(f, Y, t, alpha, h, corrector_iters)
    elif method == "l1":
        _l1(f, Y, t, alpha, h, jac, newton_tol, newton_max)
    else:
        raise ValueError("method must be 'pece' or 'l1'")
    return t, Y


def _n_steps(T, h):
    """Number of steps T/h, which must be a whole number (else the end time would shift)."""
    N = T / h
    if abs(N - round(N)) > 1e-9 * max(1.0, N):
        raise ValueError(f"T/h = {N:g} is not a whole number; choose h that divides T")
    return int(round(N))


def _pece(f, Y, t, a, h, iters):
    """Predictor-corrector for the Volterra form  y = y0 + I^a f(t, y).

    The integral over each step is computed exactly for f replaced by
      - a constant (predictor, 'product rectangle' rule),
      - a straight line (corrector, 'product trapezoid' rule).
    Weights (Diethelm, Ford & Freed 2002), with k = n - j:
      predictor  b_k = (k+1)^a - k^a,         factor h^a / Gamma(a+1)
      corrector  c_k = (k+2)^(a+1) + k^(a+1) - 2 (k+1)^(a+1)   for j >= 1,
                 first weight  n^(a+1) - (n-a)(n+1)^a,   last weight 1,
                 factor h^a / Gamma(a+2)
    """
    N = len(t) - 1
    k = np.arange(N + 1, dtype=float)
    b = (k + 1) ** a - k ** a
    c = (k + 2) ** (a + 1) + k ** (a + 1) - 2 * (k + 1) ** (a + 1)
    pred_fac = h ** a / Gamma(a + 1)
    corr_fac = h ** a / Gamma(a + 2)
    F = np.zeros_like(Y)                       # F[j] = f(t_j, y_j)
    F[0] = f(t[0], Y[0])
    y0 = Y[0]
    for n in range(N):                         # compute y_{n+1}
        # predictor: sum_{j=0}^{n} b_{n-j} F_j
        yp = y0 + pred_fac * (b[n::-1] @ F[: n + 1])
        # history part of the corrector (everything except the new point)
        hist = (n ** (a + 1) - (n - a) * (n + 1) ** a) * F[0]
        if n >= 1:
            hist = hist + c[n - 1::-1] @ F[1: n + 1]
        y_new = yp
        for _ in range(iters):                 # usually one correction
            y_new = y0 + corr_fac * (f(t[n + 1], y_new) + hist)
        Y[n + 1] = y_new
        F[n + 1] = f(t[n + 1], y_new)


def _l1(f, Y, t, a, h, jac, tol, maxit):
    """Implicit L1 scheme: at each t_n, approximate the Caputo derivative by
         c * sum_{k=0}^{n-1} w_k (y_{n-k} - y_{n-k-1}),
         c = h^(-a) / Gamma(2-a),   w_k = (k+1)^(1-a) - k^(1-a)
    (y is treated as piecewise linear), set it equal to f(t_n, y_n), and solve
    for y_n with Newton's method. With a = 1 this is backward Euler.
    """
    N = len(t) - 1
    k = np.arange(N + 1, dtype=float)
    w = (k + 1) ** (1 - a) - k ** (1 - a)
    cfac = h ** (-a) / Gamma(2 - a)
    D = np.zeros_like(Y)                       # D[m] = y_m - y_{m-1}
    d = Y.shape[1]
    I_d = np.eye(d)
    for n in range(1, N + 1):
        hist = w[1:n][::-1] @ D[1:n] if n > 1 else np.zeros(d)   # sum_{k>=1} w_k D_{n-k}
        y = Y[n - 1].copy()                    # Newton starting guess
        for _ in range(maxit):
            g = cfac * (y - Y[n - 1] + hist) - f(t[n], y)
            Jf = jac(t[n], y) if jac is not None else _fd_jac(f, t[n], y)
            step = np.linalg.solve(cfac * I_d - Jf, -g)
            y = y + step
            if np.max(np.abs(step)) <= tol * max(1.0, np.max(np.abs(y))):
                break
        else:
            raise RuntimeError(f"L1 Newton iteration did not converge at t = {t[n]:g}")
        Y[n] = y
        D[n] = y - Y[n - 1]


def _fd_jac(f, t, y, eps=1e-7):
    """Finite-difference Jacobian (fallback when no exact Jacobian is given)."""
    f0 = f(t, y)
    J = np.zeros((f0.size, y.size))
    for i in range(y.size):
        dy = np.zeros_like(y)
        dy[i] = eps * max(1.0, abs(y[i]))
        J[:, i] = (f(t, y + dy) - f0) / dy[i]
    return J


PECE_STABILITY = 1.0   # use h |lambda|^(1/alpha) <= 1.0 (measured limit 1.6-2.0
                       # on the negative real axis, lower near the imaginary axis)


def _dfe_rates(p, alpha):
    """Eigenvalues of the (E, I) block of the Jacobian at the disease-free state."""
    b, s, g, m = M.rates(p, alpha)
    B = np.array([[-(s + m), b], [s, -(g + m)]])
    return np.linalg.eigvals(B).real, b, m


def pece_max_step(p, alpha):
    """Step used by the explicit PECE method for the Caputo SEIR.

    The method is stable for D^a y = lambda y (lambda < 0) while
    h |lambda|^(1/a) is below about 1.6-2.0 (07_check_solvers.py). As the
    fastest rate we take the larger of (i) the largest eigenvalue size of the
    (E, I) block of the Jacobian at the disease-free state and (ii) b + m, a
    heuristic bound for the S equation (whose rate is b I/N + m <= b + m).
    This is not a proof: simulate_caputo_seir also checks the output and
    halves the step if anything goes negative. Note the power 1/a: at small
    alpha even moderate rates force small steps.
    """
    eig, b, m = _dfe_rates(p, alpha)
    lam = max(np.max(np.abs(eig)), b + m)
    return PECE_STABILITY / lam ** (1.0 / alpha)


def l1_max_step(p, alpha):
    """Step used by the implicit L1 method for the Caputo SEIR.

    L1 stays bounded at any step, but it keeps the compartments non-negative
    only if c = h^(-a)/Gamma(2-a) exceeds the growth eigenvalue lambda1 at the
    disease-free state (at a = 1 this is backward Euler's condition h lambda1 < 1).
    We require c >= 2 lambda1, i.e. h <= (2 lambda1 Gamma(2-a))^(-1/a).
    """
    eig, _, _ = _dfe_rates(p, alpha)
    lam1 = max(eig.max(), 0.0)
    if lam1 == 0.0:
        return np.inf
    return (2.0 * lam1 * Gamma(2.0 - alpha)) ** (-1.0 / alpha)


def simulate_caputo_seir(p, alpha, y0, T, h, method="pece", substeps=None):
    """Caputo SEIR (Section 3.2 of the paper). Returns t and columns S, E, I, R.

    Results are returned on the grid t = 0, h, 2h, ..., T. Internally the
    solver uses h / k, with k the smallest whole number that satisfies the
    step rule of the method (pece_max_step or l1_max_step), and keeps every
    k-th value. Pass substeps=k to fix k yourself (useful when fitting, so
    that k does not jump as the parameters change).

    Safety net: if the result contains a negative or non-finite value, the
    run is repeated with twice as many sub-steps (up to 4 times).
    """
    f = lambda t, y: M.caputo_seir_rhs(y, p, alpha)
    jac = lambda t, y: M.caputo_seir_jacobian(y, p, alpha)
    if substeps is None:
        hmax = pece_max_step(p, alpha) if method == "pece" else l1_max_step(p, alpha)
        k = max(1, int(math.ceil(h / hmax - 1e-12)))
    else:
        k = int(substeps)
    N = p["N"]
    for _ in range(5):
        try:
            t, Y = solve_caputo(f, y0, alpha, T, h / k, method=method, jac=jac)
            if np.all(np.isfinite(Y)) and Y.min() >= -1e-9 * N:
                return t[::k], Y[::k]
        except (RuntimeError, FloatingPointError, np.linalg.LinAlgError):
            pass
        k *= 2
    raise RuntimeError("Caputo SEIR solution failed even with small steps")


# ===========================================================================
# 2. AHM fractional-recovery model (renewal form, Section 3.5)
# ===========================================================================

def ahm_survival(t, tau, alpha, gamma_c=0.0):
    """Phi(t) = E_a(-(t/tau)^a) exp(-gamma_c t): probability still infectious at age t."""
    t = np.asarray(t, dtype=float)
    if 0.9999 < alpha < 1.0:
        raise ValueError("use alpha <= 0.9999 or alpha = 1 (fast E_alpha is validated to 0.9999)")
    phi = np.exp(-t / tau) if alpha == 1.0 else ml_neg_fast((t / tau) ** alpha, alpha)
    return phi * np.exp(-gamma_c * t)


def solve_ahm(omega, tau, alpha, gamma_c, S0, I0, T, h, newton_tol=1e-13):
    """Solve the AHM model on the grid t = 0, h, ..., T (see _solve_ahm_grid).

    Two rules set the internal step h / k (every k-th value is returned):
      - stability/positivity: the implicit trapezoid step divides by about
        1 - h omega S / 2, so we keep h omega S0 / 2 <= 0.25;
      - accuracy: the survival function drops steeply right after infection
        (like 1 - (t/tau)^alpha); the scheme converges with order about
        1 + alpha, and 80 steps per tau keep the error near 0.2% of the peak.
    """
    hmax = min(0.5 / (omega * S0), tau / 80.0)
    k = max(1, int(math.ceil(h / hmax - 1e-12)))
    t, out = _solve_ahm_grid(omega, tau, alpha, gamma_c, S0, I0, T, h / k, newton_tol)
    if out["I"].min() < -1e-9 * (S0 + I0):
        raise RuntimeError("AHM solution went negative")
    return t[::k], {key: v[::k] for key, v in out.items()}


def _solve_ahm_grid(omega, tau, alpha, gamma_c, S0, I0, T, h, newton_tol=1e-13):
    """Solve  S' = -omega S I,  I(t) = I0 Phi(t) + int_0^t Phi(t-u) omega S(u) I(u) du.

    Method: write S = S0 exp(-omega C) with C = int_0^t I, and use the
    trapezoid rule for both integrals on the grid t_n = n h. The new value
    I_n appears on both sides (through the u = t_n end of the integral and
    through C_n), so each step solves one scalar equation by Newton's method.

    Returns t and a dict with S, I, R and cumulative infections (N - S).
    """
    N = _n_steps(T, h)
    t = h * np.arange(N + 1)
    Phi = ahm_survival(t, tau, alpha, gamma_c)
    I = np.zeros(N + 1)
    C = np.zeros(N + 1)
    J = np.zeros(N + 1)                       # J = omega S I, new infections per unit time
    I[0] = I0
    J[0] = omega * S0 * I0
    for n in range(1, N + 1):
        # trapezoid: int_0^{t_n} Phi(t_n - u) J(u) du
        #   = h [ Phi_n J_0 / 2 + sum_{j=1}^{n-1} Phi_{n-j} J_j + Phi_0 J_n / 2 ]
        known = I0 * Phi[n] + h * (0.5 * Phi[n] * J[0] + (Phi[n - 1:0:-1] @ J[1:n] if n > 1 else 0.0))
        x = I[n - 1]                          # Newton for I_n
        for _ in range(50):
            Cn = C[n - 1] + 0.5 * h * (I[n - 1] + x)
            Sn = S0 * math.exp(-omega * Cn)
            g = x - known - 0.5 * h * Phi[0] * omega * Sn * x
            dg = 1.0 - 0.5 * h * Phi[0] * omega * Sn * (1.0 - omega * 0.5 * h * x)
            step = -g / dg
            x += step
            if abs(step) <= newton_tol * max(1.0, abs(x)):
                break
        else:
            raise RuntimeError(f"AHM Newton iteration did not converge at t = {t[n]:g}")
        I[n] = x
        C[n] = C[n - 1] + 0.5 * h * (I[n - 1] + x)
        J[n] = omega * S0 * math.exp(-omega * C[n]) * x
    S = S0 * np.exp(-omega * C)
    Ntot = S0 + I0
    return t, {"S": S, "I": I, "R": Ntot - S - I, "cumulative": Ntot - S, "incidence": J}


# ===========================================================================
# 3. Integer-order models (ordinary differential equations)
# ===========================================================================

def solve_ode(rhs, y0, T, t_eval, rtol=1e-10, atol=1e-12):
    """Solve y' = rhs(t, y) with an adaptive Runge-Kutta method (DOP853)."""
    sol = solve_ivp(rhs, (0.0, T), np.asarray(y0, dtype=float), method="DOP853",
                    t_eval=t_eval, rtol=rtol, atol=atol)
    if not sol.success:
        raise RuntimeError(sol.message)
    return sol.t, sol.y.T


def simulate_erlang_seir(beta, sigma, gamma, N, m_E, m_I, E0, T, t_eval, beta_fn=None):
    """SEIR with m_E latent and m_I infectious stages (Section 3.6 (a));
    optional time-varying transmission beta_fn(t) (Section 3.6 (b)).

    State: S, E_1..E_mE, I_1..I_mI, R, CumI (cumulative entries into I).
    Returns t and a dict with S, E (total), I (total), R, cumulative.
    """
    def rhs(t, y):
        S = y[0]
        E = y[1:1 + m_E]
        I = y[1 + m_E:1 + m_E + m_I]
        b = beta_fn(t) if beta_fn is not None else beta
        inf = b * S * I.sum() / N
        dE = np.empty(m_E)
        dE[0] = inf - m_E * sigma * E[0]
        dE[1:] = m_E * sigma * (E[:-1] - E[1:])
        dI = np.empty(m_I)
        entry = m_E * sigma * E[-1]
        dI[0] = entry - m_I * gamma * I[0]
        dI[1:] = m_I * gamma * (I[:-1] - I[1:])
        dR = m_I * gamma * I[-1]
        return np.concatenate([[-inf], dE, dI, [dR, entry]])

    y0 = np.zeros(1 + m_E + m_I + 2)
    y0[0] = N - E0
    y0[1] = E0
    tt, Y = solve_ode(rhs, y0, T, t_eval)
    return tt, {"S": Y[:, 0], "E": Y[:, 1:1 + m_E].sum(axis=1),
                "I": Y[:, 1 + m_E:1 + m_E + m_I].sum(axis=1),
                "R": Y[:, -2], "cumulative": Y[:, -1]}


def simulate_seir_beta_t(beta0, rho, kappa, sigma, gamma, N, E0, T, t_eval):
    """Ordinary SEIR with beta(t) = beta0 (rho + (1 - rho) e^(-kappa t))."""
    return simulate_erlang_seir(beta0, sigma, gamma, N, 1, 1, E0, T, t_eval,
                                beta_fn=lambda t: M.beta_declining(t, beta0, rho, kappa))


def apply_reporting_delay(weekly_cases, mean, shape, rho_rep=1.0):
    """Expected reported weekly cases: rho_rep * sum_j h_j c_{k-j}  (Section 3.6 (c))."""
    c = np.asarray(weekly_cases, dtype=float)
    h = M.gamma_delay_weights(mean, shape, len(c))
    return rho_rep * np.convolve(c, h)[: len(c)]
