"""
fitting.py -- maximum-likelihood fitting of the Caputo SEIR model to weekly
case counts, with a profile likelihood for the fractional order alpha.

Nothing runs here; 09_check_fitting.py and 10_synthetic_experiments.py
import these functions.

THE OBSERVATION MODEL
    Weekly reported cases y_1, ..., y_K are negative binomial with mean
        mu_k = A * m_k,   m_k = c(k) - c(k-1),   c = (I + R) / N,
    i.e. A times the fraction of the population that entered I during week k.
    A = rho * N combines the reporting fraction rho and population size N.
    (With frequency-dependent transmission the model is scale-free: rho and N
    only ever appear through their product, so we estimate A directly.)
    The negative binomial has variance mu + mu^2 / k_disp; small k_disp means
    very noisy data, k_disp -> infinity is Poisson.

    In the Caputo model the weekly increase m_k can be NEGATIVE (cumulative
    infections can fall; see Section 5 of the paper). Such weeks get the tiny
    mean M_FLOOR, so a fit that relies on them is heavily penalised.

PARAMETERS
    alpha        fractional order (profiled on a grid)
    beta         transmission rate        } optimised for each alpha
    e0 = E0 / N  initial exposed fraction }
    A, k_disp    scale and dispersion: maximised in a fast inner step
                 (no extra model solves needed)
    sigma, gamma latent and infectious rates: fixed (assumed known)

THE PROFILE LIKELIHOOD FOR ALPHA
    For each alpha on a grid, maximise the log-likelihood over all other
    parameters. That gives the curve l_p(alpha). Its maximum is alpha-hat;
    the 95% confidence interval is where 2 (l_max - l_p(alpha)) <= 3.84.
    The test of "no memory" (alpha = 1) uses LRT = 2 (l_max - l_p(1)).
    Because alpha = 1 sits on the edge of the allowed range, under the null
    hypothesis LRT follows a 50:50 mixture of 0 and chi-square(1)
    (Self & Liang 1987), so we reject alpha = 1 at the 5% level when
    LRT > 2.706 (not 3.84).
"""

import math
import time
import warnings

import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.optimize import brentq, minimize, minimize_scalar
from scipy.special import digamma, gammaln
from scipy.stats import chi2

import models as M
import solvers as S

M_FLOOR = 1e-12          # mean used for weeks where the model predicts <= 0
K_DISP_BOUNDS = (0.01, 1e4)
E0_BOUNDS = (1e-7, 1e-1)
R_MIN, R_MAX = 0.02, 1.0  # growth rates (per week) allowed in a fit
LRT_CRIT_BOUNDARY = 2.706  # 5% critical value of the 50:50 chi-square mixture
CI_CRIT = 3.841            # chi-square(1) 95% point, for the profile interval


# ===========================================================================
# 1. The model's weekly increments
# ===========================================================================

def caputo_weekly(alpha, beta, e0, K, sigma, gamma, method, k):
    """Weekly increases m_1..m_K of c = (I + R)/N for the Caputo SEIR.

    Solved on fractions (N = 1) with weekly output and k internal sub-steps
    per week, using `method` ('pece' or 'l1'). Returns None if the solver fails.
    """
    p = {"beta": beta, "sigma": sigma, "gamma": gamma, "N": 1.0, "mu": 0.0}
    try:
        with np.errstate(all="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _, Y = S.simulate_caputo_seir(p, alpha, [1.0 - e0, e0, 0.0, 0.0],
                                          float(K), 1.0, method, substeps=k)
    except (RuntimeError, ValueError, FloatingPointError, np.linalg.LinAlgError):
        return None
    c = Y[:, 2] + Y[:, 3]
    m = np.diff(c)
    return m if np.all(np.isfinite(m)) else None


def substep_table(alpha_grid, sigma, gamma, K, r_max=R_MAX, tol=2e-3,
                  candidates=(1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256)):
    """Choose a solver and number of sub-steps for each alpha.

    For each alpha we take the hardest case allowed in a fit (fastest growth
    r_max) and an easy one (r = 0.1), compute an accurate reference with PECE
    and 512 sub-steps, and find for each method the smallest k whose weekly
    increments are within tol (relative to the peak) of the reference in both
    cases. We then keep whichever method is faster at its k.
    Returns a list of dicts: alpha, method, k, max_error, seconds.
    """
    table = []
    for a in alpha_grid:
        cases = []
        for r in (r_max, 0.1):
            beta = M.beta_for_growth_rate(r, sigma, gamma, a)
            ref = caputo_weekly(a, beta, 1e-3, K, sigma, gamma, "pece", 512)
            cases.append((beta, ref))
        best = None
        for method in ("pece", "l1"):
            for k in candidates:
                errs = []
                for beta, ref in cases:
                    m = caputo_weekly(a, beta, 1e-3, K, sigma, gamma, method, k)
                    errs.append(np.inf if m is None else np.max(np.abs(m - ref)) / np.max(ref))
                if max(errs) < tol:
                    t0 = time.perf_counter()
                    caputo_weekly(a, cases[0][0], 1e-3, K, sigma, gamma, method, k)
                    secs = time.perf_counter() - t0
                    if best is None or secs < best["seconds"]:
                        best = {"alpha": float(a), "method": method, "k": int(k),
                                "max_error": float(max(errs)), "seconds": secs}
                    break
        if best is None:
            raise RuntimeError(f"no solver setting reached tolerance {tol} at alpha = {a}")
        table.append(best)
    return table


# ===========================================================================
# 2. Likelihoods
# ===========================================================================

def nb_loglik(y, mu, kd):
    """Negative binomial log-likelihood (mean mu, dispersion kd), summed."""
    return float(np.sum(gammaln(y + kd) - gammaln(kd) - gammaln(y + 1.0)
                        + kd * np.log(kd / (kd + mu)) + y * np.log(mu / (kd + mu))))


def best_scale_dispersion(y, m):
    """Maximise the NB log-likelihood over A (scale) and k_disp for fixed m.

    No model solve is needed, so this is fast. Works on u = log A and
    v = log k_disp with exact gradients:
        d l / d u = sum k (y - mu) / (k + mu)
        d l / d v = k * sum [ psi(y+k) - psi(k) + log(k/(k+mu)) + 1 - (y+k)/(k+mu) ]
    Returns (log-likelihood, A, k_disp).
    """
    m = np.maximum(m, M_FLOOR)
    u0 = math.log(max(y.sum(), 1.0) / m.sum())

    def negll(z):
        u, v = z
        A, kd = math.exp(u), math.exp(v)
        mu = A * m
        ll = nb_loglik(y, mu, kd)
        g_u = np.sum(kd * (y - mu) / (kd + mu))
        g_v = kd * np.sum(digamma(y + kd) - digamma(kd) + np.log(kd / (kd + mu))
                          + 1.0 - (y + kd) / (kd + mu))
        return -ll, -np.array([g_u, g_v])

    best = None
    for v0 in (math.log(10.0), math.log(1000.0)):          # two starts for k_disp
        res = minimize(negll, [u0, v0], jac=True, method="L-BFGS-B",
                       bounds=[(u0 - 15, u0 + 15),
                               (math.log(K_DISP_BOUNDS[0]), math.log(K_DISP_BOUNDS[1]))])
        if best is None or res.fun < best.fun:
            best = res
    return -best.fun, math.exp(best.x[0]), math.exp(best.x[1])


def expected_nb_loglik(mu_true, m, kd):
    """EXPECTED negative binomial log-likelihood, for data y ~ NB(mu_true, kd),
    maximised over the scale A (kd is held at its true value).

    In  l = sum [ lgamma(y+k) - lgamma(k) - lgamma(y+1) + k log(k/(k+mu)) + y log(mu/(k+mu)) ]
    the model enters only through the last two terms, which are LINEAR in y.
    So, up to a constant that does not depend on the model,
        E[l] = sum [ k log(k/(k+mu)) + mu_true log(mu/(k+mu)) ],   mu = A m.
    Maximising this over the model gives the pseudo-true parameters for
    negative binomial data with dispersion kd (White 1982).
    """
    m = np.maximum(m, M_FLOOR)
    u0 = math.log(mu_true.sum() / m.sum())

    def neg(u):
        mu = math.exp(u) * m
        return -float(np.sum(kd * np.log(kd / (kd + mu)) + mu_true * np.log(mu / (kd + mu))))

    res = minimize_scalar(neg, bounds=(u0 - 5, u0 + 5), method="bounded",
                          options={"xatol": 1e-10})
    return -res.fun, math.exp(res.x), kd


def poisson_quasi_loglik(y, m):
    """Poisson log-likelihood for (possibly non-integer) y, maximised over A.

    Used for the noise-free 'infinite data' fits: y is the expected count
    curve. The maximising scale is A = sum(y) / sum(m) (closed form).
    The constant -log(y!) is dropped (it does not depend on the parameters).
    """
    m = np.maximum(m, M_FLOOR)
    A = y.sum() / m.sum()
    mu = A * m
    return float(np.sum(y * np.log(mu) - mu)), A, np.inf


# ===========================================================================
# 3. Profile likelihood over alpha
# ===========================================================================

def growth_rate_start(y):
    """Rough early growth rate (per week) from the data, for starting values.

    Least-squares slope of log(y + 1) from week 1 up to the week where the
    counts first reach half their maximum (the clean rising phase).
    """
    y = np.asarray(y, dtype=float)
    stop = int(np.argmax(y >= 0.5 * y.max())) + 1
    stop = max(stop, 3)
    t = np.arange(1, stop + 1)
    slope = np.polyfit(t, np.log(y[:stop] + 1.0), 1)[0]
    return float(min(max(slope, 0.05), 0.9 * R_MAX))


class ProfileFitter:
    """Fit the Caputo SEIR to one weekly series, for every alpha on a grid.

    settings: dict with sigma, gamma, and the substep table (from substep_table).
    likelihood: 'nb' (negative binomial, for counts), 'poisson_quasi'
                (noise-free expected counts, Poisson limit) or 'nb_expected'
                (noise-free expected counts, negative binomial limit with the
                dispersion settings['k_fixed']).
    """

    def __init__(self, y, settings, likelihood="nb"):
        self.y = np.asarray(y, dtype=float)
        self.K = len(self.y)
        self.sigma = settings["sigma"]
        self.gamma = settings["gamma"]
        self.table = {round(row["alpha"], 6): row for row in settings["substeps"]}
        if likelihood == "nb":
            self.inner = best_scale_dispersion
        elif likelihood == "poisson_quasi":
            self.inner = poisson_quasi_loglik
        elif likelihood == "nb_expected":
            kfix = settings["k_fixed"]
            self.inner = lambda y, m: expected_nb_loglik(y, m, kfix)
        else:
            raise ValueError(likelihood)
        self.n_solves = 0

    def _row(self, alpha):
        """Solver setting for alpha: its own table row, or for an alpha between
        table values, the row of the next LOWER alpha. Lower alpha needs more
        sub-steps, so this choice is never less accurate (used by refine)."""
        key = round(alpha, 6)
        if key in self.table:
            return self.table[key]
        lower = [a for a in self.table if a <= key]
        if not lower:
            raise ValueError(f"alpha = {alpha} is below the solver table")
        return self.table[max(lower)]

    # ----- objective for fixed alpha -----
    def loglik(self, alpha, log_beta, log_e0):
        """Log-likelihood at (alpha, beta, e0), with A and k_disp maximised."""
        row = self._row(alpha)
        m = caputo_weekly(alpha, math.exp(log_beta), math.exp(log_e0), self.K,
                          self.sigma, self.gamma, row["method"], row["k"])
        self.n_solves += 1
        if m is None:
            return -1e12, None
        ll, A, kd = self.inner(self.y, m)
        return ll, (A, kd, m)

    def _bounds(self, alpha):
        lo = math.log(M.beta_for_growth_rate(R_MIN, self.sigma, self.gamma, alpha))
        hi = math.log(M.beta_for_growth_rate(R_MAX, self.sigma, self.gamma, alpha))
        return [(lo, hi), (math.log(E0_BOUNDS[0]), math.log(E0_BOUNDS[1]))]

    def _clip(self, z, alpha):
        b = self._bounds(alpha)
        return np.array([min(max(z[i], b[i][0] + 1e-9), b[i][1] - 1e-9) for i in range(2)])

    def optimise(self, alpha, start):
        """Nelder-Mead over (log beta, log e0) for fixed alpha, from `start`."""
        start = self._clip(start, alpha)
        res = minimize(lambda z: -self.loglik(alpha, z[0], z[1])[0], start,
                       method="Nelder-Mead", bounds=self._bounds(alpha),
                       options={"xatol": 1e-3, "fatol": 1e-3, "maxfev": 400,
                                "initial_simplex": np.array([start, start + [0.15, 0.0],
                                                             start + [0.0, 0.7]])})
        return -res.fun, res.x

    def _mapped_start(self, z, alpha_from, alpha_to):
        """Starting point for alpha_to with the SAME growth rate as z had at alpha_from.

        Uses Proposition 10: for each alpha there is exactly one beta with a
        given growth rate r.
        """
        beta = math.exp(z[0])
        p = {"beta": beta, "sigma": self.sigma, "gamma": self.gamma, "N": 1.0, "mu": 0.0}
        lam1, _ = M.early_eigenvalues(p, alpha_from)
        if lam1 <= 0:
            return np.array(z)
        r = min(max(lam1 ** (1.0 / alpha_from), R_MIN * 1.01), R_MAX * 0.99)
        return np.array([math.log(M.beta_for_growth_rate(r, self.sigma, self.gamma, alpha_to)), z[1]])

    def profile(self, alpha_grid, stop_drop=None, stop_below=0.7):
        """Profile log-likelihood at each alpha (grid must end at 1.0).

        Pass 1 runs from alpha = 1 downwards, each fit starting from the
        previous optimum mapped to the same growth rate. Pass 2 runs back
        upwards and re-fits wherever the neighbour's optimum gives a better
        starting value (protects against local optima).

        Early stop (optional): below alpha = stop_below, pass 1 stops once two
        successive grid points are both more than stop_drop log-likelihood
        units below the best value so far. Points that far down cannot affect
        alpha-hat, the 95% interval (which needs 1.92 units) or the test.
        """
        grid = sorted(alpha_grid, reverse=True)
        assert abs(grid[0] - 1.0) < 1e-12, "alpha grid must include 1.0"
        r0 = growth_rate_start(self.y)
        best = {}

        # multi-start at alpha = 1
        cands = []
        for e0 in (1e-6, 1e-5, 1e-4, 1e-3, 1e-2):
            z = np.array([math.log(M.beta_for_growth_rate(r0, self.sigma, self.gamma, 1.0)),
                          math.log(e0)])
            cands.append((self.loglik(1.0, *z)[0], z))
        cands.sort(key=lambda c: -c[0])
        for _, z in cands[:2]:
            ll, x = self.optimise(1.0, z)
            if 1.0 not in best or ll > best[1.0][0]:
                best[1.0] = (ll, x)

        # pass 1: downwards
        for a_prev, a in zip(grid[:-1], grid[1:]):
            z = self._mapped_start(best[a_prev][1], a_prev, a)
            best[a] = self.optimise(a, z)
            if stop_drop is not None and a < stop_below:
                top = max(v[0] for v in best.values())
                done = sorted(best)[:2]                      # the two lowest alphas so far
                if len(done) == 2 and all(best[d][0] < top - stop_drop for d in done):
                    break

        # pass 2: upwards (over the alphas actually computed)
        up = sorted(best)
        for a_prev, a in zip(up[:-1], up[1:]):
            z = self._mapped_start(best[a_prev][1], a_prev, a)
            if self.loglik(a, *z)[0] > best[a][0] + 1e-6:
                ll, x = self.optimise(a, z)
                if ll > best[a][0]:
                    best[a] = (ll, x)

        self._best = best
        return [self._describe(a, *best[a]) for a in sorted(best)]

    def _describe(self, a, ll, x):
        """One profile row: alpha, log-likelihood and the fitted quantities."""
        _, extra = self.loglik(a, *x)
        A, kd, m = extra
        p = {"beta": math.exp(x[0]), "sigma": self.sigma, "gamma": self.gamma,
             "N": 1.0, "mu": 0.0}
        lam1, _ = M.early_eigenvalues(p, a)
        return {"alpha": a, "loglik": ll, "beta": p["beta"], "e0": math.exp(x[1]),
                "A": A, "k_disp": kd, "R0": M.r0_caputo_seir(p, a),
                "r": lam1 ** (1.0 / a) if lam1 > 0 else float("nan"),
                "floored_weeks": int(np.sum(m <= M_FLOOR))}

    def load(self, prof):
        """Restore a stored profile (rows with alpha, loglik, beta, e0), so that
        fill() can add points to it without refitting the whole profile."""
        self._best = {round(float(p["alpha"]), 6): (float(p["loglik"]),
                      np.array([math.log(p["beta"]), math.log(p["e0"])])) for p in prof}

    def fill(self, alphas):
        """Add profile points at the given alphas (call after profile() or load()).

        Each new point starts from the nearest computed point, mapped to the
        same growth rate. Returns the merged profile, sorted by alpha.
        """
        best = self._best
        for a in sorted(alphas):
            a = round(float(a), 6)
            if a in best:
                continue
            near = min(best, key=lambda b: abs(b - a))
            best[a] = self.optimise(a, self._mapped_start(best[near][1], near, a))
        return [self._describe(a, *best[a]) for a in sorted(best)]

    def refine(self, step=0.01, half_width=0.04):
        """Add profile points every `step` within +-half_width of the best alpha.

        Call after profile(). Returns the merged profile.
        """
        best = self._best
        a_best = max(best, key=lambda a: best[a][0])
        lo = max(min(best), a_best - half_width)
        new = [round(v, 4) for v in np.arange(lo, min(1.0, a_best + half_width) + 1e-9, step)]
        return self.fill(new)


# ===========================================================================
# 4. Summarising a profile
# ===========================================================================

def summarise_profile(prof):
    """alpha-hat, 95% profile interval and the boundary LRT for alpha = 1.

    prof: output of ProfileFitter.profile (sorted by alpha, ending at 1.0).

    alpha-hat (the point estimate) is the vertex of a parabola through the
    best computed point and its two neighbours (clipped to that bracket).
    The test and the interval, however, use only log-likelihood values that
    were actually COMPUTED: l_max is the largest computed value. Because the
    true maximum can only be higher, the LRT is slightly too small, never too
    large: the test is conservative, so a rejection of alpha = 1 cannot be an
    artefact of interpolating between grid points. (It is conservative with
    respect to the grid only; the 50:50 mixture itself is an approximation.)

    95% interval: the alphas with 2 (l_max - l_p(alpha)) <= 3.84. Between
    computed points the profile is interpolated with a shape-preserving cubic
    (PCHIP), which follows curvature better than straight lines. If the set
    is split into separate pieces (the profile can have two peaks), the
    reported interval is the hull from its lowest to its highest alpha, and
    ci_split is True.
    """
    a = np.array([p["alpha"] for p in prof])
    ll = np.array([p["loglik"] for p in prof])
    i = int(np.argmax(ll))
    a_hat, ll_hat = a[i], ll[i]
    j = min(max(i, 1), len(a) - 2)                       # centre of a 3-point bracket
    coef = np.polyfit(a[j - 1:j + 2], ll[j - 1:j + 2], 2)
    if coef[0] < 0:
        v = -coef[1] / (2 * coef[0])
        v = min(max(v, a[j - 1]), a[j + 1], 1.0)
        if np.polyval(coef, v) > ll_hat:
            a_hat = float(v)

    lrt = max(0.0, 2.0 * (ll_hat - ll[-1]))
    p_value = 1.0 if lrt == 0 else 0.5 * chi2.sf(lrt, 1)

    # profile interval: where 2 (ll_hat - ll) crosses CI_CRIT (PCHIP interpolation)
    dev = 2.0 * (ll_hat - ll)
    inside = dev <= CI_CRIT
    k_in = np.where(inside)[0]
    split = bool(len(k_in) > 0 and (k_in.max() - k_in.min() + 1) != len(k_in))
    curve = PchipInterpolator(a, dev - CI_CRIT) if len(a) >= 2 else None

    def crossing(i, j):
        """alpha between a[i] and a[j] where the interpolated deviance equals CI_CRIT."""
        f = lambda x: float(curve(x))
        if f(a[i]) * f(a[j]) > 0:                  # no sign change (should not happen)
            return a[i] if abs(f(a[i])) < abs(f(a[j])) else a[j]
        return brentq(f, a[i], a[j], xtol=1e-6)

    if len(k_in) == 0:
        lo = hi = a_hat
    else:
        k0, k1 = k_in.min(), k_in.max()
        lo = a[0] if k0 == 0 else crossing(k0 - 1, k0)
        hi = a[-1] if k1 == len(a) - 1 else crossing(k1, k1 + 1)
        lo, hi = min(lo, a_hat), max(hi, a_hat)
    return {"alpha_hat": float(a_hat), "loglik_max": float(ll_hat), "loglik_alpha1": float(ll[-1]),
            "lrt": float(lrt), "p_value": float(p_value),
            "reject_alpha1": bool(lrt > LRT_CRIT_BOUNDARY),
            "ci_low": float(lo), "ci_high": float(hi),
            "ci_low_at_grid_edge": bool(inside[0]), "ci_split": split,
            "best_grid_alpha": float(a[i])}
