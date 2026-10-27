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


def ahm_weekly(alpha, beta, e0, K, sigma, gamma, k):
    """Weekly increases m_1..m_K of the cumulative entries into I for the
    AHM-SEIR model (Stage 5), with k sub-steps per week. None if it fails."""
    try:
        with np.errstate(all="ignore"):
            _, out = S.solve_ahm_seir(beta, sigma, gamma, alpha, e0, float(K), 1.0, k)
    except (RuntimeError, ValueError, FloatingPointError, OverflowError):
        return None
    m = np.diff(out["cumulative"])
    return m if np.all(np.isfinite(m)) else None


# The two fractional models share the fitting code below. These small helpers
# hide the differences between them.
#   model = "caputo": Caputo SEIR (rates raised to the power alpha)
#   model = "ahm":    AHM-SEIR (Mittag-Leffler infectious period, Stage 5)

def model_weekly(model, alpha, beta, e0, K, sigma, gamma, row):
    """Weekly increments for either model; row is the solver-table row."""
    if model == "caputo":
        return caputo_weekly(alpha, beta, e0, K, sigma, gamma, row["method"], row["k"])
    return ahm_weekly(alpha, beta, e0, K, sigma, gamma, row["k"])


def beta_for_r(model, r, sigma, gamma, alpha):
    """Transmission rate giving early growth rate r (Proposition 10 / Euler-Lotka)."""
    if model == "caputo":
        return M.beta_for_growth_rate(r, sigma, gamma, alpha)
    return M.ahm_seir_beta_for_growth_rate(r, sigma, gamma, alpha)


def r_for_beta(model, beta, sigma, gamma, alpha):
    """Early growth rate per week (NaN if the outbreak cannot grow)."""
    if model == "caputo":
        p = {"beta": beta, "sigma": sigma, "gamma": gamma, "N": 1.0, "mu": 0.0}
        lam1, _ = M.early_eigenvalues(p, alpha)
        return lam1 ** (1.0 / alpha) if lam1 > 0 else float("nan")
    r = M.ahm_seir_growth_rate(beta, sigma, gamma, alpha)
    return r if r > 0 else float("nan")


def r0_of(model, beta, sigma, gamma, alpha):
    """Basic reproduction number: (beta/gamma)^alpha (Caputo) or beta/gamma (AHM-SEIR)."""
    if model == "caputo":
        return (beta / gamma) ** alpha
    return M.ahm_seir_r0(beta, gamma)


def substep_table(alpha_grid, sigma, gamma, K, r_max=R_MAX, tol=2e-3,
                  candidates=(1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256),
                  model="caputo"):
    """Choose a solver and number of sub-steps for each alpha.

    For each alpha we take the hardest case allowed in a fit (fastest growth
    r_max) and an easy one (r = 0.1), compute an accurate reference with PECE
    and 512 sub-steps, and find for each method the smallest k whose weekly
    increments are within tol (relative to the peak) of the reference in both
    cases. We then keep whichever method is faster at its k.
    Returns a list of dicts: alpha, method, k, max_error, seconds.

    model="ahm": the same search for the AHM-SEIR solver (one method, called
    'trap' in the table: trapezoid rule for S and E, product integration for
    I; reference with 512 sub-steps).
    """
    if model == "ahm":
        return _ahm_substep_table(alpha_grid, sigma, gamma, K, r_max, tol, candidates)
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


def _ahm_substep_table(alpha_grid, sigma, gamma, K, r_max, tol, candidates):
    table = []
    for a in alpha_grid:
        cases = []
        for r in (r_max, 0.1):
            beta = M.ahm_seir_beta_for_growth_rate(r, sigma, gamma, a)
            cases.append((beta, ahm_weekly(a, beta, 1e-3, K, sigma, gamma, 512)))
        best = None
        for k in candidates:
            errs = []
            for beta, ref in cases:
                m = ahm_weekly(a, beta, 1e-3, K, sigma, gamma, k)
                errs.append(np.inf if m is None else np.max(np.abs(m - ref)) / np.max(ref))
            if max(errs) < tol:
                best = {"alpha": float(a), "method": "trap", "k": int(k),
                        "max_error": float(max(errs)), "seconds": float("nan")}
                break
        if best is None:
            raise RuntimeError(f"no AHM step reached tolerance {tol} at alpha = {a}")
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
    stop = int(np.argmax(np.nan_to_num(y, nan=-1.0) >= 0.5 * np.nanmax(y))) + 1
    stop = max(stop, 3)
    t = np.arange(1, stop + 1)
    ok = np.isfinite(y[:stop])                     # skip masked (NaN) weeks
    if ok.sum() < 2:                               # too few weeks for a slope
        return 0.2
    slope = np.polyfit(t[ok], np.log(y[:stop][ok] + 1.0), 1)[0]
    return float(min(max(slope, 0.05), 0.9 * R_MAX))


class ProfileFitter:
    """Fit a fractional model to one weekly series, for every alpha on a grid.

    settings: dict with sigma, gamma, and the substep table (from substep_table),
              and optionally model = "caputo" (default) or "ahm" (AHM-SEIR).
    y may contain NaN: those weeks are left out of the likelihood (masked).
    likelihood: 'nb' (negative binomial, for counts), 'poisson_quasi'
                (noise-free expected counts, Poisson limit) or 'nb_expected'
                (noise-free expected counts, negative binomial limit with the
                dispersion settings['k_fixed']).
    """

    def __init__(self, y, settings, likelihood="nb"):
        self.y = np.asarray(y, dtype=float)
        self.K = len(self.y)
        self.mask = np.isfinite(self.y)              # weeks that enter the likelihood
        self.model = settings.get("model", "caputo")
        self.wide_start = settings.get("wide_start", False)
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
        table values (used by refine), the row of the next LOWER alpha.
        Lower alpha USUALLY needs more sub-steps, but not always (the table is
        chosen by speed), so this rule was checked directly: for every alpha
        0.41, 0.42, ..., 0.99 between table values, with the Stage 5 tables,
        the error stays below the 0.2% tolerance (largest 0.199%)."""
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
        m = model_weekly(self.model, alpha, math.exp(log_beta), math.exp(log_e0), self.K,
                         self.sigma, self.gamma, row)
        self.n_solves += 1
        if m is None:
            return -1e12, None
        ll, A, kd = self.inner(self.y[self.mask], m[self.mask])
        return ll, (A, kd, m)

    def _bounds(self, alpha):
        lo = math.log(beta_for_r(self.model, R_MIN, self.sigma, self.gamma, alpha))
        hi = math.log(beta_for_r(self.model, R_MAX, self.sigma, self.gamma, alpha))
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
        r = r_for_beta(self.model, math.exp(z[0]), self.sigma, self.gamma, alpha_from)
        if not np.isfinite(r):
            return np.array(z)
        r = min(max(r, R_MIN * 1.01), R_MAX * 0.99)
        return np.array([math.log(beta_for_r(self.model, r, self.sigma, self.gamma, alpha_to)), z[1]])

    def profile(self, alpha_grid, stop_drop=None, stop_below=0.7, global_restart=False):
        """Profile log-likelihood at each alpha (grid must end at 1.0).

        Pass 1 runs from alpha = 1 downwards, each fit starting from the
        previous optimum mapped to the same growth rate. Pass 2 runs back
        upwards and re-fits wherever the neighbour's optimum gives a better
        starting value (protects against local optima).

        Early stop (optional): below alpha = stop_below, pass 1 stops once two
        successive grid points are both more than stop_drop log-likelihood
        units below the best value so far. Points that far down cannot affect
        alpha-hat, the 95% interval (which needs 1.92 units) or the test.

        global_restart=True (used for real data, Stage 5) adds pass 3: every
        alpha is also started from the overall best fit, mapped to the same
        growth rate. Real profiles can have two separate peaks, and passes 1
        and 2 only pass information between neighbouring alphas.
        """
        grid = sorted(alpha_grid, reverse=True)
        assert abs(grid[0] - 1.0) < 1e-12, "alpha grid must include 1.0"
        r0 = growth_rate_start(self.y)
        best = {}

        # multi-start at alpha = 1
        # (wide_start, used for real data in Stage 5: also try other growth
        # rates and a larger seed, and optimise the best three starts; a
        # series that is already large in its first week can need e0 near 0.1)
        rates = [r0] + ([0.1, 0.25, 0.5, 0.8] if self.wide_start else [])
        seeds = (1e-6, 1e-5, 1e-4, 1e-3, 1e-2) + ((5e-2,) if self.wide_start else ())
        cands = []
        for r in rates:
            for e0 in seeds:
                z = np.array([math.log(beta_for_r(self.model, r, self.sigma, self.gamma, 1.0)),
                              math.log(e0)])
                cands.append((self.loglik(1.0, *z)[0], z))
        cands.sort(key=lambda c: -c[0])
        for _, z in cands[:3 if self.wide_start else 2]:
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

        # pass 3 (optional): restart every alpha from the overall best fit
        if global_restart:
            a_top = max(best, key=lambda a: best[a][0])
            for a in sorted(best):
                if a == a_top:
                    continue
                z = self._mapped_start(best[a_top][1], a_top, a)
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
        beta = math.exp(x[0])
        return {"alpha": a, "loglik": ll, "beta": beta, "e0": math.exp(x[1]),
                "A": A, "k_disp": kd, "R0": r0_of(self.model, beta, self.sigma, self.gamma, a),
                "r": r_for_beta(self.model, beta, self.sigma, self.gamma, a),
                "floored_weeks": int(np.sum(m[self.mask] <= M_FLOOR))}

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

    def fitted_mean(self, alpha, beta, e0, A):
        """Expected weekly counts A * m_k of the fitted model (all weeks, masked or not)."""
        m = model_weekly(self.model, alpha, beta, e0, self.K, self.sigma, self.gamma,
                         self._row(alpha))
        return A * np.maximum(m, M_FLOOR)

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


# ===========================================================================
# 5. Integer-order (memoryless) alternatives, fitted to real data (Stage 5)
# ===========================================================================
# All three are ordinary differential equations, so they have no memory.
#   "seir"     ordinary SEIR (the alpha = 1 member of both fractional models)
#   "decline"  SEIR with declining transmission (Section 3.6 (b)):
#              beta(t) = beta0 (rho + (1 - rho) e^(-kappa t)),  0 <= rho <= 1
#   "gfree"    ordinary SEIR with the infectious rate gamma ESTIMATED from the
#              data (gamma' = gamma x g) instead of fixed: this checks whether
#              alpha simply makes up for a wrong infectious period
#   "hetero"   SEIR with heterogeneous recovery (Section 3.6 (d)): a fraction q
#              of new infectious people recover slowly (rate g_s), the rest
#              c times faster (rate c g_s). The MEAN infectious period is held at
#              1/gamma (as in every model we fit):
#                  q / g_s + (1 - q) / (c g_s) = 1 / gamma
#              =>  g_s = gamma (q + (1 - q) / c).
# Parameters are optimised on unbounded scales: logs for positive numbers and
# logit(x) = log(x / (1 - x)) for numbers between 0 and 1.

ALT_MODELS = ("seir", "gfree", "decline", "hetero")
ALT_EXTRA = {"seir": 0, "gfree": 1, "decline": 2, "hetero": 2}   # parameters beyond (beta, e0)


def _expit(x):
    return 1.0 / (1.0 + math.exp(-x))


def hetero_classes(gamma, q, c):
    """(gammas, fractions) of the two-class recovery model with mean 1/gamma."""
    g_s = gamma * (q + (1.0 - q) / c)
    return np.array([g_s, c * g_s]), np.array([q, 1.0 - q])


def alt_params(model, z):
    """Turn the optimiser's vector z into named parameters."""
    p = {"beta": math.exp(z[0]), "e0": math.exp(z[1])}
    if model == "decline":
        p["rho"], p["kappa"] = _expit(z[2]), math.exp(z[3])
    elif model == "hetero":
        p["q"], p["c"] = _expit(z[2]), 1.0 + math.exp(z[3])
    elif model == "gfree":
        p["g"] = math.exp(z[2])                       # gamma' = gamma x g
    return p


def alt_weekly(model, p, K, sigma, gamma, ode_method="DOP853"):
    """Weekly increments of cumulative entries into I (population fraction).
    ode_method: "DOP853" (Stage 5) or "LSODA" (Stage 6, faster when stiff)."""
    t_eval = np.arange(K + 1, dtype=float)
    try:
        if model in ("seir", "gfree"):
            g = gamma * p.get("g", 1.0)
            _, out = S.simulate_erlang_seir(p["beta"], sigma, g, 1.0, 1, 1, p["e0"], K, t_eval,
                                            method=ode_method)
        elif model == "decline":
            _, out = S.simulate_seir_beta_t(p["beta"], p["rho"], p["kappa"], sigma, gamma,
                                            1.0, p["e0"], K, t_eval, method=ode_method)
        else:
            gs, qs = hetero_classes(gamma, p["q"], p["c"])
            _, out = S.simulate_hetero_seir(p["beta"], sigma, gs, qs, 1.0, p["e0"], K, t_eval,
                                            method=ode_method)
    except (RuntimeError, ValueError, FloatingPointError):
        return None
    m = np.diff(out["cumulative"])
    return m if np.all(np.isfinite(m)) else None


def alt_growth_rate(model, p, sigma, gamma):
    """Early growth rate per week (at t = 0 for the declining model)."""
    if model == "hetero":
        gs, qs = hetero_classes(gamma, p["q"], p["c"])
        f = lambda r: p["beta"] * sigma / (sigma + r) * np.sum(qs / (gs + r)) - 1.0
    else:
        gamma = gamma * p.get("g", 1.0)
        f = lambda r: p["beta"] * sigma / ((sigma + r) * (gamma + r)) - 1.0
    lo = -0.999 * min(sigma, gamma if model != "hetero" else hetero_classes(gamma, p["q"], p["c"])[0][0])
    if f(lo) * f(50.0) > 0:
        return float("nan")
    return brentq(f, lo, 50.0, xtol=1e-12)


def alt_r0(model, p, gamma):
    """R0 = beta x mean infectious period = beta / gamma (at t = 0 for 'decline';
    beta / (gamma x g) for 'gfree')."""
    return p["beta"] / (gamma * p.get("g", 1.0))


class AltFitter:
    """Maximum-likelihood fit of one memoryless alternative to a weekly series.

    Same observation model as the fractional fits (negative binomial with the
    scale A and dispersion k_disp maximised in the inner step), so the
    log-likelihoods of all models are directly comparable. NaN weeks are
    masked.
    """

    def __init__(self, y, sigma, gamma, model, ode_method="DOP853"):
        assert model in ALT_MODELS
        self.ode_method = ode_method
        self.y = np.asarray(y, dtype=float)
        self.K = len(self.y)
        self.mask = np.isfinite(self.y)
        self.sigma, self.gamma, self.model = sigma, gamma, model
        self.n_solves = 0

    def loglik(self, z):
        p = alt_params(self.model, z)
        m = alt_weekly(self.model, p, self.K, self.sigma, self.gamma, self.ode_method)
        self.n_solves += 1
        if m is None:
            return -1e12, None
        ll, A, kd = best_scale_dispersion(self.y[self.mask], m[self.mask])
        return ll, (A, kd, m)

    # The optimiser works with w = (log r, log e0, extras), where r is the early
    # growth rate (at t = 0), instead of z = (log beta, log e0, extras). Then the
    # same range of growth rates, R_MIN to R_MAX, is allowed as in the fractional
    # fits, for every model. beta follows from r (Proposition 10 and eq. (hetero)).
    def _z_from_w(self, w):
        r = math.exp(w[0])
        if self.model == "hetero":
            p = alt_params("hetero", np.array([0.0, 0.0, w[2], w[3]]))
            beta = M.beta_for_growth_rate_hetero(r, self.sigma, *hetero_classes(self.gamma, p["q"], p["c"]))
        elif self.model == "gfree":
            beta = M.beta_for_growth_rate(r, self.sigma, self.gamma * math.exp(w[2]), 1.0)
        else:
            beta = M.beta_for_growth_rate(r, self.sigma, self.gamma, 1.0)
        return np.concatenate([[math.log(beta)], np.asarray(w[1:], dtype=float)])

    def _box(self):
        box = [(math.log(R_MIN), math.log(R_MAX)), (math.log(E0_BOUNDS[0]), math.log(E0_BOUNDS[1]))]
        # logit(rho) or logit(q); then log(kappa) (per week) or log(c - 1)
        extra = {"gfree": [(math.log(0.05), math.log(20.0))],     # log g
                 "decline": [(-12.0, 12.0), (math.log(1e-3), math.log(50.0))],
                 "hetero": [(-12.0, 12.0), (math.log(1e-3), math.log(1e4))]}
        return box + extra.get(self.model, [])

    def _nm(self, start_w):
        box = self._box()
        start = np.array([min(max(v, b[0] + 1e-9), b[1] - 1e-9) for v, b in zip(start_w, box)])
        res = minimize(lambda w: -self.loglik(self._z_from_w(w))[0], start, method="Nelder-Mead",
                       bounds=box, options={"xatol": 1e-4, "fatol": 1e-4, "maxfev": 3000})
        return -res.fun, res.x

    def fit(self, seir_start=None):
        """Multi-start Nelder-Mead; each run is restarted once from its end point.

        seir_start: optional (log r, log e0) of an SEIR fit to start from.
        Returns a dict with the fitted parameters and the log-likelihood
        (parameters reported on the beta scale, as z = (log beta, log e0, ...)).
        """
        r0 = min(growth_rate_start(self.y), 0.95 * R_MAX)
        bases = []
        if seir_start is not None:
            bases.append(np.array(seir_start, dtype=float))
        for e0 in (1e-5, 1e-3, 3e-2):
            bases.append(np.array([math.log(r0), math.log(e0)]))
        extras = {"seir": [[]], "gfree": [[0.0], [math.log(0.3)], [math.log(3.0)]],
                  "decline": [[0.0, math.log(0.1)], [-2.0, math.log(0.3)], [1.0, math.log(1.0)]],
                  "hetero": [[-1.5, math.log(4.0)], [-3.0, math.log(15.0)], [-3.0, math.log(500.0)],
                             [0.0, math.log(1.0)]]}
        starts = [np.concatenate([b, np.array(e, dtype=float)]) for b in bases
                  for e in extras[self.model]]
        # screen the starts with one evaluation each, optimise the best three
        starts.sort(key=lambda w: -self.loglik(self._z_from_w(w))[0])
        best = None
        for w in starts[:3]:
            ll, x = self._nm(w)
            ll, x = self._nm(x)                       # restart: NM can stall
            if best is None or ll > best[0]:
                best = (ll, x)
        self._best = best
        out = self.describe(self._z_from_w(best[1]))
        box = self._box()
        out["at_bound"] = " ".join(name for name, v, b in
                                   zip(["r", "e0", "x1", "x2"], best[1], box)
                                   if min(v - b[0], b[1] - v) < 1e-3)
        return out

    def describe(self, z):
        ll, (A, kd, m) = self.loglik(z)
        p = alt_params(self.model, z)
        out = {"model": self.model, "loglik": ll, "A": A, "k_disp": kd,
               "R0": alt_r0(self.model, p, self.gamma),
               "r": alt_growth_rate(self.model, p, self.sigma, self.gamma),
               "n_params": 4 + ALT_EXTRA[self.model], "z": list(map(float, z))}
        out.update(p)
        return out

    def fitted_mean(self, z, A):
        m = alt_weekly(self.model, alt_params(self.model, z), self.K, self.sigma, self.gamma,
                       self.ode_method)
        return A * np.maximum(m, M_FLOOR)
