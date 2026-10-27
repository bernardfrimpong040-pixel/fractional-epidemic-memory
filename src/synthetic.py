"""
synthetic.py -- the eight "true" models used in the synthetic experiments
(Section 5 of the paper), and tools to generate data from them.

Nothing runs here; 10_synthetic_experiments.py, 11_plot_synthetic.py and
the later stages import it.

Every truth describes an outbreak in a population of N = 100,000 started by
100 exposed people, observed for K = 36 weeks. Time is in weeks; the mean
latent period and mean infectious period are 1 week in every truth
(sigma = gamma = 1), the values the fitted model is told.

Four truths have NO memory (ordinary differential equations); four have
memory: two Caputo SEIR models, the AHM model, and -- the subtle one --
heterogeneous recovery, which has no memory for any individual but a
long-tailed infectious period for the population.

Wherever possible the transmission rate is chosen so that the epidemic
grows at the same early rate r0 = 0.3416 per week (doubling about every
2 weeks), the growth rate of the ordinary SEIR model with beta = 1.8.
"""

from functools import lru_cache

import numpy as np
import pandas as pd

import models as M
import solvers as S

N_POP = 1e5        # population
E0_SEED = 100.0    # initial exposed people
K_WEEKS = 36       # weeks observed
SIGMA = 1.0        # 1 / mean latent period (per week)
GAMMA = 1.0        # 1 / mean infectious period (per week)
BETA_SEIR = 1.8    # ordinary SEIR transmission rate (R0 = 1.8)
PEAK_COUNT = 1000.0  # every truth is scaled so its expected peak = 1000 reported cases/week
R0_GROWTH = M.growth_rate_caputo({"beta": BETA_SEIR, "sigma": SIGMA, "gamma": GAMMA,
                                  "N": 1.0, "mu": 0.0}, 1.0)        # 0.3416 per week

# ---------------------------------------------------------------------------
# The eight truths. 'memory' says whether the mechanism has memory;
# 'alpha_true' is the fractional order when the truth IS a Caputo SEIR model.
# ---------------------------------------------------------------------------
TRUTHS = [
    {"id": "T0", "label": "SEIR", "memory": False, "alpha_true": 1.0},
    {"id": "T1", "label": "Erlang SEIR (3+3 stages)", "memory": False, "alpha_true": None},
    {"id": "T2", "label": "Declining transmission", "memory": False, "alpha_true": None},
    {"id": "T3", "label": "Reporting delay", "memory": False, "alpha_true": None},
    {"id": "T4", "label": "Caputo SEIR, alpha=0.8", "memory": True, "alpha_true": 0.8},
    {"id": "T5", "label": "Caputo SEIR, alpha=0.6", "memory": True, "alpha_true": 0.6},
    {"id": "T6", "label": "AHM, alpha=0.7", "memory": True, "alpha_true": None},
    {"id": "T7", "label": "Heterogeneous recovery", "memory": False, "alpha_true": None},
]
TRUTH_IDS = [t["id"] for t in TRUTHS]

# parameters of the individual truths
ERLANG_STAGES = (3, 3)                       # T1: latent and infectious stages
BETA_T_PARAMS = (2.6, 0.35, 0.10)            # T2: beta0, final fraction rho, rate kappa
DELAY_PARAMS = (2.0, 2.0)                    # T3: gamma delay, mean 2 weeks, shape 2
AHM_PARAMS = {"alpha": 0.7, "tau": 2.0, "gamma_c": 0.5}   # T6: mean infectious period 1 week
HETERO_PARAMS = {"gammas": (2.0, 1.0 / 3.0), "fractions": (0.8, 0.2)}  # T7: 80% fast, 20% slow


def weekly_entries(truth_id):
    """Expected number of people entering I (becoming infectious) in weeks 1..K.

    For T3 these are then delayed by the reporting-delay distribution.
    Returns an array of length K (people per week, before reporting).
    Each truth is solved only once; later calls return a copy of the result.
    """
    return _weekly_entries_cached(truth_id).copy()


@lru_cache(maxsize=None)
def _weekly_entries_cached(truth_id):
    """The actual computation behind weekly_entries (remembered after the first call)."""
    t_eval = np.arange(0.0, K_WEEKS + 1.0)
    N, E0, K = N_POP, E0_SEED, K_WEEKS

    if truth_id in ("T0", "T3"):
        _, o = S.simulate_erlang_seir(BETA_SEIR, SIGMA, GAMMA, N, 1, 1, E0, K, t_eval)
        w = np.diff(o["cumulative"])
        if truth_id == "T3":
            w = S.apply_reporting_delay(w, *DELAY_PARAMS)
        return w

    if truth_id == "T1":
        mE, mI = ERLANG_STAGES
        beta = GAMMA * M.r0_from_growth_rate_erlang(R0_GROWTH, SIGMA, GAMMA, mE, mI)
        _, o = S.simulate_erlang_seir(beta, SIGMA, GAMMA, N, mE, mI, E0, K, t_eval)
        return np.diff(o["cumulative"])

    if truth_id == "T2":
        _, o = S.simulate_seir_beta_t(*BETA_T_PARAMS, SIGMA, GAMMA, N, E0, K, t_eval)
        return np.diff(o["cumulative"])

    if truth_id in ("T4", "T5"):
        a = 0.8 if truth_id == "T4" else 0.6
        beta = M.beta_for_growth_rate(R0_GROWTH, SIGMA, GAMMA, a)
        p = {"beta": beta, "sigma": SIGMA, "gamma": GAMMA, "N": N, "mu": 0.0}
        # accurate solve: predictor-corrector with 256 sub-steps per week
        _, Y = S.simulate_caputo_seir(p, a, [N - E0, E0, 0.0, 0.0], float(K), 1.0,
                                      "pece", substeps=256)
        return np.diff(Y[:, 2] + Y[:, 3])

    if truth_id == "T6":
        a, tau, gc = AHM_PARAMS["alpha"], AHM_PARAMS["tau"], AHM_PARAMS["gamma_c"]
        omega_S0 = M.ahm_transmission_for_growth_rate(R0_GROWTH, tau, a, gc)
        _, o = S.solve_ahm(omega_S0 / N, tau, a, gc, N - E0, E0, float(K), 1.0)
        return np.diff(o["cumulative"])

    if truth_id == "T7":
        g, q = HETERO_PARAMS["gammas"], HETERO_PARAMS["fractions"]
        beta = M.beta_for_growth_rate_hetero(R0_GROWTH, SIGMA, g, q)
        _, o = S.simulate_hetero_seir(beta, SIGMA, g, q, N, E0, K, t_eval)
        return np.diff(o["cumulative"])

    raise ValueError(truth_id)


def expected_counts(truth_id):
    """Expected reported weekly cases: weekly_entries scaled so the peak is PEAK_COUNT."""
    w = weekly_entries(truth_id)
    return w * PEAK_COUNT / w.max()


def replicate(truth_id, k_disp, rep):
    """One noisy data set: negative binomial counts around expected_counts.

    The random seed is built from (truth number, k_disp, replicate number),
    so every data set is the same on every computer and in any order.
    """
    mu = expected_counts(truth_id)
    rng = np.random.default_rng([TRUTH_IDS.index(truth_id), int(k_disp), int(rep)])
    # numpy's negative_binomial(n, p) has mean n(1-p)/p; with n = k, p = k/(k+mu)
    # this is mean mu and variance mu + mu^2/k
    return rng.negative_binomial(k_disp, k_disp / (k_disp + mu)).astype(float)


def true_R0(truth_id):
    """The reproduction number of the truth itself, from its own parameters.

    T2 has time-varying transmission; its value here is the initial
    R0 = beta0 / gamma (it falls to 0.35 times that).
    """
    r0 = R0_GROWTH
    if truth_id in ("T0", "T3"):
        return BETA_SEIR / GAMMA
    if truth_id == "T1":
        return M.r0_from_growth_rate_erlang(r0, SIGMA, GAMMA, *ERLANG_STAGES)
    if truth_id == "T2":
        return BETA_T_PARAMS[0] / GAMMA
    if truth_id in ("T4", "T5"):
        a = 0.8 if truth_id == "T4" else 0.6
        p = {"beta": M.beta_for_growth_rate(r0, SIGMA, GAMMA, a), "sigma": SIGMA, "gamma": GAMMA,
             "N": 1.0, "mu": 0.0}
        return M.r0_caputo_seir(p, a)
    if truth_id == "T6":
        a, tau, gc = AHM_PARAMS["alpha"], AHM_PARAMS["tau"], AHM_PARAMS["gamma_c"]
        return M.ahm_r0(M.ahm_transmission_for_growth_rate(r0, tau, a, gc), tau, a, gc)
    if truth_id == "T7":
        g, q = np.array(HETERO_PARAMS["gammas"]), np.array(HETERO_PARAMS["fractions"])
        return M.beta_for_growth_rate_hetero(r0, SIGMA, g, q) * float(np.sum(q / g))
    raise ValueError(truth_id)


def describe_truths():
    """One row per truth: peak week, totals, and quantities used in the paper."""
    rows = []
    for t in TRUTHS:
        w = weekly_entries(t["id"])
        mu = w * PEAK_COUNT / w.max()
        rows.append({"id": t["id"], "label": t["label"], "memory": t["memory"],
                     "peak_week": int(np.argmax(w)) + 1, "true_R0": true_R0(t["id"]),
                     "infections_in_window_per_N": float(w.sum() / N_POP),
                     "expected_reported_total": float(mu.sum()),
                     "last_week_over_peak": float(w[-1] / w.max()),
                     "min_weekly": float(w.min())})
    return pd.DataFrame(rows)
