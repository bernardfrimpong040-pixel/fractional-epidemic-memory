"""
scoring.py -- model comparison and forecast scores (Stage 6).

Nothing runs here; 15_check_comparison.py, 16_compare_forecast.py and
17_plot_comparison.py import these functions.

INFORMATION CRITERIA
    AIC  = 2k - 2 l                     (Akaike 1974)
    AICc = AIC + 2k(k + 1)/(n - k - 1)  (small-sample correction, Hurvich & Tsai 1989)
    l: maximised log-likelihood, k: number of estimated parameters, n: number
    of weekly counts. Lower is better. Differences of about 2 or less mean
    the data barely separate two models; the Akaike weight
        w_i = exp(-Delta_i / 2) / sum_j exp(-Delta_j / 2),   Delta_i = AIC_i - min AIC,
    is the relative support for model i within the set.

FORECAST SCORES (for a negative binomial forecast with mean mu, dispersion k)
    log score  log P(Y = y): higher is better; it rewards putting high
               probability on what happened (and is the same quantity the
               likelihood uses, so it is the forecast version of AIC).
    CRPS       continuous ranked probability score, in cases:
                   CRPS = sum over x = 0, 1, 2, ... of (F(x) - 1{y <= x})^2,
               F the forecast's distribution function. Lower is better; for a
               forecast with no spread it is the absolute error |y - mu|.
    coverage   whether y falls inside the central 50% and 90% forecast intervals.
"""

import numpy as np
from scipy.stats import nbinom

# Parameters estimated in each model: transmission, e0, scale A and dispersion k
# (4), plus the model's own extra ones.
N_PARAMS = {"seir": 4, "caputo": 5, "ahm": 5, "gfree": 5, "decline": 6, "hetero": 6}


def aic(loglik, k):
    return 2.0 * k - 2.0 * loglik


def aicc(loglik, k, n):
    return aic(loglik, k) + 2.0 * k * (k + 1) / (n - k - 1)


def akaike_weights(values):
    """Weights from a list of AIC (or AICc) values."""
    v = np.asarray(values, dtype=float)
    w = np.exp(-(v - v.min()) / 2.0)
    return w / w.sum()


def _p(mu, kd):
    """scipy's nbinom(n, p) with n = k, p = k/(k + mu) has mean mu, variance mu + mu^2/k."""
    return kd / (kd + mu)


def nb_logscore(y, mu, kd):
    return nbinom.logpmf(np.asarray(y), kd, _p(np.asarray(mu, dtype=float), kd))


def nb_crps(y, mu, kd, tail=1e-9):
    """CRPS of NB(mu, kd) forecasts for observed counts y (arrays of equal length).

    The sum runs over x = 0 .. x_max, where x_max is beyond both y and the
    forecast's 1 - tail quantile, so the neglected terms are below tail.
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    mu = np.atleast_1d(np.asarray(mu, dtype=float))
    out = np.empty(len(y))
    for i in range(len(y)):
        p = _p(mu[i], kd)
        x_max = int(max(y[i], nbinom.ppf(1.0 - tail, kd, p))) + 1
        x = np.arange(x_max + 1)
        F = nbinom.cdf(x, kd, p)
        out[i] = np.sum((F - (y[i] <= x)) ** 2)
    return out


def nb_interval(mu, kd, level):
    """Central forecast interval [lower, upper] with probability >= level."""
    p = _p(np.asarray(mu, dtype=float), kd)
    a = (1.0 - level) / 2.0
    return nbinom.ppf(a, kd, p), nbinom.ppf(1.0 - a, kd, p)
