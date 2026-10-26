"""
mittag_leffler.py -- the Mittag-Leffler functions E_a(z) and E_{a,b}(z), real z.

WHAT IT IS
    E_{a,b}(z) = sum over k = 0, 1, 2, ... of  z^k / Gamma(a*k + b),
    and E_a(z) = E_{a,1}(z). With a = 1 every Gamma(k + 1) is k!, so
    E_1(z) = e^z. It plays the role for fractional equations that e^z plays
    for ordinary ones: the solution of  D^a y = -lam * y,  y(0) = 1  is
    y(t) = E_a(-lam * t^a).

HOW IT IS COMPUTED (two methods, chosen automatically)
 1. The series, added up with high-precision arithmetic (mpmath).
    For z >= 0 all terms are positive and this is easy.
    For z < 0 the terms alternate in sign and first grow huge (the largest is
    about exp(|z|^(1/a))), so ordinary 16-digit arithmetic loses everything to
    cancellation. We use about |z|^(1/a)/ln(10) + 30 digits. That is exact but
    slow when |z|^(1/a) is large, so we only use it while |z|^(1/a) <= 60.
 2. For z < 0 beyond that, and 0 < a < 1, an exact integral formula
    (Gorenflo et al. 2014, Mittag-Leffler Functions, Sec. 3.7): with x = |z|
    and t = x^(1/a),
        E_a(-x)      = integral_0^inf  exp(-r t) K(r) dr,
        E_{a,a}(-x)  = t^(1-a) * integral_0^inf  r exp(-r t) K(r) dr,
    where K(r) = sin(a pi) r^(a-1) / ( pi (r^(2a) + 2 r^a cos(a pi) + 1) ) >= 0
    (the denominator is >= sin(a pi)^2 r^(2a) > 0, so K is never negative).
    The second line follows from the first because
        d/dt E_a(-t^a) = -t^(a-1) E_{a,a}(-t^a).
    Both integrands are positive, so there is no cancellation at all.
    Note K(r) >= 0 shows directly that E_a(-x) > 0 and E_{a,a}(-x) >= 0,
    facts used in the positivity proof of the paper (Theorem 3).
"""

import math

import mpmath as mp
import numpy as np
from scipy.integrate import quad

SERIES_PEAK_LIMIT = 60.0   # use the series while |z|^(1/a) <= this


def _series(z, a, b):
    """Sum the series with enough digits to survive cancellation."""
    peak = abs(z) ** (1.0 / a)
    digits = int(peak / math.log(10.0)) + 30
    with mp.workdps(digits):
        zz = mp.mpf(z)
        aa, bb = mp.mpf(a), mp.mpf(b)   # a*k must ALSO be done in high precision:
        # a rounding error of 1e-16 in the Gamma argument, multiplied by the
        # huge cancelling terms, would otherwise ruin the answer.
        total = mp.mpf(0)
        z_power = mp.mpf(1)                     # holds z^k
        tiny = mp.mpf(10) ** (-(digits - 5))
        k = 0
        while True:
            term = z_power * mp.rgamma(aa * k + bb)  # rgamma(x) = 1/Gamma(x)
            total += term
            # stop once past the largest term (it sits near k = peak/a)
            # and the terms have become negligible
            if k > peak / a + 10 and abs(term) <= tiny * max(abs(total), mp.mpf(1)):
                break
            z_power *= zz
            k += 1
        return float(total)


def _integral(x, a, b):
    """E_a(-x) (b = 1) or E_{a,a}(-x) (b = a) for x > 0 and 0 < a < 1.

    Uses the integral formula in the module docstring with two changes of
    variable. First r = u^(1/a): since r^(a-1) dr = (1/a) du, the factor
    r^(a-1) (infinite at r = 0) disappears. Then u = w / t^a, which rescales
    the integrand so it has width about 1 whatever the size of t:
        E_a(-x)     = c t^(-a)  * integral_0^inf exp(-w^(1/a)) / D(w) dw,
        E_{a,a}(-x) = c t^(-2a) * integral_0^inf w^(1/a) exp(-w^(1/a)) / D(w) dw,
    with c = sin(a pi)/(a pi), t = x^(1/a) and
        D(w) = (w/t^a)^2 + 2 (w/t^a) cos(a pi) + 1   (always > 0).
    """
    t = x ** (1.0 / a)
    ta = t ** a                                   # equals x
    c = math.cos(a * math.pi)
    const = math.sin(a * math.pi) / (a * math.pi)
    D = lambda w: (w / ta) ** 2 + 2.0 * (w / ta) * c + 1.0
    if b == 1.0:
        f = lambda w: math.exp(-w ** (1.0 / a)) / D(w)
        factor = t ** (-a)
    else:
        f = lambda w: w ** (1.0 / a) * math.exp(-w ** (1.0 / a)) / D(w)
        factor = t ** (-2.0 * a)
    total = 0.0
    for lo, hi in [(0.0, 1.0), (1.0, 10.0), (10.0, np.inf)]:
        part, _ = quad(f, lo, hi, limit=200, epsabs=0, epsrel=1e-13)
        total += part
    return const * factor * total


def ml_scalar(z, a, b=1.0):
    """E_{a,b}(z) for one real number z (0 < a <= 2, b > 0)."""
    z = float(z)
    if z == 0.0:
        return float(mp.rgamma(b))
    peak = abs(z) ** (1.0 / a)
    if z < 0 and peak > SERIES_PEAK_LIMIT and 0 < a < 1 and (b == 1.0 or b == a):
        return _integral(-z, a, b)
    if peak > 4000:
        raise ValueError(f"E_{{{a},{b}}}({z}) is out of range for this implementation")
    return _series(z, a, b)


def ml(z, a, b=1.0):
    """E_{a,b}(z) for a number or a NumPy array z (element by element)."""
    z_arr = np.asarray(z, dtype=float)
    out = np.array([ml_scalar(v, a, b) for v in z_arr.ravel()])
    return out.reshape(z_arr.shape) if z_arr.ndim else float(out[0])


# ===========================================================================
# Fast vectorised E_a(-x), x >= 0, 0 < a < 1  (used by the AHM solver)
# ===========================================================================
# The integral formula above is evaluated with FIXED Gauss-Legendre rules on
# a few sub-intervals, all values of x at once (NumPy arrays), so thousands
# of values take milliseconds. Accuracy is checked against ml() in
# 07_check_solvers.py (about 1e-12).
#   u-form (no rescaling):  E_a(-x) = c * int_0^inf exp(-t u^(1/a)) / (u^2 + 2u cos(a pi) + 1) du
#   with t = x^(1/a) and c = sin(a pi)/(a pi). The denominator has its
#   smallest value near u = -cos(a pi) (close to u = 1 when a is near 1),
#   so the sub-intervals are refined around there.

_GL_X, _GL_W = np.polynomial.legendre.leggauss(64)


def _gauss_nodes(edges):
    """Nodes and weights of 64-point Gauss-Legendre rules on each [edges[i], edges[i+1]]."""
    nodes, weights = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        nodes.append(0.5 * (hi - lo) * _GL_X + 0.5 * (hi + lo))
        weights.append(0.5 * (hi - lo) * _GL_W)
    return np.concatenate(nodes), np.concatenate(weights)


def ml_neg_fast(x, a):
    """E_a(-x) for an array x >= 0 and 0 < a < 1 (fast, about 12 correct digits)."""
    x = np.asarray(x, dtype=float)
    if a >= 1.0:
        return np.exp(-x)                       # E_1(-x) = e^(-x)
    out = np.empty_like(x)
    small = x <= 0.5
    # 1. Small x: the series has no cancellation problem; 60 terms suffice.
    if np.any(small):
        xs = x[small]
        k = np.arange(60)
        from scipy.special import rgamma
        out[small] = ((-xs[:, None]) ** k * rgamma(a * k + 1.0)).sum(axis=1)
    # 2. Larger x: the integral. Two features need resolving:
    #    - the denominator's minimum near u0 = -cos(a pi), of width ~ sin(a pi);
    #    - the factor exp(-t u^(1/a)), which falls off where u ~ t^(-a) = 1/x.
    #    So the sub-interval edges include points around u0 and at multiples
    #    of 1/x; each value of x gets its own set of nodes.
    if np.any(~small):
        c = math.cos(a * math.pi)
        u0 = max(-c, 0.0)
        # near a = 1 the denominator's dip at u0 becomes very narrow (width
        # sin(a pi)), so we refine around it on several scales
        width = max(math.sin(a * math.pi), 1e-12)
        fixed = {0.0, u0, 4.0, 20.0}
        for mult in (1.0, 4.0, 16.0, 64.0, 256.0):
            fixed |= {max(u0 - mult * width, 0.0), u0 + mult * width}
        vn, vw = _gauss_nodes([0.0, 1.0])          # tail [20, inf): u = 20 / v
        tail_u, tail_w = 20.0 / vn, vw * 20.0 / vn ** 2
        const = math.sin(a * math.pi) / (a * math.pi)
        idx = np.flatnonzero(~small)
        for i in idx:
            xi = x[i]
            t = xi ** (1.0 / a)
            scale = {e / xi for e in (0.2, 1.0, 5.0, 30.0) if e / xi < 20.0}
            un, uw = _gauss_nodes(sorted(fixed | scale))
            u = np.concatenate([un, tail_u])
            w = np.concatenate([uw, tail_w])
            vals = np.exp(-t * u ** (1.0 / a)) / (u * u + 2.0 * u * c + 1.0)
            out[i] = const * (vals @ w)
    return out
