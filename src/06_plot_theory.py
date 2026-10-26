"""
06_plot_theory.py -- figures for Section 3 (models and theory).

  memory_kernel      (a) how much a change at an earlier time still counts
                         in the Caputo derivative; (b) the fraction still
                         infectious, E_a(-(gamma t)^a), versus exponential decay
  identifiability    (a) the R0 implied by one observed growth rate r, as a
                         function of alpha; (b) early epidemic curves for
                         four values of alpha, all tuned to the SAME r,
                         each divided by its own long-run exponential

Figures are saved in figures/ (.png and .pdf) and copied to paper/figures/.
Takes about 20 seconds.
"""

import math
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import gamma as Gamma

import config
import models as M
from mittag_leffler import ml

plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "axes.edgecolor": config.TEXT_SOFT, "axes.labelcolor": config.TEXT_DARK,
    "xtick.color": config.TEXT_SOFT, "ytick.color": config.TEXT_SOFT,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": config.GRID, "grid.linewidth": 0.6,
    "legend.frameon": False, "savefig.bbox": "tight",
})

ALPHAS = [0.5, 0.7, 0.9, 1.0]     # one colour + marker each, in this fixed order


def save(fig, name):
    """Save as PNG and PDF in figures/, and copy both to paper/figures/."""
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    config.PAPER_FIG_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        path = config.FIGURES_DIR / f"{name}.{ext}"
        fig.savefig(path, dpi=200)
        shutil.copy(path, config.PAPER_FIG_DIR / path.name)
    plt.close(fig)
    print(f"Saved figures/{name}.png and .pdf (copied to paper/figures/)")


def style(i):
    """Colour, marker and spacing of markers for curve number i."""
    return dict(color=config.COLORS[i], marker=config.MARKERS[i], markevery=0.12,
                markersize=4, linewidth=1.6)


# ---------------------------------------------------------------------------
# Figure: memory kernel and relaxation
# ---------------------------------------------------------------------------
def plot_memory():
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.1))

    # (a) Weight given to a change that happened u time units ago:
    #     w(u) = u^(-a) / Gamma(1-a). (alpha = 1 has no curve: all weight at u = 0.)
    u = np.logspace(-2, 2, 200)
    for i, a in enumerate(ALPHAS[:3]):
        axes[0].plot(u, u ** (-a) / Gamma(1 - a), label=f"$\\alpha$ = {a}", **style(i))
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("Time since the change, $t-s$")
    axes[0].set_ylabel("Weight in the Caputo derivative")
    axes[0].set_title("(a) memory: weight of past changes", loc="left", color=config.TEXT_DARK)
    axes[0].legend(loc="lower left", fontsize=8)

    # (b) Fraction still infectious after time t (in units of 1/gamma):
    #     E_a(-(gamma t)^a); alpha = 1 gives exp(-gamma t).
    t = np.logspace(-2, 3, 160)
    for i, a in enumerate(ALPHAS):
        y = np.exp(-t) if a == 1.0 else ml(-t ** a, a)
        keep = y > 1e-4
        axes[1].plot(t[keep], y[keep], label=f"$\\alpha$ = {a}" + (" (exponential)" if a == 1 else ""),
                     **style(i))
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[1].set_ylim(1e-4, 1.5)
    axes[1].set_xlabel("Time, in units of $1/\\gamma$")
    axes[1].set_ylabel("Fraction remaining, $E_\\alpha(-(\\gamma t)^\\alpha)$")
    axes[1].set_title("(b) slow, power-law decay", loc="left", color=config.TEXT_DARK)
    axes[1].legend(loc="lower left", fontsize=8)
    fig.tight_layout()
    save(fig, "memory_kernel")


# ---------------------------------------------------------------------------
# Figure: what the growth rate can and cannot identify
# ---------------------------------------------------------------------------
def plot_identifiability():
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2))

    # (a) R0 implied by a fixed growth rate r, as alpha varies (sigma = gamma)
    a_grid = np.linspace(0.3, 1.0, 120)
    ratios = [0.2, 0.5, 1.0, 2.0]                # r / gamma
    for i, q in enumerate(ratios):
        R0 = M.r0_from_growth_rate_caputo(q, 1.0, 1.0, a_grid)
        axes[0].plot(a_grid, R0, label=f"$r/\\gamma$ = {q}", **style(i))
    axes[0].set_xlabel("Fractional order $\\alpha$")
    axes[0].set_ylabel("$R_0$ implied by the growth rate")
    axes[0].set_title("(a) same $r$, different $R_0$", loc="left", color=config.TEXT_DARK)
    axes[0].legend(loc="upper left", fontsize=8)

    # (b) Early curves, all with the same long-run growth rate r = 0.5 gamma.
    #     We plot each curve DIVIDED BY its own long-run form (c1 v1_I / a) e^(r t)
    #     (Proposition 6). Every ratio tends to 1, because all four have the
    #     same r; what is left is the transient, the only place alpha shows.
    r, sigma, gam = 0.5, 2.0, 1.0
    t = np.linspace(0.02, 12, 150)
    for i, a in enumerate(ALPHAS):
        beta = M.beta_for_growth_rate(r, sigma, gam, a)
        p = {"beta": beta, "sigma": sigma, "gamma": gam, "N": 1.0, "mu": 0.0}
        I = M.early_solution(p, a, t, x0=(1.0, 0.0))[:, 1]
        lam, V = np.linalg.eig(M.early_matrix(p, a))
        j = int(np.argmax(lam.real))
        c = np.linalg.solve(V.real, np.array([1.0, 0.0]))
        amp = c[j] * V.real[1, j] / a                      # long-run amplitude
        axes[1].plot(t, I / (amp * np.exp(r * t)), label=f"$\\alpha$ = {a}", **style(i))
    axes[1].axhline(1.0, color=config.TEXT_SOFT, linestyle="--", linewidth=1.0)
    axes[1].set_ylim(0, 1.15)
    axes[1].set_xlabel("Time, in units of $1/\\gamma$")
    axes[1].set_ylabel("Infectious / long-run exponential")
    axes[1].set_title("(b) same $r$: $\\alpha$ shows only early on", loc="left", color=config.TEXT_DARK)
    axes[1].legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    save(fig, "identifiability")


def main():
    plot_memory()
    plot_identifiability()


if __name__ == "__main__":
    main()
