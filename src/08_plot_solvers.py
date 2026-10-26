"""
08_plot_solvers.py -- figures for Section 4 (numerical methods).

  solver_convergence  error against step size for both Caputo methods
                      (from results/stage3_convergence.csv, made by 07)
  matched_epidemics   whole epidemics for several alpha, each tuned to the
                      SAME early growth rate (Proposition 10): (a) Caputo SEIR,
                      (b) AHM model
  decline_tails       local slope d log I / d log t of the decline, which
                      tends to -alpha for both fractional models

Saved in figures/ (.png and .pdf) and copied to paper/figures/. About 20 s.
"""

import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config
import models as M
import solvers as S

plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "axes.edgecolor": config.TEXT_SOFT, "axes.labelcolor": config.TEXT_DARK,
    "xtick.color": config.TEXT_SOFT, "ytick.color": config.TEXT_SOFT,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": config.GRID, "grid.linewidth": 0.6,
    "legend.frameon": False, "savefig.bbox": "tight",
})


def save(fig, name):
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    config.PAPER_FIG_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        path = config.FIGURES_DIR / f"{name}.{ext}"
        fig.savefig(path, dpi=200)
        shutil.copy(path, config.PAPER_FIG_DIR / path.name)
    plt.close(fig)
    print(f"Saved figures/{name}.png and .pdf (copied to paper/figures/)")


def style(i, markevery=None):
    return dict(color=config.COLORS[i], marker=config.MARKERS[i], markersize=4,
                linewidth=1.6, markevery=markevery)


def local_slopes(t, y, ratio=1.1):
    """d log y / d log t over [t/ratio, t] at each grid point with t/ratio >= t[1]."""
    keep = t / ratio >= t[1]
    tt = t[keep]
    lo = np.interp(np.log(tt / ratio), np.log(t[1:]), np.log(y[1:]))
    return tt, (np.log(y[keep]) - lo) / np.log(ratio)


# ---------------------------------------------------------------------------
def plot_convergence():
    conv = pd.read_csv(config.RESULTS_DIR / "stage3_convergence.csv")
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.1), sharey=True)
    alphas = [0.3, 0.5, 0.7, 0.9]
    for ax, method, title in [(axes[0], "pece", "(a) predictor-corrector (PECE)"),
                              (axes[1], "l1", "(b) L1 scheme")]:
        for i, a in enumerate(alphas):
            d = conv[(conv.method == method) & (conv.alpha == a)]
            ax.plot(d.h, d.error, label=f"$\\alpha$ = {a}", **style(i))
        # reference slopes, anchored at the right end
        h = np.array([0.00625, 0.1])
        if method == "pece":
            for q, ls in [(1.3, ":"), (1.9, "--")]:
                ax.plot(h, 2e-3 * (h / 0.1) ** q, color=config.TEXT_SOFT, linestyle=ls, linewidth=1,
                        label=f"slope {q}")
        else:
            ax.plot(h, 8e-3 * (h / 0.1), color=config.TEXT_SOFT, linestyle="--", linewidth=1,
                    label="slope 1")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Step size $h$")
        ax.set_title(title, loc="left", color=config.TEXT_DARK)
        ax.legend(loc="upper left", fontsize=7.5)
    axes[0].set_ylabel("Error at $t=2$ for $D^\\alpha y=-y$")
    fig.tight_layout()
    save(fig, "solver_convergence")


# ---------------------------------------------------------------------------
def plot_matched():
    """Epidemics with the same early growth rate r = 0.5 per week."""
    r, sigma, gam, N = 0.5, 1.0, 1.0, 1e6
    alphas = [0.6, 0.8, 0.9, 1.0]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2), sharey=True)

    for i, a in enumerate(alphas):
        beta = M.beta_for_growth_rate(r, sigma, gam, a)
        p = {"beta": beta, "sigma": sigma, "gamma": gam, "N": N, "mu": 0.0}
        t, Y = S.simulate_caputo_seir(p, a, [N - 10, 10, 0, 0], 120.0, 1 / 7, "pece")
        axes[0].plot(t, Y[:, 2], label=f"$\\alpha$ = {a}", **style(i, markevery=60))
    axes[0].set_title("(a) Caputo SEIR, same $r$", loc="left", color=config.TEXT_DARK)

    tau, gc = 1.0, 0.5
    for i, a in enumerate(alphas):
        omegaN = M.ahm_transmission_for_growth_rate(r, tau, a, gc)
        t, o = S.solve_ahm(omegaN / N, tau, a, gc, N - 10, 10.0, 120.0, 1 / 7)
        axes[1].plot(t, o["I"], label=f"$\\alpha$ = {a}", **style(i, markevery=60))
    axes[1].set_title("(b) AHM, same $r$ ($\\gamma_c=0.5$)", loc="left", color=config.TEXT_DARK)

    for ax in axes:
        ax.set_yscale("log")
        ax.set_ylim(1, 3e5)
        ax.set_xlabel("Weeks")
        ax.legend(loc="lower right", fontsize=7.5)
    axes[0].set_ylabel("Infectious $I(t)$")
    fig.tight_layout()
    save(fig, "matched_epidemics")


# ---------------------------------------------------------------------------
def plot_tails():
    P = {"beta": 2.0, "sigma": 1.0, "gamma": 1.0, "N": 1e6, "mu": 0.0}
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.1), sharey=True)
    for i, a in enumerate([0.6, 0.8]):
        t, Y = S.simulate_caputo_seir(P, a, [1e6 - 10, 10, 0, 0], 6000.0, 0.5, "l1")
        tt, sl = local_slopes(t, Y[:, 2])
        m = tt >= 30
        axes[0].plot(tt[m], sl[m], label=f"$\\alpha$ = {a}", **style(i, markevery=0.1))
        axes[0].axhline(-a, color=config.COLORS[i], linestyle="--", linewidth=1)
        t, o = S.solve_ahm(2.5e-6, 1.0, a, 0.0, 1e6 - 10, 10.0, 2000.0, 0.5)
        tt, sl = local_slopes(t, o["I"])
        m = tt >= 30
        axes[1].plot(tt[m], sl[m], label=f"$\\alpha$ = {a}", **style(i, markevery=0.1))
        axes[1].axhline(-a, color=config.COLORS[i], linestyle="--", linewidth=1)
    axes[0].set_title("(a) Caputo SEIR", loc="left", color=config.TEXT_DARK)
    axes[1].set_title("(b) AHM ($\\gamma_c=0$)", loc="left", color=config.TEXT_DARK)
    for ax in axes:
        ax.set_xscale("log")
        ax.set_ylim(-4.5, 1)
        ax.set_xlabel("Weeks since start (log scale)")
        ax.legend(loc="lower right", fontsize=7.5)
    axes[0].set_ylabel("Local slope $d\\log I/d\\log t$")
    fig.tight_layout()
    save(fig, "decline_tails")


def main():
    plot_convergence()
    plot_matched()
    plot_tails()


if __name__ == "__main__":
    main()
