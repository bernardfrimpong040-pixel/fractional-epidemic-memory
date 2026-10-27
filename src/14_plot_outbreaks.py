"""
14_plot_outbreaks.py -- figures for Section 6 (fitting to outbreak data).

  outbreak_fits        the seven series with the best Caputo fit, the ordinary
                       SEIR and the declining-transmission and
                       heterogeneous-recovery SEIR
  outbreak_profiles    profile likelihoods for alpha: Caputo SEIR and AHM-SEIR
  outbreak_bootstrap   alpha-hat in data simulated from four fitted memoryless
                       models, against the alpha-hat of the real data
  outbreak_sensitivity alpha-hat and 95% interval for every data variant
  outbreak_residuals   randomised quantile residuals over time

Reads the CSV files written by 13_fit_outbreaks.py. Saved in figures/
(.png and .pdf) and copied to paper/figures/. About 10 seconds.
"""

import shutil
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config
import fitting as F
import outbreaks as O

warnings.simplefilter("ignore")
plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9,
    "axes.edgecolor": config.TEXT_SOFT, "axes.labelcolor": config.TEXT_DARK,
    "xtick.color": config.TEXT_SOFT, "ytick.color": config.TEXT_SOFT,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": config.GRID, "grid.linewidth": 0.6,
    "legend.frameon": False, "savefig.bbox": "tight",
})
BLUE, ORANGE, AQUA, YELLOW = config.COLORS
R = config.RESULTS_DIR
TITLES = {"mpox-US": "mpox, United States", "ebola-GN": "Ebola, Guinea",
          "ebola-LR": "Ebola, Liberia", "ebola-SL": "Ebola, Sierra Leone",
          "measles-TX": "measles, West Texas", "measles-SC": "measles, Upstate SC",
          "measles-SCk": "measles, Short Creek"}


def save(fig, name):
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    config.PAPER_FIG_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        path = config.FIGURES_DIR / f"{name}.{ext}"
        fig.savefig(path, dpi=200)
        shutil.copy(path, config.PAPER_FIG_DIR / path.name)
    plt.close(fig)
    print(f"Saved figures/{name}.png and .pdf (copied to paper/figures/)")


def panels():
    """A 4 x 2 grid; the eighth panel is left for the legend."""
    fig, axes = plt.subplots(4, 2, figsize=(7.0, 8.4))
    return fig, axes.ravel()


# ---------------------------------------------------------------------------
def plot_fits():
    res = pd.read_csv(R / "stage5_residuals.csv")
    fig, axes = panels()
    styles = {"caputo": (ORANGE, "-", "Caputo SEIR, best $\\alpha$"),
              "seir": (BLUE, "--", "ordinary SEIR ($\\alpha=1$)"),
              "decline": (AQUA, "-.", "SEIR, declining transmission"),
              "hetero": (YELLOW, ":", "SEIR, heterogeneous recovery")}
    for ax, short in zip(axes, O.SHORT_NAMES):
        d = res[res.series == short]
        flag = O.load_series(short)["flag"]
        first = d[d.model == "caputo"]
        ax.plot(first.week, first.y, "o", color=config.TEXT_DARK, markersize=2.6,
                label="reported cases")
        if flag.any():
            ax.plot(first.week[flag], first.y[flag], "x", color=config.TEXT_DARK, markersize=6,
                    label="flagged reporting spike")
        for m, (c, ls, lab) in styles.items():
            e = d[d.model == m]
            if m == "caputo":
                lab = f"Caputo SEIR, $\\alpha={e.alpha.iloc[0]:.2f}$"
            ax.plot(e.week, e["mean"], color=c, linestyle=ls, linewidth=1.5, label=lab)
        ax.set_title(TITLES[short])
        ax.set_xlabel("weeks since take-off")
        ax.set_ylabel("weekly cases")
    h, l = [], []
    for ax in axes[:7]:
        for hh, ll in zip(*ax.get_legend_handles_labels()):
            key = ll.split(",")[0] if ll.startswith("Caputo") else ll
            if key not in [x.split(",")[0] if x.startswith("Caputo") else x for x in l]:
                h.append(hh)
                l.append("Caputo SEIR, best $\\alpha$" if ll.startswith("Caputo") else ll)
    axes[7].axis("off")
    axes[7].legend(h, l, loc="center")
    fig.tight_layout()
    save(fig, "outbreak_fits")


def plot_profiles():
    prof = pd.read_csv(R / "stage5_profiles.csv")
    prof = prof[prof.variant == "main"]
    fig, axes = panels()
    for ax, short in zip(axes, O.SHORT_NAMES):
        for m, c, mk, lab in (("caputo", ORANGE, "o", "Caputo SEIR"),
                              ("ahm", BLUE, "s", "AHM-SEIR")):
            d = prof[(prof.series == short) & (prof.model == m)].sort_values("alpha")
            dev = 2 * (d.loglik.max() - d.loglik)
            ax.plot(d.alpha, np.minimum(dev, 40), color=c, marker=mk, markersize=2.5,
                    linewidth=1.3, label=lab)
        ax.axhline(F.CI_CRIT, color=config.TEXT_SOFT, linestyle=":", linewidth=0.9,
                   label="95% interval cut-off (3.84)")
        ax.set_ylim(-1, 41)
        ax.set_xlim(0.38, 1.02)
        ax.set_title(TITLES[short])
        ax.set_xlabel("$\\alpha$")
        ax.set_ylabel("$2(\\ell_{\\max}-\\ell_p(\\alpha))$")
    axes[7].axis("off")
    axes[7].legend(*axes[0].get_legend_handles_labels(), loc="center")
    axes[7].text(0.5, 0.15, "values above 40 drawn at 40", ha="center", transform=axes[7].transAxes,
                 color=config.TEXT_SOFT)
    fig.tight_layout()
    save(fig, "outbreak_profiles")


def plot_bootstrap():
    boot = pd.read_csv(R / "stage5_bootstrap.csv")
    boot = boot[boot.model == "caputo"]
    summ = pd.read_csv(R / "stage5_summary.csv")
    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    rng = np.random.default_rng(3)
    nulls = [("seir", BLUE, "o", "ordinary SEIR"),
             ("gfree", config.TEXT_SOFT, "D", "SEIR, $\\gamma$ estimated"),
             ("decline", AQUA, "^", "declining transmission"),
             ("hetero", YELLOW, "s", "heterogeneous recovery")]
    for i, short in enumerate(O.SHORT_NAMES):
        y0 = len(O.SHORT_NAMES) - 1 - i
        for j, (n, c, mk, lab) in enumerate(nulls):
            a = boot[(boot.series == short) & (boot["null"] == n)].alpha_hat
            yy = y0 + 0.27 - 0.18 * j + rng.uniform(-0.05, 0.05, len(a))
            ax.plot(a, yy, mk, color=c, markersize=2.6, alpha=0.75, label=lab if i == 0 else None)
        obs = summ[(summ.series == short) & (summ.model == "caputo")].alpha_hat.iloc[0]
        ax.plot([obs, obs], [y0 - 0.38, y0 + 0.38], color=ORANGE, linewidth=2.2,
                label="real data" if i == 0 else None)
    ax.set_yticks(range(len(O.SHORT_NAMES)))
    ax.set_yticklabels([TITLES[s] for s in reversed(O.SHORT_NAMES)])
    ax.set_xlabel("fitted $\\alpha$ (Caputo SEIR)")
    ax.set_xlim(0.38, 1.02)
    ax.grid(axis="y", visible=False)
    ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0), title="data simulated from")
    save(fig, "outbreak_bootstrap")


def plot_sensitivity():
    s = pd.read_csv(R / "stage5_sensitivity.csv")
    s = s[s.model == "caputo"]
    variants = [("main", "all weeks"), ("gamma0.8", "$\\gamma\\times0.8$"),
                ("gamma1.2", "$\\gamma\\times1.2$"), ("masked", "spikes masked"),
                ("truncated", "to peak + 4 weeks")]
    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    cols = [ORANGE, BLUE, AQUA, YELLOW, config.TEXT_SOFT]
    mks = ["o", "v", "^", "x", "s"]
    for i, short in enumerate(O.SHORT_NAMES):
        y0 = len(O.SHORT_NAMES) - 1 - i
        for j, (v, lab) in enumerate(variants):
            d = s[(s.series == short) & (s.variant == v)]
            if not len(d):
                continue
            d = d.iloc[0]
            yy = y0 + 0.3 - 0.15 * j
            ax.plot([d.ci_low, d.ci_high], [yy, yy], color=cols[j], linewidth=1.2)
            ax.plot(d.alpha_hat, yy, mks[j], color=cols[j], markersize=4,
                    label=lab if i == 0 or (v == "masked" and short == "ebola-GN") else None)
    ax.axvline(1.0, color=config.TEXT_SOFT, linestyle=":", linewidth=0.9)
    ax.set_yticks(range(len(O.SHORT_NAMES)))
    ax.set_yticklabels([TITLES[s] for s in reversed(O.SHORT_NAMES)])
    ax.set_xlabel("$\\hat\\alpha$ and 95% profile interval (Caputo SEIR)")
    ax.set_xlim(0.38, 1.02)
    ax.grid(axis="y", visible=False)
    h, l = ax.get_legend_handles_labels()
    order = [l.index(v[1]) for v in variants if v[1] in l]
    ax.legend([h[k] for k in order], [l[k] for k in order], loc="upper left",
              bbox_to_anchor=(1.0, 1.0), title="data / assumption")
    save(fig, "outbreak_sensitivity")


def plot_residuals():
    res = pd.read_csv(R / "stage5_residuals.csv")
    fig, axes = panels()
    for ax, short in zip(axes, O.SHORT_NAMES):
        for m, c, mk, lab in (("seir", BLUE, "o", "ordinary SEIR"),
                              ("caputo", ORANGE, "s", "Caputo SEIR, best $\\alpha$"),
                              ("decline", AQUA, "^", "SEIR, declining transmission"),
                              ("hetero", YELLOW, "D", "SEIR, heterogeneous recovery")):
            d = res[(res.series == short) & (res.model == m)]
            ax.plot(d.week, d.residual, marker=mk, color=c, markersize=2.5, linewidth=0.8,
                    label=lab)
        ax.axhline(0, color=config.TEXT_SOFT, linewidth=0.8)
        for v in (-2, 2):
            ax.axhline(v, color=config.TEXT_SOFT, linestyle=":", linewidth=0.8)
        ax.set_title(TITLES[short])
        ax.set_xlabel("weeks since take-off")
        ax.set_ylabel("quantile residual")
    axes[7].axis("off")
    axes[7].legend(*axes[0].get_legend_handles_labels(), loc="center")
    fig.tight_layout()
    save(fig, "outbreak_residuals")


if __name__ == "__main__":
    plot_fits()
    plot_profiles()
    if (R / "stage5_bootstrap.csv").exists():
        plot_bootstrap()
    plot_sensitivity()
    plot_residuals()
