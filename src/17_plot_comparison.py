"""
17_plot_comparison.py -- figures for Section 7 (model comparison and forecasting).

  aic_weights          Akaike (AICc) weights of the six models for each series
  synthetic_aic        for data with a known truth: how often each model has
                       the lowest AICc
  forecast_scores      forecast skill relative to the ordinary SEIR model:
                       relative CRPS and median log-score gain, per series
  forecast_examples    mpox: forecasts from three origins by four models
  alpha_by_origin      the Caputo order fitted at each forecast origin

Reads the CSV files written by 16_compare_forecast.py. Saved in figures/
(.png and .pdf) and copied to paper/figures/. About 10 seconds.
"""

import importlib
import shutil
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config
import outbreaks as O
import synthetic as Y

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
# one colour and one marker per model, used in every figure
STYLE = {"seir": (BLUE, "o", "ordinary SEIR"),
         "gfree": (config.TEXT_SOFT, "D", "SEIR, $\\gamma$ estimated"),
         "decline": (AQUA, "^", "SEIR, declining transmission"),
         "hetero": (YELLOW, "s", "SEIR, heterogeneous recovery"),
         "caputo": (ORANGE, "v", "Caputo SEIR"),
         "ahm": (config.TEXT_DARK, "P", "AHM-SEIR")}
MODELS = list(STYLE)
TITLES = {"mpox-US": "mpox, US", "ebola-GN": "Ebola, Guinea", "ebola-LR": "Ebola, Liberia",
          "ebola-SL": "Ebola, Sierra Leone", "measles-TX": "measles, W. Texas",
          "measles-SC": "measles, Upstate SC", "measles-SCk": "measles, Short Creek"}


def save(fig, name):
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    config.PAPER_FIG_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        path = config.FIGURES_DIR / f"{name}.{ext}"
        fig.savefig(path, dpi=200)
        shutil.copy(path, config.PAPER_FIG_DIR / path.name)
    plt.close(fig)
    print(f"Saved figures/{name}.png and .pdf (copied to paper/figures/)")


def stacked(ax, labels, shares, invert=True):
    """Horizontal stacked bars: shares is a dict model -> array (one value per label)."""
    left = np.zeros(len(labels))
    for m in MODELS:
        if m not in shares:
            continue
        c, _, lab = STYLE[m]
        ax.barh(range(len(labels)), shares[m], left=left, color=c, label=lab,
                edgecolor="white", linewidth=0.6, height=0.7)
        left += shares[m]
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    if invert:                     # first label at the top (once only for shared axes)
        ax.invert_yaxis()
    ax.grid(axis="y", visible=False)


# ---------------------------------------------------------------------------
def plot_aic_weights():
    d = pd.read_csv(R / "stage6_aic.csv")
    fig, ax = plt.subplots(figsize=(7.0, 3.2))
    shares = {m: np.array([d[(d.series == s) & (d.model == m)].weight_aicc.iloc[0]
                           for s in O.SHORT_NAMES]) for m in MODELS}
    stacked(ax, [TITLES[s] for s in O.SHORT_NAMES], shares)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Akaike weight (AICc)")
    ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0))
    save(fig, "aic_weights")


def plot_synthetic():
    d = pd.read_csv(R / "stage6_synthetic_aic.csv")
    labels = {t["id"]: f"{t['id']} {t['label']}".replace("alpha=", "$\\alpha=$") for t in Y.TRUTHS}
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.4), sharey=True)
    for i, (ax, kd) in enumerate(zip(axes, sorted(d.k_noise.unique()))):
        e = d[d.k_noise == kd].set_index("truth").loc[Y.TRUTH_IDS]
        shares = {m: e[f"pct_best_{m}"].to_numpy() / 100 for m in MODELS if f"pct_best_{m}" in e}
        stacked(ax, [labels[t] for t in Y.TRUTH_IDS], shares, invert=(i == 0))
        ax.set_xlim(0, 1)
        ax.set_title(f"noise $k_{{\\mathrm{{nb}}}}={kd}$")
        ax.set_xlabel("share of data sets with lowest AICc")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, bbox_to_anchor=(0.55, -0.12))
    fig.tight_layout()
    save(fig, "synthetic_aic")


def plot_scores():
    s = pd.read_csv(R / "stage6_forecast_summary.csv")
    rows = O.SHORT_NAMES + ["all"]
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.8), sharey=True)
    for j, m in enumerate(MODELS):
        c, mk, lab = STYLE[m]
        e = s[s.model == m].set_index("series").loc[rows]
        yy = np.arange(len(rows)) + (j - 2.5) * 0.11
        axes[0].plot(e.rel_crps, yy, mk, color=c, markersize=4.5, label=lab)
        axes[1].plot(e.logscore_gain_median, yy, mk, color=c, markersize=4.5)
    axes[0].axvline(1.0, color=config.TEXT_SOFT, linestyle=":", linewidth=0.9)
    axes[1].axvline(0.0, color=config.TEXT_SOFT, linestyle=":", linewidth=0.9)
    axes[1].set_xscale("symlog", linthresh=0.1)    # linear near 0, logarithmic beyond 0.1
    axes[1].set_xlim(-0.3, 15)
    axes[0].set_xlabel("mean CRPS / mean CRPS of ordinary SEIR\n(< 1: better)")
    axes[1].set_xlabel("median log-score gain over ordinary SEIR\n(> 0: better; log scale beyond 0.1)")
    axes[0].set_yticks(range(len(rows)))
    axes[0].set_yticklabels([TITLES.get(r, "all series") for r in rows])
    axes[0].invert_yaxis()
    for ax in axes:
        ax.grid(axis="y", visible=False)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=3,
               bbox_to_anchor=(0.55, -0.14))
    fig.tight_layout()
    save(fig, "forecast_scores")


def plot_examples():
    f = pd.read_csv(R / "stage6_forecasts.csv")
    y = O.load_series("mpox-US")["y"]
    show = ["seir", "decline", "hetero", "caputo"]
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.8), sharey=True)
    for ax, o in zip(axes, (8, 12, 16)):
        ax.plot(np.arange(len(y)), y, "o", color=config.TEXT_DARK, markersize=2.4,
                label="reported cases")
        ax.axvline(o - 0.5, color=config.TEXT_SOFT, linestyle=":", linewidth=0.9)
        for m in show:
            c, mk, lab = STYLE[m]
            e = f[(f.series == "mpox-US") & (f.model == m) & (f.origin == o)].sort_values("week")
            ax.plot(e.week, e["mean"], marker=mk, color=c, markersize=3.5, linewidth=1.3, label=lab)
        ax.set_title(f"forecast from week {o}")
        ax.set_xlabel("weeks since take-off")
        ax.set_xlim(-0.5, 24)
    axes[0].set_ylabel("weekly cases")
    axes[0].set_ylim(0, 4500)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=5,
               bbox_to_anchor=(0.5, -0.1))
    fig.tight_layout()
    save(fig, "forecast_examples")


def plot_alpha_by_origin():
    """The Caputo order fitted at each forecast origin: how it moves as data accumulate."""
    f = pd.read_csv(R / "stage6_forecasts.csv")
    f = f[(f.model == "caputo") & (f.horizon == 1)]
    fig, ax = plt.subplots(figsize=(7.0, 3.0))
    cols = [ORANGE, BLUE, AQUA, YELLOW, config.TEXT_SOFT, config.TEXT_DARK, ORANGE]
    mks = ["o", "s", "^", "D", "v", "P", "x"]
    for j, short in enumerate(O.SHORT_NAMES):
        e = f[f.series == short].sort_values("origin")
        ax.plot(e.origin, e.alpha, marker=mks[j], color=cols[j], markersize=3.5, linewidth=1.1,
                linestyle="-" if j < 6 else "--", label=TITLES[short])
    ax.set_xlabel("weeks of data used (forecast origin)")
    ax.set_ylabel("fitted $\\alpha$ (Caputo SEIR)")
    ax.set_ylim(0.37, 1.03)
    ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0))
    save(fig, "alpha_by_origin")


if __name__ == "__main__":
    plot_aic_weights()
    if (R / "stage6_synthetic_aic.csv").exists():
        plot_synthetic()
    if (R / "stage6_forecast_summary.csv").exists():
        plot_scores()
        plot_examples()
        plot_alpha_by_origin()
