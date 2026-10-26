"""
03_plot_data.py -- draw the Stage 1 figures from the cleaned data.

Figures (saved in figures/ as .png and .pdf, and copied to paper/figures/):
  data_overview         weekly cases for all seven series (3 panels)
  ebola_cleaning        Ebola running totals: raw reports vs cleaned curve
  growth_shape          cumulative cases from take-off to peak, on
                        (a) log-linear axes and (b) log-log axes

Why figure (c), growth_shape?
  Exponential growth, C(t) ~ e^(r t), is a STRAIGHT line when log C is plotted
  against t. Power-law growth, C(t) ~ t^p, is a straight line when log C is
  plotted against log t. Fractional-order models naturally produce power-law
  (slower than exponential) growth, so this figure shows by eye which data
  look "exponential" and which look "power-law" BEFORE any model is fitted.
"""

import shutil

import matplotlib
matplotlib.use("Agg")                 # draw to files only (no pop-up windows)
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config

# ---- One consistent, quiet style for every figure ----
plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "axes.edgecolor": config.TEXT_SOFT, "axes.labelcolor": config.TEXT_DARK,
    "xtick.color": config.TEXT_SOFT, "ytick.color": config.TEXT_SOFT,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": config.GRID, "grid.linewidth": 0.6,
    "legend.frameon": False, "savefig.bbox": "tight",
})


def save(fig, name):
    """Save a figure as PNG and PDF in figures/, and copy both to paper/figures/."""
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    config.PAPER_FIG_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        path = config.FIGURES_DIR / f"{name}.{ext}"
        fig.savefig(path, dpi=200)
        shutil.copy(path, config.PAPER_FIG_DIR / path.name)
    plt.close(fig)
    print(f"Saved figures/{name}.png and .pdf (copied to paper/figures/)")


def load():
    data = pd.read_csv(config.PROCESSED_DIR / "all_series_weekly.csv", parse_dates=["date"])
    summary = pd.read_csv(config.RESULTS_DIR / "stage1_data_summary.csv")
    reports = pd.read_csv(config.PROCESSED_DIR / "ebola_reports.csv", parse_dates=["date"])
    return data, summary, reports


# ---------------------------------------------------------------------------
# Figure 1: weekly cases, all series
# ---------------------------------------------------------------------------
def plot_overview(data, summary):
    panels = [("mpox", "(a) Mpox, United States, 2022"),
              ("ebola", "(b) Ebola, West Africa, 2014 (weekly, interpolated)"),
              ("measles", "(c) Measles, three US outbreaks, 2025-26")]
    fig, axes = plt.subplots(3, 1, figsize=(7.0, 7.6))

    for ax, (dataset, title) in zip(axes, panels):
        d = data[data["dataset"] == dataset]
        for i, (series, g) in enumerate(d.groupby("series", sort=False)):
            ax.plot(g["date"], g["cases"], color=config.COLORS[i],
                    marker=config.MARKERS[i], markersize=3.5, linewidth=1.6,
                    label=series)
            # Ring the weeks flagged as possible reporting batches
            s = g[g["spike_flag"]]
            ax.scatter(s["date"], s["cases"], s=70, facecolors="none",
                       edgecolors=config.TEXT_DARK, linewidths=1.0, zorder=5)
            # Short dashed line at the take-off week if it is not the first week
            row = summary[(summary["dataset"] == dataset) & (summary["series"] == series)].iloc[0]
            if row["takeoff_week"] > 0:
                ax.axvline(pd.Timestamp(row["takeoff_date"]), color=config.COLORS[i],
                           linestyle="--", linewidth=1.0, alpha=0.8)
        ax.set_title(title, loc="left", color=config.TEXT_DARK)
        ax.set_ylabel("Cases per week")
        ax.set_ylim(bottom=0)
        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
        if dataset != "mpox":
            ax.legend(loc="upper left", fontsize=8)

    # One note explaining the rings and dashed lines
    axes[0].scatter([], [], s=70, facecolors="none", edgecolors=config.TEXT_DARK,
                    label="flagged week (possible reporting batch)")
    axes[0].plot([], [], color=config.TEXT_SOFT, linestyle="--", label="take-off week")
    axes[0].legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    save(fig, "data_overview")


# ---------------------------------------------------------------------------
# Figure 2: Ebola cleaning check
# ---------------------------------------------------------------------------
def plot_ebola_cleaning(reports):
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.6), sharex=True)
    for i, (ax, (country, g)) in enumerate(zip(axes, reports.groupby("country", sort=False))):
        ax.plot(g["day"], g["cum_clean"], color=config.COLORS[i], linewidth=1.6,
                label="cleaned")
        ax.scatter(g["day"], g["cum_raw"], s=9, color=config.TEXT_SOFT,
                   label="reported", zorder=3)
        adj = g[g["adjusted"]]
        ax.scatter(adj["day"], adj["cum_raw"], s=28, facecolors="none",
                   edgecolors=config.TEXT_DARK, linewidths=0.9,
                   label="value adjusted", zorder=4)
        ax.set_title(country, loc="left", color=config.TEXT_DARK)
        ax.set_xlabel("Days since 22 Mar 2014")
    axes[0].set_ylabel("Cumulative cases")
    axes[0].legend(loc="upper left", fontsize=7)
    fig.tight_layout()
    save(fig, "ebola_cleaning")


# ---------------------------------------------------------------------------
# Figure 3: growth shape from take-off to peak
# ---------------------------------------------------------------------------
def plot_growth_shape(data, summary):
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.4))
    order = [("mpox", "United States", "Mpox US"),
             ("ebola", "Guinea", "Ebola Guinea"),
             ("ebola", "Liberia", "Ebola Liberia"),
             ("ebola", "Sierra Leone", "Ebola Sierra Leone"),
             ("measles", "West Texas", "Measles W. Texas"),
             ("measles", "Upstate South Carolina", "Measles S. Carolina"),
             ("measles", "Short Creek (UT-AZ)", "Measles Short Creek")]
    # Colour = dataset (fixed order); marker = series within the dataset
    colour_of = {"mpox": config.COLORS[0], "ebola": config.COLORS[1], "measles": config.COLORS[2]}
    marker_count = {"mpox": 0, "ebola": 0, "measles": 0}

    for dataset, series, label in order:
        row = summary[(summary["dataset"] == dataset) & (summary["series"] == series)].iloc[0]
        g = data[(data["dataset"] == dataset) & (data["series"] == series)]
        g = g[(g["week"] >= row["takeoff_week"]) & (g["week"] <= row["peak_week_number"])]

        # Cumulative cases counted from the take-off week, and time since take-off.
        # We use (weeks since take-off + 1) so the first point is at t = 1, not 0
        # (log 0 is undefined).
        cum = g["cases"].cumsum().to_numpy()
        t = np.arange(1, len(cum) + 1)
        keep = cum > 0
        m = config.MARKERS[marker_count[dataset]]
        marker_count[dataset] += 1
        for ax in axes:
            ax.plot(t[keep], cum[keep], color=colour_of[dataset], marker=m,
                    markersize=3.5, linewidth=1.3, label=label)

    axes[0].set_yscale("log")
    axes[0].set_title("(a) log cases vs time\nstraight line = exponential growth",
                      loc="left", color=config.TEXT_DARK)
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[1].set_title("(b) log cases vs log time\nstraight line = power-law growth",
                      loc="left", color=config.TEXT_DARK)
    for ax in axes:
        ax.set_xlabel("Weeks since take-off (+1)")
    axes[0].set_ylabel("Cumulative cases since take-off")
    # One shared legend below both panels, so it never covers the data
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=7.5,
               bbox_to_anchor=(0.5, -0.12))
    fig.tight_layout()
    save(fig, "growth_shape")


def main():
    data, summary, reports = load()
    plot_overview(data, summary)
    plot_ebola_cleaning(reports)
    plot_growth_shape(data, summary)


if __name__ == "__main__":
    main()
