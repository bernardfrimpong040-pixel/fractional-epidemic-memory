"""
11_plot_synthetic.py -- figures for Section 5 (synthetic experiments).

  pseudotrue_profiles  noise-free profile likelihoods for the eight truths:
                       where the fit of alpha converges with unlimited data
  replicate_alpha      fitted alpha in 30 noisy data sets per truth, at three
                       noise levels, with the share of fits rejecting alpha = 1
  test_calibration     (a) the test statistic for 200 ordinary-SEIR data sets
                       against the 50:50 chi-square mixture; (b) how often each
                       truth rejects alpha = 1 at each noise level, with 95%
                       intervals
  example_fits         two memoryless outbreaks: the ordinary SEIR fit and the
                       Caputo fit at its best alpha (from the stored profiles)
and the table results/stage4_gain_by_phase.csv (see gain_by_phase).

Reads the CSV files written by 10_synthetic_experiments.py. Colour has one
meaning: blue = no memory in the true mechanism, orange = memory.
Saved in figures/ (.png and .pdf) and copied to paper/figures/. About 10 s.
"""

import shutil
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import chi2

import config
import fitting as F
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
NO_MEMORY, MEMORY = config.COLORS[0], config.COLORS[1]      # blue, orange
R = config.RESULTS_DIR
REPS = 30


def save(fig, name):
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    config.PAPER_FIG_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        path = config.FIGURES_DIR / f"{name}.{ext}"
        fig.savefig(path, dpi=200)
        shutil.copy(path, config.PAPER_FIG_DIR / path.name)
    plt.close(fig)
    print(f"Saved figures/{name}.png and .pdf (copied to paper/figures/)")


def colour(truth_id):
    return MEMORY if next(t for t in Y.TRUTHS if t["id"] == truth_id)["memory"] else NO_MEMORY


def label(truth_id):
    return next(t["label"] for t in Y.TRUTHS if t["id"] == truth_id)


# ---------------------------------------------------------------------------
def plot_profiles():
    prof = pd.read_csv(R / "stage4_pseudotrue_profiles.csv")
    summ = pd.read_csv(R / "stage4_pseudotrue_summary.csv")
    fig, axes = plt.subplots(2, 4, figsize=(7.4, 3.9), sharex=True, sharey=True)
    for ax, t in zip(axes.ravel(), Y.TRUTHS):
        d = prof[prof.truth == t["id"]].sort_values("alpha")
        s = summ[summ.truth == t["id"]].iloc[0]
        c = colour(t["id"])
        ax.plot(d.alpha, d.rel_loglik, color=c, marker="o", markersize=2.5, linewidth=1.5)
        ax.axhline(-F.CI_CRIT / 2, color=config.TEXT_SOFT, linestyle=":", linewidth=0.8)
        if t["alpha_true"] is not None and t["alpha_true"] < 1:
            ax.axvline(t["alpha_true"], color=config.TEXT_DARK, linestyle="--", linewidth=0.9)
        ax.plot([s.alpha_star], [0], marker="D", color=config.TEXT_DARK, markersize=4.5, zorder=5)
        ax.set_title(f"{t['id']} {t['label']}\n$\\alpha^*$ = {s.alpha_star:.2f}", loc="left",
                     color=config.TEXT_DARK, fontsize=8)
        ax.set_ylim(-120, 8)
        ax.set_xlim(0.38, 1.02)
    for ax in axes[1]:
        ax.set_xlabel("$\\alpha$")
    for ax in axes[:, 0]:
        ax.set_ylabel("$\\ell_p(\\alpha)-\\ell_{\\max}$")
    fig.tight_layout()
    save(fig, "pseudotrue_profiles")


# ---------------------------------------------------------------------------
def plot_replicates():
    res = pd.read_csv(R / "stage4_replicate_results.csv")
    res = res[res["rep"] < REPS]
    sens = pd.read_csv(R / "stage4_sensitivity.csv")
    noise = sorted(res.k_noise.unique())
    fig, axes = plt.subplots(1, len(noise), figsize=(7.4, 3.7), sharey=True)
    rng = np.random.default_rng(0)                      # jitter only (reproducible)
    for ax, kd in zip(axes, noise):
        for i, t in enumerate(Y.TRUTHS):
            d = res[(res.truth == t["id"]) & (res.k_noise == kd)]
            yy = len(Y.TRUTHS) - 1 - i
            c = colour(t["id"])
            rej = d.reject_alpha1.to_numpy().astype(bool)
            jit = rng.uniform(-0.22, 0.22, len(d))
            # filled dot = rejects alpha = 1 at the 5% level; open dot = does not
            ax.scatter(d.alpha_hat[rej], yy + jit[rej], s=9, color=c, alpha=0.85, linewidths=0)
            ax.scatter(d.alpha_hat[~rej], yy + jit[~rej], s=9, facecolors="none", edgecolors=c,
                       linewidths=0.6, alpha=0.9)
            st = sens[(sens.truth == t["id"]) & (sens.variant == f"nb_k{kd}")]
            if not st.empty:                              # NB pseudo-true alpha* at this noise level
                ax.plot([st.alpha_star.iloc[0]], [yy + 0.38], marker="v", color=config.TEXT_DARK,
                        markersize=3.5)
            ax.text(1.04, yy, f"{100 * rej.mean():.0f}%", va="center", fontsize=7,
                    color=config.TEXT_DARK)
        ax.set_title(f"$k$ = {kd}", loc="left", color=config.TEXT_DARK)
        ax.set_xlim(0.38, 1.12)
        ax.set_xticks([0.4, 0.6, 0.8, 1.0])
        ax.set_xlabel("Fitted $\\hat\\alpha$")
    axes[0].set_yticks(range(len(Y.TRUTHS)))
    axes[0].set_yticklabels([f"{t['id']} {t['label']}" for t in reversed(Y.TRUTHS)], fontsize=7.5)
    axes[0].set_ylim(-0.6, len(Y.TRUTHS) - 0.2)
    fig.tight_layout()
    save(fig, "replicate_alpha")


# ---------------------------------------------------------------------------
def plot_calibration():
    res = pd.read_csv(R / "stage4_replicate_results.csv")
    summ = pd.read_csv(R / "stage4_summary.csv")
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.2), gridspec_kw={"width_ratios": [1, 1.35]})

    # (a) empirical distribution of the LRT for ordinary-SEIR data vs the mixture
    ax = axes[0]
    x = np.linspace(0, 8, 400)
    ax.plot(x, 0.5 + 0.5 * chi2.cdf(x, 1), color=config.TEXT_SOFT, linestyle="--", linewidth=1.2,
            label="50:50 mixture")
    for i, kd in enumerate((10, 100)):
        d = np.sort(res[(res.truth == "T0") & (res.k_noise == kd)].lrt.to_numpy())
        ecdf = np.arange(1, len(d) + 1) / len(d)
        ax.step(np.r_[0, d], np.r_[np.mean(d == 0), ecdf], where="post", color=config.COLORS[i],
                linewidth=1.5, label=f"T0 data, $k$ = {kd} ($n$ = {len(d)})")
    ax.axvline(F.LRT_CRIT_BOUNDARY, color=config.TEXT_DARK, linestyle=":", linewidth=0.9)
    ax.set_xlim(0, 8)
    ax.set_ylim(0.4, 1.01)
    ax.set_xlabel("LRT statistic $\\Lambda$")
    ax.set_ylabel("Proportion $\\leq\\Lambda$")
    ax.set_title("(a) test statistic, ordinary SEIR data", loc="left", color=config.TEXT_DARK)
    ax.legend(loc="lower right", fontsize=7)

    # (b) rejection rates with 95% Wilson intervals
    ax = axes[1]
    markers = {1: "o", 10: "s", 100: "^"}
    offsets = {1: -0.22, 10: 0.0, 100: 0.22}
    for i, t in enumerate(Y.TRUTHS):
        yy = len(Y.TRUTHS) - 1 - i
        for kd in (1, 10, 100):
            d = summ[(summ.truth == t["id"]) & (summ.k_noise == kd)]
            if d.empty:
                continue
            d = d.iloc[0]
            ax.errorbar(d.reject_alpha1_pct, yy + offsets[kd],
                        xerr=[[d.reject_alpha1_pct - d.reject_ci_low], [d.reject_ci_high - d.reject_alpha1_pct]],
                        fmt=markers[kd], color=colour(t["id"]), markersize=4, elinewidth=0.8, capsize=0)
    for kd in (1, 10, 100):
        ax.plot([], [], markers[kd], color=config.TEXT_SOFT, label=f"$k$ = {kd}")
    ax.axvline(5, color=config.TEXT_DARK, linestyle=":", linewidth=0.9)
    ax.set_yticks(range(len(Y.TRUTHS)))
    ax.set_yticklabels([t["id"] for t in reversed(Y.TRUTHS)])
    ax.set_xlim(-2, 102)
    ax.set_xlabel("Data sets rejecting $\\alpha=1$ (%)")
    ax.set_title("(b) how often the test reports memory", loc="left", color=config.TEXT_DARK)
    ax.legend(loc="upper right", fontsize=7)
    fig.tight_layout()
    save(fig, "test_calibration")


# ---------------------------------------------------------------------------
def plot_examples():
    """Two memoryless data sets: curves at alpha = 1 and the best grid alpha, from stored fits.

    For each truth we show the TYPICAL data set at k = 100: the one whose test
    statistic is closest to the median over its 30 data sets.
    """
    tab = pd.read_csv(R / "stage4_substeps.csv").set_index("alpha")
    profs = pd.read_csv(R / "stage4_replicate_profiles.csv").drop_duplicates(
        ["truth", "k_noise", "rep", "alpha"], keep="last")
    res = pd.read_csv(R / "stage4_replicate_results.csv")
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.1), sharey=True)
    weeks = np.arange(1, Y.K_WEEKS + 1)
    for ax, tid, letter in zip(axes, ["T2", "T7"], "ab"):
        d = res[(res.truth == tid) & (res.k_noise == 100) & (res["rep"] < REPS)]
        r = d.iloc[int(np.argmin(np.abs(d.lrt - d.lrt.median())))]
        rep = int(r["rep"])
        p = profs[(profs.truth == tid) & (profs.k_noise == 100) & (profs["rep"] == rep)].sort_values("alpha")
        y = Y.replicate(tid, 100, rep)
        ax.scatter(weeks, y, s=12, color=config.TEXT_SOFT, zorder=3, label=f"data (data set {rep})")
        for a, c, ls in ((1.0, NO_MEMORY, "-"), (r.best_grid_alpha, MEMORY, "--")):
            row = p.iloc[int(np.argmin(np.abs(p.alpha - a)))]
            setting = tab.loc[round(row.alpha, 2)]
            m = F.caputo_weekly(row.alpha, row.beta, row.e0, Y.K_WEEKS, Y.SIGMA, Y.GAMMA,
                                setting.method, int(setting.k))
            ax.plot(weeks, row.A * m, color=c, linestyle=ls, linewidth=1.8,
                    label=f"$\\alpha$ = {row.alpha:.2f}" + (" (ordinary SEIR)" if row.alpha == 1 else ""))
        ax.set_yscale("log")
        ax.set_ylim(5, 3000)
        ax.set_xlabel("Week")
        ax.set_title(f"({letter}) {label(tid)}\nno memory; LRT for $\\alpha$=1: {r.lrt:.1f}", loc="left",
                     color=config.TEXT_DARK)
        ax.legend(loc="lower center", fontsize=7.5)
    axes[0].set_ylabel("Weekly cases")
    fig.tight_layout()
    save(fig, "example_fits")


def gain_by_phase():
    """For each truth's noise-free fit: where does the best alpha beat alpha = 1?

    Splits the Poisson log-likelihood gain of the best computed alpha over
    alpha = 1 into weeks up to the truth's peak and weeks after it, and gives
    the residual deviance 2 (l_saturated - l) of both fits (how far each
    fitted curve is from the truth). Saved to results/stage4_gain_by_phase.csv.
    """
    prof = pd.read_csv(R / "stage4_pseudotrue_profiles.csv")
    peaks = pd.read_csv(R / "stage4_truths.csv").set_index("id").peak_week
    tab = pd.read_csv(R / "stage4_substeps.csv")
    settings = {"sigma": Y.SIGMA, "gamma": Y.GAMMA, "substeps": tab.to_dict("records")}
    rows = []
    for t in Y.TRUTH_IDS:
        y = Y.expected_counts(t)
        fit = F.ProfileFitter(y, settings, "poisson_quasi")
        d = prof[prof.truth == t].sort_values("alpha")
        best, one = d.loc[d.loglik.idxmax()], d.iloc[-1]
        contrib = {}
        for name, p in (("best", best), ("one", one)):
            row = fit._row(p.alpha)
            m = F.caputo_weekly(p.alpha, p.beta, p.e0, Y.K_WEEKS, Y.SIGMA, Y.GAMMA, row["method"], row["k"])
            mu = p.A * np.maximum(m, F.M_FLOOR)
            contrib[name] = y * np.log(mu) - mu
        sat = y * np.log(y) - y
        gain = contrib["best"] - contrib["one"]
        k = int(peaks[t])
        rows.append({"truth": t, "best_alpha": best.alpha,
                     "gain_total": gain.sum(), "gain_up_to_peak": gain[:k].sum(),
                     "gain_after_peak": gain[k:].sum(),
                     "residual_deviance_best": 2 * (sat - contrib["best"]).sum(),
                     "residual_deviance_alpha1": 2 * (sat - contrib["one"]).sum()})
    out = pd.DataFrame(rows)
    out.to_csv(R / "stage4_gain_by_phase.csv", index=False)
    print("Saved results/stage4_gain_by_phase.csv")
    return out


def main():
    gain_by_phase()
    plot_profiles()
    plot_replicates()
    plot_calibration()
    plot_examples()


if __name__ == "__main__":
    main()
