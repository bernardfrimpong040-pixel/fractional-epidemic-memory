"""
10_synthetic_experiments.py -- Stage 4: does a memoryless process yield alpha < 1?

What it does
  1. Solver settings: for each alpha on the grid, the method (PECE or L1) and
     number of sub-steps that reach 0.2% accuracy (fitting.substep_table).
     Built once and saved (results/stage4_substeps*.csv); later runs reuse
     the saved tables so that results are identical on every computer.
  2. Data: the expected weekly counts of the eight truths (synthetic.py) and
     the noisy replicates, saved in data/synthetic/ for later stages.
  3. Noise-free fits ("infinite data"): the Caputo SEIR fitted to each truth's
     expected counts. The best alpha is the PSEUDO-TRUE alpha*: the value a
     fit converges to as (Poisson) data accumulate.
  4. Noisy replicate fits, in parallel: profile likelihood for alpha, the
     boundary likelihood-ratio test of alpha = 1 and the 95% interval.
       - main experiment: 8 truths x 3 noise levels (k = 1, 10, 100) x 30 data sets
       - calibration: 200 data sets from the ordinary SEIR (T0) at each noise
         level, to measure how often the test wrongly rejects alpha = 1, and to
         set a CALIBRATED critical value (the 95th percentile of the statistic
         when the ordinary SEIR is true)
  5. Robustness of the noise-free results (in parallel): pseudo-true alpha*
     for negative binomial data (k = 1, 10, 100); the fitted model given the
     wrong infectious period (gamma = 0.8 or 1.2 instead of 1); and data that
     stop 4 weeks after the peak.
  6. Validation of two shortcuts: the early stop of the profile, and the
     interpolated 95% intervals, checked against extra computed points.
  7. Summary tables, recomputed from the stored profiles, with 95% (Wilson)
     intervals for every percentage; rejection rates with the nominal critical
     value 2.706 and with the calibrated one.

Output (all in results/)
  stage4_substeps*.csv                 solver settings
  stage4_truths.csv                    the eight truths
  stage4_pseudotrue_summary.csv, _profiles.csv     step 3
  stage4_replicate_fits.csv, _profiles.csv         step 4 (one row per fit / per profile point)
  stage4_sensitivity.csv, _profiles.csv            step 5
  stage4_validation.csv                            step 6
  stage4_replicate_results.csv         step 7: final numbers for every fit
  stage4_summary.csv                   step 7: one row per truth and noise level
  stage4_calibration.csv               step 7: size of the test (200 data sets)

Running it
    python src\\10_synthetic_experiments.py              (resumes; skips finished work)
    python src\\10_synthetic_experiments.py --fresh      (deletes old fit results first)
    python src\\10_synthetic_experiments.py --workers 3  (number of parallel processes)

The complete run is about 1,300 fits of 10-20 seconds each, spread over the
computer's cores: roughly 5 hours with 2 cores, 2.5 with 4, 1.5 with 8. All
results are saved as they finish; if the run stops, start it again.
"""

import argparse
import os
import time
import traceback
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd

import config
import fitting as F
import synthetic as Y

ALPHA_GRID_FULL = [round(0.40 + 0.05 * i, 2) for i in range(13)]          # 0.40, 0.45, ..., 1.00
ALPHA_GRID_REPS = [0.40, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00]
NOISE_LEVELS = [1, 10, 100]   # negative binomial dispersion k (1 = very noisy, 100 = low noise)
REPS = 30                     # data sets per truth and noise level
CALIBRATION_REPS = 200        # data sets from T0 for measuring the test's error rate
CALIBRATION_NOISE = [1, 10, 100]
STOP_DROP = 15.0              # early stop of the profile (see fitting.ProfileFitter.profile)
GAMMA_WRONG = [0.8, 1.2]      # infectious-period misspecification (truth: gamma = 1)
WEEKS_AFTER_PEAK = 4          # truncated-window experiment

SYN_DIR = config.PROJECT_DIR / "data" / "synthetic"
R = config.RESULTS_DIR
PSEUDO_PROFILES = R / "stage4_pseudotrue_profiles.csv"
PSEUDO_SUMMARY = R / "stage4_pseudotrue_summary.csv"
REP_FILE = R / "stage4_replicate_fits.csv"
REP_PROFILES = R / "stage4_replicate_profiles.csv"
SENS_FILE = R / "stage4_sensitivity.csv"
SENS_PROFILES = R / "stage4_sensitivity_profiles.csv"
VALID_FILE = R / "stage4_validation.csv"
RESULTS_FILE = R / "stage4_replicate_results.csv"
SUMMARY_FILE = R / "stage4_summary.csv"
CALIB_FILE = R / "stage4_calibration.csv"
FIT_FILES = [PSEUDO_PROFILES, PSEUDO_SUMMARY, REP_FILE, REP_PROFILES, SENS_FILE, SENS_PROFILES,
             VALID_FILE, RESULTS_FILE, SUMMARY_FILE, CALIB_FILE]


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def append_rows(path, rows):
    """Append rows (list of dicts) to a CSV file.

    If the rows have columns the file does not have yet, the file is
    rewritten with the extra columns (left empty for the old rows), so
    files written by older versions of this script stay readable.
    """
    new = pd.DataFrame(rows)
    if not path.exists():
        new.to_csv(path, index=False)
        return
    header = list(pd.read_csv(path, nrows=0).columns)
    if set(new.columns) <= set(header):
        new.reindex(columns=header).to_csv(path, mode="a", header=False, index=False)
    else:
        old = pd.read_csv(path)
        pd.concat([old, new], ignore_index=True).to_csv(path, index=False)


def wilson(successes, n, z=1.96):
    """95% Wilson interval (in %) for a proportion: better than +-1.96 se for small n."""
    if n == 0:
        return np.nan, np.nan
    p = successes / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return 100 * max(0.0, centre - half), 100 * min(1.0, centre + half)


def load_or_build_table(gamma=1.0):
    """Solver settings for fits that assume infectious rate gamma (sigma = 1)."""
    name = "stage4_substeps.csv" if gamma == 1.0 else f"stage4_substeps_gamma{gamma}.csv"
    path = R / name
    if path.exists():
        tab = pd.read_csv(path)
    else:
        print(f"    building solver settings for gamma = {gamma} (about a minute) ...")
        tab = pd.DataFrame(F.substep_table(ALPHA_GRID_FULL, Y.SIGMA, gamma, Y.K_WEEKS))
        tab.to_csv(path, index=False)
    return {"sigma": Y.SIGMA, "gamma": gamma, "substeps": tab.to_dict("records")}


def run_parallel(func, tasks, workers, on_result, label):
    """Run func(task) for every task in parallel; on_result(output) saves each one.

    A task that fails is reported and skipped (the others continue); run the
    script again to retry it.
    """
    if not tasks:
        return
    t_start, n_done, n_failed = time.time(), 0, 0
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(func, task): task for task in tasks}
        for fut in as_completed(futures):
            n_done += 1
            try:
                msg = on_result(fut.result())
            except Exception:                                   # keep going; report it
                n_failed += 1
                msg = "FAILED: " + traceback.format_exc().strip().splitlines()[-1]
            eta = (time.time() - t_start) / n_done * (len(tasks) - n_done)
            print(f"    {label} {n_done:4d}/{len(tasks)}  {msg}   [about {eta / 60:.0f} min left]",
                  flush=True)
    if n_failed:
        print(f"    {n_failed} task(s) failed; run the script again to retry them")


# ---------------------------------------------------------------------------
# 2. Data
# ---------------------------------------------------------------------------
def all_datasets():
    """(truth, k_noise, rep) for every noisy data set used."""
    main = [(t, kd, r) for t in Y.TRUTH_IDS for kd in NOISE_LEVELS for r in range(REPS)]
    calib = [("T0", kd, r) for kd in CALIBRATION_NOISE for r in range(REPS, CALIBRATION_REPS)]
    return main + calib


def make_data():
    SYN_DIR.mkdir(parents=True, exist_ok=True)
    weeks = np.arange(1, Y.K_WEEKS + 1)
    exp_rows = [{"truth": t, "week": w, "expected": v}
                for t in Y.TRUTH_IDS for w, v in zip(weeks, Y.expected_counts(t))]
    pd.DataFrame(exp_rows).to_csv(SYN_DIR / "expected_counts.csv", index=False)
    rows = []
    for t, kd, r in all_datasets():
        rows += [{"truth": t, "k_noise": kd, "rep": r, "week": w, "cases": int(v)}
                 for w, v in zip(weeks, Y.replicate(t, kd, r))]
    pd.DataFrame(rows).to_csv(SYN_DIR / "replicates.csv", index=False)
    Y.describe_truths().to_csv(R / "stage4_truths.csv", index=False)
    print(f"[2] Saved expected counts and {len(all_datasets())} noisy data sets in data/synthetic/")


# ---------------------------------------------------------------------------
# 3 and 5. Noise-free fits (the pseudo-true alpha*) and their variants
# ---------------------------------------------------------------------------
def fit_noise_free(task):
    """Profile + refinement for one noise-free curve. Runs in a worker process."""
    warnings.simplefilter("ignore")
    truth_id, variant, settings, likelihood, n_weeks = task
    y = Y.expected_counts(truth_id)[:n_weeks]
    fit = F.ProfileFitter(y, settings, likelihood=likelihood)
    coarse = fit.profile(ALPHA_GRID_FULL)
    prof = fit.refine(step=0.01, half_width=0.04)
    s = F.summarise_profile(prof)
    on_grid = {round(p["alpha"], 6) for p in coarse}
    prof_rows = [{"truth": truth_id, "variant": variant, **p,
                  "rel_loglik": p["loglik"] - s["loglik_max"],
                  "refined_point": round(p["alpha"], 6) not in on_grid} for p in prof]
    best = min(prof, key=lambda p: abs(p["alpha"] - s["best_grid_alpha"]))
    t = next(x for x in Y.TRUTHS if x["id"] == truth_id)
    row = {"truth": truth_id, "variant": variant, "label": t["label"], "memory": t["memory"],
           "alpha_true": t["alpha_true"], "alpha_star": s["alpha_hat"],
           "deviance_alpha1": s["lrt"], "R0_true": Y.true_R0(truth_id),
           "R0_at_alpha_star": best["R0"], "R0_at_alpha1": prof[-1]["R0"],
           "beta_at_alpha_star": best["beta"], "floored_weeks": best["floored_weeks"],
           "weeks_fitted": n_weeks}
    return prof_rows, row


def noise_free_fits(workers):
    if PSEUDO_SUMMARY.exists():
        print("[3] Noise-free fits already done")
        summ = pd.read_csv(PSEUDO_SUMMARY)
        if "R0_true" not in summ.columns:                     # files from an older version
            summ.insert(summ.columns.get_loc("R0_at_alpha_star"), "R0_true",
                        [Y.true_R0(t) for t in summ.truth])
            summ.to_csv(PSEUDO_SUMMARY, index=False)
        return
    print("[3] Noise-free fits (refined to steps of 0.01 near the best alpha):")
    settings = load_or_build_table(1.0)
    tasks = [(t, "poisson", settings, "poisson_quasi", Y.K_WEEKS) for t in Y.TRUTH_IDS]
    out = {}

    def save(result):
        prof_rows, row = result
        out[row["truth"]] = (prof_rows, row)
        return f"{row['truth']}: alpha* = {row['alpha_star']:.3f}"

    run_parallel(fit_noise_free, tasks, workers, save, "noise-free")
    prof_rows = [r for t in Y.TRUTH_IDS for r in out[t][0]]
    pd.DataFrame(prof_rows).drop(columns="variant").to_csv(PSEUDO_PROFILES, index=False)
    pd.DataFrame([out[t][1] for t in Y.TRUTH_IDS]).drop(columns="variant").to_csv(PSEUDO_SUMMARY, index=False)


def sensitivity_fits(workers):
    done = set()
    if SENS_FILE.exists():
        old = pd.read_csv(SENS_FILE)
        done = set(zip(old.truth, old.variant))
    tasks = []
    base = load_or_build_table(1.0)
    for kd in NOISE_LEVELS:
        s = dict(base, k_fixed=float(kd))
        tasks += [(t, f"nb_k{kd}", s, "nb_expected", Y.K_WEEKS) for t in Y.TRUTH_IDS]
    for g in GAMMA_WRONG:
        s = load_or_build_table(g)
        tasks += [(t, f"gamma{g}", s, "poisson_quasi", Y.K_WEEKS) for t in Y.TRUTH_IDS]
    peaks = Y.describe_truths().set_index("id").peak_week
    tasks += [(t, "truncated", base, "poisson_quasi", int(peaks[t]) + WEEKS_AFTER_PEAK)
              for t in Y.TRUTH_IDS]
    tasks = [x for x in tasks if (x[0], x[1]) not in done]
    print(f"[5] Robustness of the noise-free fits: {len(tasks)} to do")

    def save(result):
        prof_rows, row = result
        append_rows(SENS_PROFILES, prof_rows)
        append_rows(SENS_FILE, [row])
        return f"{row['truth']} {row['variant']:9s}: alpha* = {row['alpha_star']:.3f}"

    run_parallel(fit_noise_free, tasks, workers, save, "robustness")


# ---------------------------------------------------------------------------
# 4. Noisy replicate fits
# ---------------------------------------------------------------------------
def fit_replicate(task):
    """Fit one noisy data set. Runs in a worker process; returns its profile and a summary row."""
    warnings.simplefilter("ignore")
    truth_id, kd, rep, settings = task
    y = Y.replicate(truth_id, kd, rep)
    t0 = time.time()
    fit = F.ProfileFitter(y, settings, likelihood="nb")
    prof = fit.profile(ALPHA_GRID_REPS, stop_drop=STOP_DROP)
    s = F.summarise_profile(prof)
    best = min(prof, key=lambda p: abs(p["alpha"] - s["best_grid_alpha"]))
    one = prof[-1]
    prof_rows = [{"truth": truth_id, "k_noise": kd, "rep": rep, "alpha": p["alpha"],
                  "loglik": p["loglik"], "beta": p["beta"], "e0": p["e0"], "A": p["A"],
                  "k_disp": p["k_disp"], "R0": p["R0"], "floored_weeks": p["floored_weeks"]}
                 for p in prof]
    row = {"truth": truth_id, "k_noise": kd, "rep": rep, **s,
           "beta_hat": best["beta"], "e0_hat": best["e0"], "A_hat": best["A"],
           "k_disp_hat": best["k_disp"], "R0_hat": best["R0"], "r_hat": best["r"],
           "R0_alpha1": one["R0"], "k_disp_alpha1": one["k_disp"],
           "floored_weeks": best["floored_weeks"], "lowest_alpha_fitted": prof[0]["alpha"],
           "n_solves": fit.n_solves, "seconds": time.time() - t0}
    return prof_rows, row


def replicate_fits(workers):
    done = set()
    if REP_FILE.exists():
        old = pd.read_csv(REP_FILE)
        done = set(zip(old.truth, old.k_noise, old["rep"]))
    settings = load_or_build_table(1.0)
    tasks = [(t, kd, r, settings) for t, kd, r in all_datasets() if (t, kd, r) not in done]
    print(f"[4] Replicate fits: {len(tasks)} to do ({len(done)} already done), "
          f"{workers} parallel processes")

    def save(result):
        prof_rows, row = result
        append_rows(REP_PROFILES, prof_rows)        # profile first: a fit counts as
        append_rows(REP_FILE, [row])                # done once its summary row exists
        return (f"{row['truth']} k={row['k_noise']:3d} rep {row['rep']:3d}: "
                f"alpha-hat {row['alpha_hat']:.3f}, LRT {row['lrt']:6.2f}"
                f"{'  REJECT alpha=1' if row['reject_alpha1'] else ''}")

    run_parallel(fit_replicate, tasks, workers, save, "fit")


# ---------------------------------------------------------------------------
# 6. Validation of the early stop and of the interval interpolation
# ---------------------------------------------------------------------------
def validate_one(task):
    """Add points to one stored profile and see whether the results change."""
    warnings.simplefilter("ignore")
    kind, truth_id, kd, rep, prof, settings = task
    y = Y.replicate(truth_id, kd, rep)
    fit = F.ProfileFitter(y, settings, likelihood="nb")
    fit.load(prof)
    before = F.summarise_profile(sorted(prof, key=lambda p: p["alpha"]))
    if kind == "early_stop":
        lowest = min(p["alpha"] for p in prof)
        extra = [a for a in ALPHA_GRID_FULL if a < lowest - 1e-9]
    else:                                                       # interval endpoints and maximum
        pts = set()
        for centre in (before["ci_low"], before["ci_high"], before["alpha_hat"]):
            pts |= {round(centre + d, 2) for d in np.arange(-0.03, 0.031, 0.01)}
        extra = sorted(a for a in pts if 0.40 <= a <= 1.0)
    after_prof = fit.fill(extra)
    after = F.summarise_profile(after_prof)
    new_pts = [p for p in after_prof if all(abs(p["alpha"] - q["alpha"]) > 1e-9 for q in prof)]
    return {"check": kind, "truth": truth_id, "k_noise": kd, "rep": rep,
            "points_added": len(new_pts),
            "best_new_minus_old_max": (max(p["loglik"] for p in new_pts) - before["loglik_max"])
            if new_pts else np.nan,
            "alpha_hat_before": before["alpha_hat"], "alpha_hat_after": after["alpha_hat"],
            "ci_low_before": before["ci_low"], "ci_low_after": after["ci_low"],
            "ci_high_before": before["ci_high"], "ci_high_after": after["ci_high"],
            "reject_before": before["reject_alpha1"], "reject_after": after["reject_alpha1"]}


def validation(workers):
    if VALID_FILE.exists():
        print("[6] Validation already done")
        return
    settings = load_or_build_table(1.0)
    fits = pd.read_csv(REP_FILE).drop_duplicates(["truth", "k_noise", "rep"], keep="last")
    profs = pd.read_csv(REP_PROFILES).drop_duplicates(["truth", "k_noise", "rep", "alpha"], keep="last")
    main = fits[fits["rep"] < REPS]
    # (a) early stop: fits that stopped above alpha = 0.40, two per truth (noisiest data first)
    stopped = main[main.lowest_alpha_fitted > 0.40 + 1e-9].sort_values(["truth", "k_noise", "rep"])
    pick_a = stopped.groupby("truth").head(2)
    # (b) intervals: the Caputo truths, two data sets per noise level
    pick_b = main[main.truth.isin(["T4", "T5"])].sort_values("rep").groupby(["truth", "k_noise"]).head(2)
    tasks = []
    for kind, pick in (("early_stop", pick_a), ("interval", pick_b)):
        for _, r in pick.iterrows():
            p = profs[(profs.truth == r.truth) & (profs.k_noise == r.k_noise) & (profs["rep"] == r["rep"])]
            tasks.append((kind, r.truth, int(r.k_noise), int(r["rep"]), p.to_dict("records"), settings))
    print(f"[6] Validation: {len(tasks)} stored profiles to extend")
    rows = []

    def save(row):
        rows.append(row)
        return (f"{row['check']:10s} {row['truth']} k={row['k_noise']} rep {row['rep']}: "
                f"alpha-hat {row['alpha_hat_before']:.3f} -> {row['alpha_hat_after']:.3f}")

    run_parallel(validate_one, tasks, workers, save, "validate")
    pd.DataFrame(rows).to_csv(VALID_FILE, index=False)


# ---------------------------------------------------------------------------
# 7. Summaries (recomputed from the stored profiles)
# ---------------------------------------------------------------------------
def summarise():
    fits = pd.read_csv(REP_FILE).drop_duplicates(["truth", "k_noise", "rep"], keep="last")
    profs = pd.read_csv(REP_PROFILES).drop_duplicates(["truth", "k_noise", "rep", "alpha"], keep="last")
    rows = []
    for (t, kd, rep), g in profs.groupby(["truth", "k_noise", "rep"]):
        f = fits[(fits.truth == t) & (fits.k_noise == kd) & (fits["rep"] == rep)]
        if f.empty:
            continue                           # profile written but fit not finished
        g = g.sort_values("alpha")
        s = F.summarise_profile(g.to_dict("records"))
        best = g.iloc[int(np.argmin(np.abs(g.alpha - s["best_grid_alpha"])))]
        rows.append({"truth": t, "k_noise": kd, "rep": rep, **s,
                     "R0_best": best.R0, "R0_alpha1": g.R0.iloc[-1],
                     "k_disp_best": best.k_disp, "floored_weeks_best": f.floored_weeks.iloc[0],
                     "lowest_alpha_fitted": g.alpha.iloc[0]})
    res = pd.DataFrame(rows)
    res.to_csv(RESULTS_FILE, index=False)

    out = []
    for t in Y.TRUTHS:
        for kd in NOISE_LEVELS:
            d = res[(res.truth == t["id"]) & (res.k_noise == kd) & (res["rep"] < REPS)]
            if d.empty:
                continue
            n, k = len(d), int(d.reject_alpha1.sum())
            lo, hi = wilson(k, n)
            row = {"truth": t["id"], "label": t["label"], "memory": t["memory"], "k_noise": kd,
                   "n": n, "alpha_hat_median": d.alpha_hat.median(),
                   "alpha_hat_q10": d.alpha_hat.quantile(0.1), "alpha_hat_q90": d.alpha_hat.quantile(0.9),
                   "reject_alpha1_pct": 100 * k / n, "reject_ci_low": lo, "reject_ci_high": hi,
                   "R0_ratio_median": (d.R0_best / d.R0_alpha1).median(),
                   "k_disp_hat_median": d.k_disp_best.median(),
                   "ci_split_pct": 100 * d.ci_split.mean()}
            a_true = t["alpha_true"]
            if a_true is not None:
                cov = int(((d.ci_low <= a_true + 1e-9) & (a_true - 1e-9 <= d.ci_high)).sum())
                c_lo, c_hi = wilson(cov, n)
                row.update({"ci_coverage_pct": 100 * cov / n, "coverage_ci_low": c_lo,
                            "coverage_ci_high": c_hi})
            out.append(row)
    summ = pd.DataFrame(out)

    # Calibration with ordinary-SEIR data (T0): the real size of the nominal
    # test, and a calibrated critical value = 95th percentile of the statistic
    cal, crit = [], {}
    for kd in CALIBRATION_NOISE:
        d = res[(res.truth == "T0") & (res.k_noise == kd)]
        n, k = len(d), int(d.reject_alpha1.sum())
        lo, hi = wilson(k, n)
        crit[kd] = max(F.LRT_CRIT_BOUNDARY, float(np.quantile(d.lrt, 0.95)))
        cal.append({"k_noise": kd, "n": n, "rejections": k, "size_pct": 100 * k / n,
                    "size_ci_low": lo, "size_ci_high": hi,
                    "lrt_zero_pct": 100 * (d.lrt == 0).mean(),
                    "chi2_mixture_expected_zero_pct": 50.0,
                    "lrt_q95": float(np.quantile(d.lrt, 0.95)),
                    "calibrated_critical_value": crit[kd]})
    pd.DataFrame(cal).to_csv(CALIB_FILE, index=False)

    # rejection rates with the calibrated critical value (the larger of 2.706
    # and the T0 95th percentile, so calibration can only make the test stricter)
    cal_cols = []
    for _, row in summ.iterrows():
        d = res[(res.truth == row.truth) & (res.k_noise == row.k_noise) & (res["rep"] < REPS)]
        k = int((d.lrt > crit[row.k_noise]).sum())
        lo, hi = wilson(k, len(d))
        cal_cols.append({"reject_calibrated_pct": 100 * k / len(d),
                         "reject_calibrated_ci_low": lo, "reject_calibrated_ci_high": hi})
    summ = pd.concat([summ, pd.DataFrame(cal_cols)], axis=1)
    summ.to_csv(SUMMARY_FILE, index=False)

    print("[7] Summary (results/stage4_summary.csv):")
    show = summ[["truth", "k_noise", "n", "alpha_hat_median", "reject_alpha1_pct",
                 "reject_calibrated_pct", "R0_ratio_median"]].round(2)
    print(show.to_string(index=False))
    print("    Size of the test with ordinary SEIR data (results/stage4_calibration.csv):")
    print(pd.DataFrame(cal).round(1).to_string(index=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fresh", action="store_true", help="delete previous Stage 4 fit results first")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    args = ap.parse_args()
    R.mkdir(parents=True, exist_ok=True)
    if args.fresh:
        for f in FIT_FILES:
            if f.exists():
                f.unlink()
        print("Deleted previous Stage 4 fit results (the solver-settings tables are kept).")

    print("[1] Solver settings:")
    for _, r in pd.DataFrame(load_or_build_table(1.0)["substeps"]).iterrows():
        print(f"    alpha {r.alpha:.2f}: {r.method:4s} with {int(r.k):3d} sub-steps/week "
              f"(max error {r.max_error:.1e})")
    make_data()
    noise_free_fits(args.workers)
    replicate_fits(args.workers)
    sensitivity_fits(args.workers)
    validation(args.workers)
    summarise()


if __name__ == "__main__":
    main()
