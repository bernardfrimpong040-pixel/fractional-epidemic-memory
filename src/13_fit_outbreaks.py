"""
13_fit_outbreaks.py -- Stage 5: fit the models to the seven outbreak series.

What it does
  1. Solver settings for each disease (its own latent and infectious rates),
     for both fractional models and for gamma x 0.8, 1, 1.2. Built once and
     saved (results/stage5_substeps.csv) so every computer uses the same ones.
  2. Fits to the observed data (in parallel):
       - profile likelihood for alpha in the Caputo SEIR and in the AHM-SEIR,
         refined to steps of 0.01 around the best alpha;
       - the four memoryless alternatives: ordinary SEIR, SEIR with the
         infectious period estimated, SEIR with declining transmission, SEIR
         with heterogeneous recovery;
       - sensitivity variants of the two profiles: gamma x 0.8 and x 1.2,
         flagged reporting spikes left out (masked), and data stopping
         4 weeks after the peak.
  3. Pseudo-true alpha under each fitted memoryless alternative: the Caputo
     model fitted to the alternative's EXPECTED counts (no noise), as in
     Stage 4, with the expected negative binomial likelihood at the
     alternative's fitted dispersion (and, for comparison, the Poisson limit).
     This is the alpha that ever more data sets like the real one would give
     if the alternative were the truth.
  4. Parametric bootstrap (in parallel). New data sets are simulated from a
     fitted memoryless model (negative binomial noise with the fitted
     dispersion) and the fractional model is re-fitted to each, exactly as
     to the real data:
       - null = fitted ordinary SEIR: B_SEIR data sets; Caputo and AHM
         profiles. The fraction of simulated LRTs at least as large as the
         observed one is the BOOTSTRAP p-value of the test of alpha = 1.
       - null = fitted SEIR with estimated infectious period, declining-
         transmission or heterogeneous-recovery model: B_ALT data sets each;
         Caputo profile only. This shows which
         alpha-hat a memoryless but more realistic process produces.
  5. Residual checks: randomised quantile residuals (Dunn & Smyth 1996) of the
     best Caputo fit and of the four memoryless models.
  6. Summary tables.

Output (all in results/)
  stage5_substeps.csv          solver settings
  stage5_profiles.csv          every profile point (series, variant, model, alpha, ...)
  stage5_fits.csv              one row per profile: alpha-hat, interval, LRT
  stage5_alt_fits.csv          the memoryless alternatives' fits
  stage5_pseudotrue.csv        step 3
  stage5_bootstrap.csv         step 4 (one row per simulated data set and model)
  stage5_residuals.csv         step 5
  stage5_summary.csv           step 6: one row per series and model
  stage5_sensitivity.csv       step 6: alpha-hat and LRT for every variant

Running it
    python src\\13_fit_outbreaks.py              (resumes; skips finished work)
    python src\\13_fit_outbreaks.py --fresh      (deletes old Stage 5 fit results first)
    python src\\13_fit_outbreaks.py --workers 3  (number of parallel processes)
The complete run is about 1,100 profile fits of 10-60 seconds each:
roughly 7 hours with 2 cores, 3.5 with 4. Results are saved as they finish;
if the run stops, start it again.
"""

import argparse
import math
import os
import time
import traceback
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
from scipy.stats import nbinom, norm

import config
import fitting as F
import outbreaks as O

ALPHA_GRID = [round(0.40 + 0.05 * i, 2) for i in range(13)]       # 0.40, 0.45, ..., 1.00
STOP_DROP = 15.0              # early stop of the profile (fitting.ProfileFitter.profile)
GAMMA_FACTORS = [0.8, 1.2]    # sensitivity: infectious rate gamma x 0.8 and x 1.2
WEEKS_AFTER_PEAK = 4          # sensitivity: data stopping 4 weeks after the peak
B_SEIR = 60                   # bootstrap data sets from the fitted ordinary SEIR
B_AHM = 30                    #   ... of which the AHM-SEIR is re-fitted to the first 30
B_ALT = 20                    # bootstrap data sets from each fitted alternative
FRACTIONAL = ["caputo", "ahm"]
NULLS = ["seir", "decline", "hetero", "gfree"]

R = config.RESULTS_DIR
SUBSTEPS = R / "stage5_substeps.csv"
PROFILES = R / "stage5_profiles.csv"
FITS = R / "stage5_fits.csv"
ALT_FITS = R / "stage5_alt_fits.csv"
PSEUDO = R / "stage5_pseudotrue.csv"
BOOT = R / "stage5_bootstrap.csv"
RESID = R / "stage5_residuals.csv"
SUMMARY = R / "stage5_summary.csv"
SENS = R / "stage5_sensitivity.csv"
FIT_FILES = [PROFILES, FITS, ALT_FITS, PSEUDO, BOOT, RESID, SUMMARY, SENS]


# ---------------------------------------------------------------------------
# Small helpers (the same as in 10_synthetic_experiments.py)
# ---------------------------------------------------------------------------
def append_rows(path, rows):
    """Append rows (list of dicts) to a CSV file (adding new columns if needed)."""
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


def run_parallel(func, tasks, workers, on_result, label):
    """Run func(task) for every task in parallel; on_result(output) saves each one.
    A task that fails is reported and skipped; run the script again to retry it."""
    if not tasks:
        return
    t_start, n_done, n_failed = time.time(), 0, 0
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(func, task): task for task in tasks}
        for fut in as_completed(futures):
            n_done += 1
            try:
                msg = on_result(fut.result())
            except Exception:
                n_failed += 1
                msg = "FAILED: " + traceback.format_exc().strip().splitlines()[-1]
            eta = (time.time() - t_start) / n_done * (len(tasks) - n_done)
            print(f"    {label} {n_done:4d}/{len(tasks)}  {msg}   [about {eta / 60:.0f} min left]",
                  flush=True)
    if n_failed:
        print(f"    {n_failed} task(s) failed; run the script again to retry them")


def disease_of(short):
    return next(s[0] for s in O.SERIES if s[2] == short)


def series_data(short, variant):
    """Weekly counts for one series and data variant ('main', 'gamma0.8',
    'gamma1.2', 'masked' or 'truncated')."""
    d = O.load_series(short, mask_flagged=(variant == "masked"))
    y = d["y"]
    if variant == "truncated":
        y = O.truncate_after_peak(y, WEEKS_AFTER_PEAK, flag=d["flag"])
    return y


def gamma_factor(variant):
    return float(variant[5:]) if variant.startswith("gamma") else 1.0


def variants_for(short):
    """Data variants used for a series: 'masked' only if it has flagged weeks."""
    v = ["main"] + [f"gamma{g}" for g in GAMMA_FACTORS]
    if O.load_series(short)["flag"].any():
        v.append("masked")
    return v + ["truncated"]


# ---------------------------------------------------------------------------
# 1. Solver settings
# ---------------------------------------------------------------------------
def max_weeks(disease):
    return max(len(O.load_series(s[2])["y"]) for s in O.SERIES if s[0] == disease)


def build_table(task):
    warnings.simplefilter("ignore")
    disease, gfac, model = task
    sigma, gamma = O.rates(disease, gfac)
    tab = F.substep_table(ALPHA_GRID, sigma, gamma, max_weeks(disease), model=model)
    return [{"disease": disease, "gamma_factor": gfac, "model": model, **row} for row in tab]


def solver_tables(workers):
    tasks = [(d, g, m) for d in O.DISEASES for g in [1.0] + GAMMA_FACTORS for m in FRACTIONAL]
    if SUBSTEPS.exists():
        have = pd.read_csv(SUBSTEPS)
        done = set(zip(have.disease, have.gamma_factor, have.model))
        tasks = [t for t in tasks if t not in done]
    print(f"[1] Solver settings: {len(tasks)} table(s) to build")
    run_parallel(build_table, tasks, workers,
                 lambda rows: (append_rows(SUBSTEPS, rows), f"{rows[0]['disease']} "
                               f"gamma x {rows[0]['gamma_factor']} {rows[0]['model']}")[1],
                 "table")


def settings_for(short, variant, model):
    """Fitting settings (sigma, gamma, solver table, model) for one fit."""
    disease = disease_of(short)
    gfac = gamma_factor(variant)
    sigma, gamma = O.rates(disease, gfac)
    tab = pd.read_csv(SUBSTEPS)
    tab = tab[(tab.disease == disease) & np.isclose(tab.gamma_factor, gfac) & (tab.model == model)]
    return {"sigma": sigma, "gamma": gamma, "model": model, "wide_start": True,
            "substeps": tab[["alpha", "method", "k", "max_error"]].to_dict("records")}


# ---------------------------------------------------------------------------
# 2. Fits to the observed data
# ---------------------------------------------------------------------------
def profile_rows(y, settings, labels):
    """Profile + refinement; returns (profile rows, summary row)."""
    fit = F.ProfileFitter(y, settings)
    coarse = fit.profile(ALPHA_GRID, stop_drop=STOP_DROP, global_restart=True)
    prof = fit.refine(step=0.01, half_width=0.04)
    s = F.summarise_profile(prof)
    on_grid = {round(p["alpha"], 6) for p in coarse}
    rows = [{**labels, **p, "on_grid": round(p["alpha"], 6) in on_grid} for p in prof]
    best = max(prof, key=lambda p: p["loglik"])
    one = next(p for p in prof if p["alpha"] == 1.0)
    summ = {**labels, **s, "n_weeks": int(np.isfinite(y).sum()),
            "R0_best": best["R0"], "r_best": best["r"], "k_disp_best": best["k_disp"],
            "R0_alpha1": one["R0"], "k_disp_alpha1": one["k_disp"],
            "floored_weeks_best": best["floored_weeks"], "n_solves": fit.n_solves}
    return rows, summ


def fit_profile_task(task):
    warnings.simplefilter("ignore")
    short, variant, model, settings = task
    y = series_data(short, variant)
    rows, summ = profile_rows(y, settings, {"series": short, "variant": variant, "model": model})
    return {"kind": "profile", "rows": rows, "summ": summ}


def fit_alt_task(task):
    warnings.simplefilter("ignore")
    short, alt = task
    sigma, gamma = O.rates(disease_of(short))
    y = series_data(short, "main")
    out = F.AltFitter(y, sigma, gamma, alt).fit()
    out["z"] = " ".join(f"{v:.10g}" for v in out["z"])
    return {"kind": "alt", "row": {"series": short, **out}}


def save_fit(res):
    if res["kind"] == "profile":
        append_rows(PROFILES, res["rows"])
        append_rows(FITS, [res["summ"]])
        s = res["summ"]
        return (f"{s['series']:<12} {s['variant']:<10} {s['model']:<7} alpha-hat {s['alpha_hat']:.3f} "
                f"LRT {s['lrt']:.2f}")
    append_rows(ALT_FITS, [res["row"]])
    r = res["row"]
    return f"{r['series']:<12} {r['model']:<8} loglik {r['loglik']:.2f}"


def observed_fits(workers):
    done_p, done_a = set(), set()
    if FITS.exists():
        f = pd.read_csv(FITS)
        done_p = set(zip(f.series, f.variant, f.model))
    if PROFILES.exists():
        # a run stopped between saving a profile and its summary leaves profile
        # rows without a summary; remove them (the profile is refitted)
        p = pd.read_csv(PROFILES)
        keep = [k in done_p for k in zip(p.series, p.variant, p.model)]
        if not all(keep):
            p[keep].to_csv(PROFILES, index=False)
    if ALT_FITS.exists():
        a = pd.read_csv(ALT_FITS)
        done_a = set(zip(a.series, a.model))
    tasks_a = [(s, m) for s in O.SHORT_NAMES for m in F.ALT_MODELS if (s, m) not in done_a]
    tasks_p = [(s, v, m, settings_for(s, v, m)) for s in O.SHORT_NAMES for v in variants_for(s)
               for m in FRACTIONAL if (s, v, m) not in done_p]
    # main profiles first (other steps need them), then the sensitivity variants
    tasks_p.sort(key=lambda t: t[1] != "main")
    print(f"[2] Fits to the observed data: {len(tasks_a)} alternative fit(s), "
          f"{len(tasks_p)} profile(s) to do")
    run_parallel(fit_alt_task, tasks_a, workers, save_fit, "alt")
    run_parallel(fit_profile_task, tasks_p, workers, save_fit, "profile")


# ---------------------------------------------------------------------------
# 3 and 4. What a memoryless truth produces: pseudo-true alpha and bootstrap
# ---------------------------------------------------------------------------
def null_mean(short, null):
    """Expected weekly counts and dispersion of a fitted memoryless model."""
    sigma, gamma = O.rates(disease_of(short))
    a = pd.read_csv(ALT_FITS)
    row = a[(a.series == short) & (a.model == null)].iloc[0]
    z = np.array([float(v) for v in row.z.split()])
    y = series_data(short, "main")
    fit = F.AltFitter(y, sigma, gamma, null)
    return fit.fitted_mean(z, row.A), float(row.k_disp)


def pseudo_task(task):
    """alpha* for one fitted alternative, for negative binomial data with the
    alternative's own fitted dispersion (the noise model of the real fits) and,
    for comparison, in the Poisson limit (as in Stage 4)."""
    warnings.simplefilter("ignore")
    short, null, settings = task
    mu, kd = null_mean(short, null)
    out = {"series": short, "null": null, "k_disp": kd}
    for lik, tag in (("nb_expected", ""), ("poisson_quasi", "_poisson")):
        fit = F.ProfileFitter(mu, {**settings, "k_fixed": kd}, likelihood=lik)
        fit.profile(ALPHA_GRID, global_restart=True)
        s = F.summarise_profile(fit.refine(step=0.01, half_width=0.04))
        out["alpha_star" + tag] = s["alpha_hat"]
        out["deviance_alpha1" + tag] = s["lrt"]
    return out


def pseudo_true(workers):
    done = set()
    if PSEUDO.exists():
        p = pd.read_csv(PSEUDO)
        done = set(zip(p.series, p.null))
    tasks = [(s, n, settings_for(s, "main", "caputo")) for s in O.SHORT_NAMES
             for n in ("gfree", "decline", "hetero") if (s, n) not in done]
    print(f"[3] Pseudo-true alpha under the fitted alternatives: {len(tasks)} to do")
    run_parallel(pseudo_task, tasks, workers,
                 lambda r: (append_rows(PSEUDO, [r]),
                            f"{r['series']:<12} {r['null']:<8} alpha* {r['alpha_star']:.3f}")[1],
                 "pseudo")


def boot_data(short, null, rep):
    """One simulated data set; the seed is built from (series, null, rep)."""
    mu, kd = null_mean(short, null)
    rng = np.random.default_rng([O.SHORT_NAMES.index(short), NULLS.index(null), int(rep), 5])
    return rng.negative_binomial(kd, kd / (kd + mu)).astype(float)


def boot_task(task):
    warnings.simplefilter("ignore")
    short, null, rep, model, settings = task
    y = boot_data(short, null, rep)
    fit = F.ProfileFitter(y, settings)
    fit.profile(ALPHA_GRID, stop_drop=STOP_DROP, global_restart=True)
    s = F.summarise_profile(fit.refine(step=0.01, half_width=0.04))
    return {"series": short, "null": null, "rep": rep, "model": model,
            "alpha_hat": s["alpha_hat"], "lrt": s["lrt"], "ci_low": s["ci_low"],
            "ci_high": s["ci_high"], "loglik_max": s["loglik_max"],
            "loglik_alpha1": s["loglik_alpha1"], "n_solves": fit.n_solves}


def bootstrap(workers):
    done = set()
    if BOOT.exists():
        b = pd.read_csv(BOOT)
        done = set(zip(b.series, b["null"], b.rep, b.model))
    plan = []
    for s in O.SHORT_NAMES:
        plan += [(s, "seir", r, "caputo") for r in range(B_SEIR)]
        plan += [(s, "seir", r, "ahm") for r in range(B_AHM)]
        plan += [(s, n, r, "caputo") for n in ("gfree", "decline", "hetero") for r in range(B_ALT)]
    settings = {(s, m): settings_for(s, "main", m) for s in O.SHORT_NAMES for m in FRACTIONAL}
    # order: replicate number first, so that every series gets results early
    plan.sort(key=lambda t: (t[2], O.SHORT_NAMES.index(t[0]), NULLS.index(t[1]), t[3]))
    tasks = [(s, n, r, m, settings[(s, m)]) for s, n, r, m in plan if (s, n, r, m) not in done]
    print(f"[4] Parametric bootstrap: {len(tasks)} of {len(plan)} fits to do")
    run_parallel(boot_task, tasks, workers,
                 lambda r: (append_rows(BOOT, [r]),
                            f"{r['series']:<12} null {r['null']:<8} {r['model']:<7} rep {r['rep']:3d} "
                            f"alpha-hat {r['alpha_hat']:.2f} LRT {r['lrt']:.2f}")[1],
                 "boot")


# ---------------------------------------------------------------------------
# 5. Residual checks
# ---------------------------------------------------------------------------
def quantile_residuals(y, mu, kd, rng):
    """Randomised quantile residuals (Dunn & Smyth 1996).

    For a count y with distribution function G, draw u uniformly between
    G(y - 1) and G(y) and return Phi^(-1)(u). If the model is right these are
    independent standard normal values, so patterns in them (runs of the same
    sign, autocorrelation) show where the model fails.
    """
    p = kd / (kd + mu)
    lo = nbinom.cdf(y - 1, kd, p)
    hi = nbinom.cdf(y, kd, p)
    u = lo + rng.uniform(size=len(y)) * (hi - lo)
    return norm.ppf(np.clip(u, 1e-12, 1 - 1e-12))


def residuals():
    prof = pd.read_csv(PROFILES)
    prof = prof[prof.variant == "main"]
    alt = pd.read_csv(ALT_FITS)
    rows = []
    for i, short in enumerate(O.SHORT_NAMES):
        y = series_data(short, "main")
        sigma, gamma = O.rates(disease_of(short))
        fits = {}
        p = prof[(prof.series == short) & (prof.model == "caputo")]
        b = p.loc[p.loglik.idxmax()]
        fitter = F.ProfileFitter(y, settings_for(short, "main", "caputo"))
        fits["caputo"] = (fitter.fitted_mean(b.alpha, b.beta, b.e0, b.A), b.k_disp, b.alpha)
        for m in F.ALT_MODELS:
            r = alt[(alt.series == short) & (alt.model == m)].iloc[0]
            z = np.array([float(v) for v in r.z.split()])
            fits[m] = (F.AltFitter(y, sigma, gamma, m).fitted_mean(z, r.A), r.k_disp, 1.0)
        for j, (m, (mu, kd, a)) in enumerate(fits.items()):
            rng = np.random.default_rng([i, j, 55])
            q = quantile_residuals(y, mu, kd, rng)
            rows += [{"series": short, "model": m, "alpha": a, "week": w, "y": y[w],
                      "mean": mu[w], "k_disp": kd, "residual": q[w]} for w in range(len(y))]
    pd.DataFrame(rows).to_csv(RESID, index=False)
    print(f"[5] Residuals saved ({len(rows)} rows)")


# ---------------------------------------------------------------------------
# 6. Summary tables
# ---------------------------------------------------------------------------
def lag1(x):
    x = np.asarray(x) - np.mean(x)
    return float(np.sum(x[1:] * x[:-1]) / np.sum(x * x))


def summarise():
    fits = pd.read_csv(FITS)
    alt = pd.read_csv(ALT_FITS)
    boot = pd.read_csv(BOOT) if BOOT.exists() else pd.DataFrame(columns=["series"])
    pseudo = pd.read_csv(PSEUDO) if PSEUDO.exists() else pd.DataFrame(columns=["series"])
    res = pd.read_csv(RESID) if RESID.exists() else pd.DataFrame(columns=["series"])
    rows = []
    for short in O.SHORT_NAMES:
        a = alt[alt.series == short].set_index("model")
        for model in FRACTIONAL:
            f = fits[(fits.series == short) & (fits.variant == "main") & (fits.model == model)].iloc[0]
            row = {"series": short, "model": model, "n_weeks": f.n_weeks,
                   "alpha_hat": f.alpha_hat, "ci_low": f.ci_low, "ci_high": f.ci_high,
                   "ci_low_at_grid_edge": f.ci_low_at_grid_edge, "lrt": f.lrt,
                   "p_asymptotic": f.p_value, "loglik_max": f.loglik_max,
                   "loglik_seir": a.loc["seir", "loglik"],
                   "loglik_gfree": a.loc["gfree", "loglik"],
                   "loglik_decline": a.loc["decline", "loglik"],
                   "loglik_hetero": a.loc["hetero", "loglik"],
                   "R0_best": f.R0_best, "R0_seir": a.loc["seir", "R0"],
                   "k_disp_best": f.k_disp_best, "k_disp_seir": a.loc["seir", "k_disp"],
                   "floored_weeks_best": f.floored_weeks_best}
            b = boot[(boot.series == short) & (boot["null"] == "seir") & (boot.model == model)]
            if len(b):
                row["B_seir"] = len(b)
                row["p_bootstrap"] = (1 + np.sum(b.lrt >= f.lrt - 1e-9)) / (len(b) + 1)
                row["boot_lrt95"] = float(np.quantile(b.lrt, 0.95))
                row["boot_size_nominal_pct"] = 100 * float(np.mean(b.lrt > F.LRT_CRIT_BOUNDARY))
            if model == "caputo":
                for n in ("gfree", "decline", "hetero"):
                    b = boot[(boot.series == short) & (boot["null"] == n) & (boot.model == model)]
                    ps = pseudo[(pseudo.series == short) & (pseudo["null"] == n)]
                    if len(ps):
                        row[f"alpha_star_{n}"] = float(ps.alpha_star.iloc[0])
                    if len(b):
                        row[f"B_{n}"] = len(b)
                        row[f"median_alpha_{n}"] = float(b.alpha_hat.median())
                        row[f"q05_alpha_{n}"] = float(b.alpha_hat.quantile(0.05))
                        row[f"q95_alpha_{n}"] = float(b.alpha_hat.quantile(0.95))
                        row[f"frac_alpha_le_obs_{n}"] = float(np.mean(b.alpha_hat <= f.alpha_hat + 1e-9))
                        row[f"frac_lrt_ge_obs_{n}"] = float(np.mean(b.lrt >= f.lrt - 1e-9))
                r = res[(res.series == short) & (res.model == "caputo")]
                if len(r):
                    row["resid_lag1_best"] = lag1(r.residual)
                r = res[(res.series == short) & (res.model == "seir")]
                if len(r):
                    row["resid_lag1_seir"] = lag1(r.residual)
            rows.append(row)
    pd.DataFrame(rows).to_csv(SUMMARY, index=False)
    sens = fits[["series", "variant", "model", "n_weeks", "alpha_hat", "ci_low", "ci_high",
                 "lrt", "p_value", "R0_best", "R0_alpha1"]]
    sens.to_csv(SENS, index=False)
    print(f"[6] Summary saved: {SUMMARY.name}, {SENS.name}")
    show = pd.DataFrame(rows)[["series", "model", "alpha_hat", "ci_low", "ci_high", "lrt"]
                              + [c for c in ("p_bootstrap",) if c in rows[0]]]
    print(show.round(3).to_string(index=False))


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fresh", action="store_true", help="delete old Stage 5 fit results first")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2)))
    ap.add_argument("--skip-bootstrap", action="store_true",
                    help="do steps 1-3 and 5-6 only (minutes instead of hours)")
    args = ap.parse_args()
    if args.fresh:
        for f in FIT_FILES:
            f.unlink(missing_ok=True)
    t0 = time.time()
    solver_tables(args.workers)
    observed_fits(args.workers)
    pseudo_true(args.workers)
    if not args.skip_bootstrap:
        bootstrap(args.workers)
    residuals()
    summarise()
    print(f"Done in {(time.time() - t0) / 60:.1f} minutes.")


if __name__ == "__main__":
    main()
