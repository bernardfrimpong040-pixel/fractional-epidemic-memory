"""
16_compare_forecast.py -- Stage 6: model comparison by AIC and by forecasts.

What it does
  1. AIC for the real data: the six models fitted in Stage 5 (Caputo SEIR,
     AHM-SEIR, ordinary SEIR, SEIR with gamma estimated, declining
     transmission, heterogeneous recovery), compared by AIC, AICc and Akaike
     weights. (No new fits: the log-likelihoods come from Stage 5.)
  2. AIC for synthetic data with a KNOWN truth: the memoryless models are
     fitted to the Stage 4 data sets (the Caputo fits come from Stage 4), so
     we can see whether AIC picks the right kind of model: memory when the
     truth has memory, and the right memoryless mechanism when it has none.
  3. Rolling-origin forecasts for the real data: for every forecast origin o
     (8, 10, 12, ... weeks of data), every model is fitted to weeks 1..o only
     and forecasts weeks o+1..o+4. The forecasts are negative binomial with
     the fitted mean and dispersion ("plug-in": parameter uncertainty is not
     added) and are scored against what happened (log score, CRPS, 50% and
     90% interval coverage). Unlike AIC, this needs no parameter counting and
     no regularity conditions; it only asks which model predicts better.
  4. Summary tables.

Output (all in results/)
  stage6_aic.csv               step 1: one row per series and model
  stage6_synthetic_fits.csv    step 2: memoryless fits to synthetic data
  stage6_synthetic_aic.csv     step 2: which model AIC picks, per truth and noise level
  stage6_forecasts.csv         step 3: one row per series, model, origin and horizon
  stage6_forecast_summary.csv  step 4: mean scores per series and model
  stage6_forecast_horizon.csv  step 4: mean scores per model and horizon

Running it
    python src\\16_compare_forecast.py              (resumes; skips finished work)
    python src\\16_compare_forecast.py --fresh      (deletes old Stage 6 results first)
    python src\\16_compare_forecast.py --workers 3
The complete run is about 1,400 fits: roughly 3 hours with 2 cores.
"""

import argparse
import importlib
import math
import os
import time
import warnings

import numpy as np
import pandas as pd

import config
import fitting as F
import outbreaks as O
import scoring as SC
import synthetic as Y

fit13 = importlib.import_module("13_fit_outbreaks")   # settings, helpers and data variants

FIRST_ORIGIN = 8              # weeks of data at the first forecast origin
ORIGIN_STEP = 2               # a new origin every 2 weeks
HORIZONS = [1, 2, 3, 4]       # weeks ahead
MODELS = ["seir", "gfree", "decline", "hetero", "caputo", "ahm"]
SYN_NOISE = [10, 100]         # synthetic data: noise levels used
SYN_REPS = 10                 # synthetic data: data sets per truth and noise level
ODE = "LSODA"                 # ODE method for the memoryless fits (see solvers.solve_ode)

R = config.RESULTS_DIR
AIC_FILE = R / "stage6_aic.csv"
SYN_FITS = R / "stage6_synthetic_fits.csv"
SYN_AIC = R / "stage6_synthetic_aic.csv"
FORECASTS = R / "stage6_forecasts.csv"
FC_SUMMARY = R / "stage6_forecast_summary.csv"
FC_HORIZON = R / "stage6_forecast_horizon.csv"
FC_CI = R / "stage6_forecast_ci.csv"
B_ORIGINS = 2000              # bootstrap resamples of forecast origins (uncertainty of skill)
FIT_FILES = [AIC_FILE, SYN_FITS, SYN_AIC, FORECASTS, FC_SUMMARY, FC_HORIZON, FC_CI]


# ---------------------------------------------------------------------------
# 1. AIC for the real data
# ---------------------------------------------------------------------------
def real_aic():
    fits = pd.read_csv(R / "stage5_fits.csv")
    alt = pd.read_csv(R / "stage5_alt_fits.csv")
    rows = []
    for short in O.SHORT_NAMES:
        n = len(O.load_series(short)["y"])
        ll = {m: alt[(alt.series == short) & (alt.model == m)].loglik.iloc[0] for m in F.ALT_MODELS}
        for m in ("caputo", "ahm"):
            ll[m] = fits[(fits.series == short) & (fits.variant == "main") & (fits.model == m)].loglik_max.iloc[0]
        block = []
        for m in MODELS:
            k = SC.N_PARAMS[m]
            block.append({"series": short, "model": m, "n_weeks": n, "k": k, "loglik": ll[m],
                          "aic": SC.aic(ll[m], k), "aicc": SC.aicc(ll[m], k, n)})
        a = np.array([b["aic"] for b in block])
        c = np.array([b["aicc"] for b in block])
        for b, w, wc in zip(block, SC.akaike_weights(a), SC.akaike_weights(c)):
            b.update({"delta_aic": b["aic"] - a.min(), "weight_aic": w,
                      "delta_aicc": b["aicc"] - c.min(), "weight_aicc": wc})
        rows += block
    pd.DataFrame(rows).to_csv(AIC_FILE, index=False)
    d = pd.DataFrame(rows)
    best = d.loc[d.groupby("series").aicc.idxmin(), ["series", "model", "delta_aicc"]]
    print("[1] AIC for the real data. Lowest AICc:")
    print("    " + "; ".join(f"{s} {m}" for s, m in zip(best.series, best.model)))


# ---------------------------------------------------------------------------
# 2. AIC for synthetic data
# ---------------------------------------------------------------------------
def syn_task(task):
    warnings.simplefilter("ignore")
    truth, kd, rep, model = task
    d = pd.read_csv(config.PROJECT_DIR / "data" / "synthetic" / "replicates.csv")
    y = d[(d.truth == truth) & (d.k_noise == kd) & (d["rep"] == rep)].sort_values("week").cases.to_numpy(float)
    out = F.AltFitter(y, Y.SIGMA, Y.GAMMA, model, ode_method=ODE).fit()
    return {"truth": truth, "k_noise": kd, "rep": rep, "model": model, "loglik": out["loglik"],
            "at_bound": out["at_bound"], "n_weeks": len(y)}


def synthetic_aic(workers):
    plan = [(t, kd, r, m) for t in Y.TRUTH_IDS for kd in SYN_NOISE for r in range(SYN_REPS)
            for m in F.ALT_MODELS]
    done = set()
    if SYN_FITS.exists():
        f = pd.read_csv(SYN_FITS)
        done = set(zip(f.truth, f.k_noise, f.rep, f.model))
    tasks = [t for t in plan if t not in done]
    print(f"[2] Memoryless fits to synthetic data: {len(tasks)} of {len(plan)} to do")
    fit13.run_parallel(syn_task, tasks, workers,
                       lambda r: (fit13.append_rows(SYN_FITS, [r]),
                                  f"{r['truth']} k={r['k_noise']:<3} rep {r['rep']} {r['model']:<8} "
                                  f"loglik {r['loglik']:.2f}")[1], "syn")
    summarise_synthetic()


def summarise_synthetic():
    if not SYN_FITS.exists():
        return
    fits = pd.read_csv(SYN_FITS)
    cap = pd.read_csv(R / "stage4_replicate_fits.csv").drop_duplicates(["truth", "k_noise", "rep"])
    rows, per = [], []
    for (t, kd, r), g in fits.groupby(["truth", "k_noise", "rep"]):
        if len(g) < len(F.ALT_MODELS):
            continue
        c = cap[(cap.truth == t) & (cap.k_noise == kd) & (cap["rep"] == r)].iloc[0]
        ll = dict(zip(g.model, g.loglik))
        ll["caputo"] = c.loglik_max
        n = int(g.n_weeks.iloc[0])
        crit = {m: SC.aicc(ll[m], SC.N_PARAMS[m], n) for m in ll}
        best = min(crit, key=crit.get)
        best_memoryless = min((m for m in crit if m != "caputo"), key=crit.get)
        per.append({"truth": t, "k_noise": kd, "rep": r, "best": best,
                    "caputo_minus_best_memoryless": crit["caputo"] - crit[best_memoryless],
                    "seir_check": ll["seir"] - c.loglik_alpha1})
    per = pd.DataFrame(per)
    for (t, kd), g in per.groupby(["truth", "k_noise"]):
        row = {"truth": t, "k_noise": kd, "n": len(g),
               "median_caputo_minus_best_memoryless": g.caputo_minus_best_memoryless.median(),
               "max_abs_seir_check": g.seir_check.abs().max()}
        for m in ["caputo"] + list(F.ALT_MODELS):
            row[f"pct_best_{m}"] = 100.0 * np.mean(g.best == m)
        rows.append(row)
    pd.DataFrame(rows).to_csv(SYN_AIC, index=False)
    print(f"    synthetic AIC summary saved ({SYN_AIC.name})")


# ---------------------------------------------------------------------------
# 3. Rolling-origin forecasts
# ---------------------------------------------------------------------------
def origins(short):
    K = len(O.load_series(short)["y"])
    return list(range(FIRST_ORIGIN, K, ORIGIN_STEP))


def fit_forecast(y_fit, short, model, H):
    """Fit `model` to y_fit only and return (means for the next H weeks, k_disp, alpha).

    Only y_fit is ever passed to the fitting code, so the forecast cannot see
    the weeks it predicts.
    """
    o = len(y_fit)
    disease = fit13.disease_of(short)
    sigma, gamma = O.rates(disease)
    if model in ("caputo", "ahm"):
        settings = fit13.settings_for(short, "main", model)
        fit = F.ProfileFitter(y_fit, settings)
        fit.profile(fit13.ALPHA_GRID, stop_drop=fit13.STOP_DROP, global_restart=True)
        prof = fit.refine(step=0.01, half_width=0.04)
        b = max(prof, key=lambda p: p["loglik"])
        m = F.model_weekly(model, b["alpha"], b["beta"], b["e0"], o + H, sigma, gamma,
                           fit._row(b["alpha"]))
        mu = b["A"] * np.maximum(m, F.M_FLOOR)
        return mu[o:], b["k_disp"], b["alpha"], b["loglik"]
    fitter = F.AltFitter(y_fit, sigma, gamma, model, ode_method=ODE)
    out = fitter.fit()
    m = F.alt_weekly(model, F.alt_params(model, np.array(out["z"])), o + H, sigma, gamma, ODE)
    mu = out["A"] * np.maximum(m, F.M_FLOOR)
    return mu[o:], out["k_disp"], 1.0, out["loglik"]


def forecast_task(task):
    warnings.simplefilter("ignore")
    short, model, origin = task
    y = O.load_series(short)["y"]
    H = min(len(HORIZONS), len(y) - origin)
    t0 = time.perf_counter()
    mu, kd, alpha, ll = fit_forecast(y[:origin], short, model, H)
    obs = y[origin:origin + H]
    ls = SC.nb_logscore(obs, mu, kd)
    crps = SC.nb_crps(obs, mu, kd)
    rows = []
    for h in range(H):
        lo50, hi50 = SC.nb_interval(mu[h], kd, 0.5)
        lo90, hi90 = SC.nb_interval(mu[h], kd, 0.9)
        rows.append({"series": short, "model": model, "origin": origin, "horizon": h + 1,
                     "week": origin + h, "y": obs[h], "mean": mu[h], "k_disp": kd,
                     "alpha": alpha, "loglik_fit": ll, "logscore": ls[h], "crps": crps[h],
                     "abs_error": abs(obs[h] - mu[h]),
                     "in50": bool(lo50 <= obs[h] <= hi50), "in90": bool(lo90 <= obs[h] <= hi90),
                     "seconds": time.perf_counter() - t0})
    return rows


def forecasts(workers):
    plan = [(s, m, o) for s in O.SHORT_NAMES for o in origins(s) for m in MODELS]
    done = set()
    if FORECASTS.exists():
        f = pd.read_csv(FORECASTS)
        done = set(zip(f.series, f.model, f.origin))
    # slow fractional fits first, so the workers finish together
    tasks = sorted([t for t in plan if t not in done], key=lambda t: t[1] not in ("caputo", "ahm"))
    print(f"[3] Rolling-origin forecasts: {len(tasks)} of {len(plan)} fits to do")
    fit13.run_parallel(forecast_task, tasks, workers,
                       lambda rows: (fit13.append_rows(FORECASTS, rows),
                                     f"{rows[0]['series']:<12} {rows[0]['model']:<8} origin "
                                     f"{rows[0]['origin']:2d}  log score {np.mean([r['logscore'] for r in rows]):.2f}")[1],
                       "forecast")


# ---------------------------------------------------------------------------
# 4. Summaries
# ---------------------------------------------------------------------------
def summarise_forecasts():
    if not FORECASTS.exists():
        return
    f = pd.read_csv(FORECASTS).drop_duplicates(["series", "model", "origin", "horizon"])
    # keep only (series, origin) pairs where every model has a forecast, so
    # all models are scored on exactly the same weeks
    n_models = f.groupby(["series", "origin"]).model.nunique()
    full = n_models[n_models == len(MODELS)].index
    f = f.set_index(["series", "origin"]).loc[full].reset_index()
    base = f[f.model == "seir"].set_index(["series", "origin", "horizon"])[["logscore", "crps"]]
    f = f.join(base, on=["series", "origin", "horizon"], rsuffix="_seir")
    f["logscore_gain"] = f.logscore - f.logscore_seir           # > 0: better than SEIR
    f["crps_ratio"] = f.crps / f.crps_seir.where(f.crps_seir > 0)  # < 1: better than SEIR
    agg = {"logscore": "mean", "crps": "mean", "abs_error": "mean", "in50": "mean",
           "in90": "mean", "logscore_gain": "mean"}
    s = f.groupby(["series", "model"]).agg(agg).reset_index()
    s["n_forecasts"] = f.groupby(["series", "model"]).size().to_numpy()
    s["pct_beats_seir"] = f.groupby(["series", "model"]).apply(
        lambda g: 100.0 * np.mean(g.logscore > g.logscore_seir + 1e-9)).to_numpy()
    # medians: a few very poor early forecasts (log score below -1000) dominate means
    s["logscore_gain_median"] = f.groupby(["series", "model"]).logscore_gain.median().to_numpy()
    s["rel_crps"] = s.crps / s.merge(s[s.model == "seir"][["series", "crps"]], on="series",
                                     suffixes=("", "_seir")).crps_seir.to_numpy()
    allrows = f.groupby("model").agg(agg).reset_index()
    allrows["series"] = "all"
    allrows["n_forecasts"] = f.groupby("model").size().to_numpy()
    allrows["pct_beats_seir"] = f.groupby("model").apply(
        lambda g: 100.0 * np.mean(g.logscore > g.logscore_seir + 1e-9)).to_numpy()
    allrows["logscore_gain_median"] = f.groupby("model").logscore_gain.median().to_numpy()
    # overall relative CRPS: geometric mean over series of each series' ratio
    # (case numbers differ 100-fold between series, so a plain ratio of means
    # would be decided by the largest outbreaks)
    gm = s.groupby("model").rel_crps.apply(lambda v: float(np.exp(np.mean(np.log(v)))))
    allrows["rel_crps"] = allrows.model.map(gm)
    out = pd.concat([s, allrows], ignore_index=True)
    out.to_csv(FC_SUMMARY, index=False)
    h = f.groupby(["model", "horizon"]).agg(agg).reset_index()
    h.to_csv(FC_HORIZON, index=False)
    skill_intervals(f)
    print(f"[4] Forecast summaries saved ({FC_SUMMARY.name}, {FC_HORIZON.name})")
    show = allrows.set_index("model").loc[MODELS][["logscore_gain", "logscore_gain_median", "rel_crps",
                                                   "pct_beats_seir", "in50", "in90"]]
    print(show.round(3).to_string())


def skill_intervals(f):
    """95% intervals for overall forecast skill, by resampling forecast origins.

    Within each series, the origins (each with its 1-4 forecasts) are drawn
    with replacement and the summaries recomputed, B_ORIGINS times:
      rel_crps        geometric mean over series of (mean CRPS / mean CRPS of SEIR)
      logscore_gain   mean log-score gain over SEIR, all forecasts
    Both also without mpox, the series that dominates. Forecasts from nearby
    origins overlap and are correlated, so these intervals are, if anything,
    too narrow; they show the order of magnitude of the uncertainty.
    """
    rng = np.random.default_rng(2026)
    groups = {s: g for s, g in f.groupby("series")}
    orig = {s: np.sort(g.origin.unique()) for s, g in groups.items()}
    # per series and origin: sums of CRPS and log-score gain per model
    tab = f.groupby(["series", "origin", "model"]).agg(crps=("crps", "sum"),
                                                        gain=("logscore_gain", "sum"),
                                                        n=("crps", "size")).reset_index()
    cube = {s: tab[tab.series == s].pivot(index="origin", columns="model") for s in groups}

    def stats(pick, series):
        rel, gain, n = {m: [] for m in MODELS}, {m: 0.0 for m in MODELS}, 0
        for s in series:
            c = cube[s].loc[pick[s]]
            for m in MODELS:
                rel[m].append(c["crps"][m].sum() / c["crps"]["seir"].sum())
                gain[m] += c["gain"][m].sum()
            n += c["n"]["seir"].sum()
        return ({m: float(np.exp(np.mean(np.log(rel[m])))) for m in MODELS},
                {m: gain[m] / n for m in MODELS})

    rows = []
    for label, series in (("all", list(groups)), ("without mpox", [s for s in groups if s != "mpox-US"])):
        point = stats({s: orig[s] for s in series}, series)
        draws = [stats({s: rng.choice(orig[s], len(orig[s])) for s in series}, series)
                 for _ in range(B_ORIGINS)]
        for m in MODELS:
            rc = [d[0][m] for d in draws]
            lg = [d[1][m] for d in draws]
            rows.append({"set": label, "model": m,
                         "rel_crps": point[0][m], "rel_crps_low": np.quantile(rc, 0.025),
                         "rel_crps_high": np.quantile(rc, 0.975),
                         "logscore_gain": point[1][m], "logscore_gain_low": np.quantile(lg, 0.025),
                         "logscore_gain_high": np.quantile(lg, 0.975)})
    pd.DataFrame(rows).to_csv(FC_CI, index=False)
    print(f"    skill intervals saved ({FC_CI.name})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fresh", action="store_true", help="delete old Stage 6 results first")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2)))
    args = ap.parse_args()
    if args.fresh:
        for f in FIT_FILES:
            f.unlink(missing_ok=True)
    t0 = time.time()
    real_aic()
    forecasts(args.workers)
    synthetic_aic(args.workers)
    summarise_forecasts()
    print(f"Done in {(time.time() - t0) / 60:.1f} minutes.")


if __name__ == "__main__":
    main()
