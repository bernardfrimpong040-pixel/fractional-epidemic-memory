# Memory or misspecification? When does a fractional-order epidemic model measure real memory?

Research code for a paper on fractional-order (Caputo) epidemic models. The
question: when a fractional model fitted to outbreak data reports an order
alpha < 1, is that real "memory", or is the model absorbing something an
ordinary model would explain (gamma-distributed stage durations, falling
transmission, reporting delays)?

Python throughout. All data are public and pinned to fixed GitHub commits.

**Target journal:** Bulletin of Mathematical Biology (subscription route, no fee).
Back-ups: Journal of Mathematical Biology, Chaos, Solitons & Fractals.

## Stages

| Stage | Content | Status |
|---|---|---|
| 1 | Data: download, clean, weekly series, figures, checks | **done** |
| 2 | Models and theory: Caputo SEIR (rates^alpha), Angstmann-Henry-McGann fractional-recovery SIR, integer-order alternatives; existence, uniqueness, positivity, R0, Matignon stability, identifiability proposition | **done** |
| 3 | Numerical solver: Garrappa predictor-corrector / L1 scheme, Mittag-Leffler convergence test | **done** |
| 4 | Synthetic tests: does memoryless data give a spurious alpha < 1? | **done** |
| 5 | Fitting to the seven series, profile likelihood for alpha | next |
| 6 | AIC comparison and rolling-origin forecasts | |
| 7 | Writing, journal template, submission | |

## Project layout

```
Fractional_paper/
  src/
    config.py                 all paths, pinned URLs, fingerprints and rules
    01_download_data.py       download the 4 raw files, verify fingerprints
    02_clean_data.py          build the 7 weekly series + summary table
    03_plot_data.py           Stage 1 figures
    04_check_data.py          12 independent checks (all must PASS)
    mittag_leffler.py         Mittag-Leffler function E_{a,b}(z) (series + integral formula)
    models.py                 all six models and every formula of Section 3
    05_check_theory.py        21 numerical checks of the theory (all must PASS)
    06_plot_theory.py         Stage 2 figures
    solvers.py                PECE and L1 solvers, AHM solver, ODE models
    07_check_solvers.py       21 solver validation checks (all must PASS)
    08_plot_solvers.py        Stage 3 figures
    fitting.py                profile-likelihood fitting of the Caputo SEIR (NB counts)
    synthetic.py              the eight true models of Stage 4 and their data
    09_check_fitting.py       19 checks of the fitting machinery (all must PASS)
    10_synthetic_experiments.py  Stage 4 experiment (noise-free, noisy, calibration,
                              robustness and validation fits)
    11_plot_synthetic.py      Stage 4 figures
  data/
    raw/                      downloads (not on GitHub; re-created by script 01)
    processed/
      mpox_us_weekly.csv
      ebola_reports.csv       raw and cleaned cumulative counts per report
      ebola_weekly.csv
      measles_weekly.csv
      all_series_weekly.csv   all 7 series in one table (used by later stages)
    synthetic/
      expected_counts.csv     expected weekly cases of the 8 truths
      replicates.csv          the 1,230 noisy data sets (reused in Stage 6)
  results/
    data_sources.csv          where each file came from, commit, fingerprint
    stage1_data_summary.csv   one line per series (Table 2 of the paper)
    stage3_convergence.csv    solver errors by step size (Table 4 of the paper)
    stage4_substeps.csv       solver and sub-steps used for each alpha in fits
    stage4_truths.csv         the 8 truths: peak week, totals
    stage4_pseudotrue_summary.csv / _profiles.csv   noise-free fits (alpha*)
    stage4_replicate_fits.csv / _profiles.csv       every noisy fit and its whole profile
    stage4_replicate_results.csv   final numbers per fit (recomputed from profiles)
    stage4_summary.csv        rejection rates (nominal and calibrated), alpha-hat, R0 ratio
    stage4_calibration.csv    real size of the test from 200 SEIR data sets per noise level
    stage4_sensitivity.csv / _profiles.csv          robustness of alpha*
    stage4_validation.csv     checks of the early stop and interval interpolation
    stage4_gain_by_phase.csv  where the fractional fit gains (before/after the peak)
    stage4_substeps_gamma*.csv     solver settings for the wrong-gamma fits
  figures/                    data_overview, ebola_cleaning, growth_shape,
                              memory_kernel, identifiability, solver_convergence,
                              matched_epidemics, decline_tails, pseudotrue_profiles,
                              replicate_alpha, test_calibration, example_fits (.png, .pdf)
  paper/                      main.tex + figures/ (not on GitHub until submission)
  requirements.txt
```

## Running it (Windows PowerShell, from the project folder)

First time only:

```
py -3.13 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Every time:

```
.venv\Scripts\activate
python src\01_download_data.py
python src\02_clean_data.py
python src\03_plot_data.py
python src\04_check_data.py
python src\05_check_theory.py
python src\06_plot_theory.py
python src\07_check_solvers.py
python src\08_plot_solvers.py
python src\09_check_fitting.py
python src\10_synthetic_experiments.py
python src\11_plot_synthetic.py
cd paper
pdflatex main.tex
pdflatex main.tex
cd ..
```

Expected last lines: `04_check_data.py` `All 12 checks passed.`,
`05_check_theory.py` `All 21 checks passed.` (about a minute),
`07_check_solvers.py` `All 21 checks passed.` (about a minute),
`09_check_fitting.py` `All 19 checks passed.` (about two minutes).

`10_synthetic_experiments.py` is the long one: about 1,300 profile-likelihood
fits of 10-20 seconds each, run in parallel (roughly 5 hours on 2 cores, 2.5 on
4, 1.5 on 8). Its results are included in `results/`, so run as above it finds
every fit done and only rebuilds the summary tables (about a minute). To redo
everything from scratch: `python src\10_synthetic_experiments.py --fresh`. If
interrupted, run it again: finished fits are kept.
`pdflatex` is run twice so cross-references resolve; the paper is `paper\main.pdf`.
It needs a local LaTeX installation (MiKTeX or TeX Live); Overleaf works too.

## Data

| Series | Source (commit) | Window | Weeks | Cases | Dated by |
|---|---|---|---|---|---|
| Mpox, United States | OWID `owid/monkeypox` (08f97c1), WHO confirmed | 9 May - 19 Dec 2022 | 33 | 30,056 | report |
| Ebola, Guinea | `cmrivers/ebola` (ddcbf44) | 29 Mar 2014 - 3 Jan 2015 | 41 | 2,720 | report |
| Ebola, Liberia | same | 5 Apr 2014 - 3 Jan 2015 | 40 | 8,159 | report |
| Ebola, Sierra Leone | same | 5 Apr 2014 - 3 Jan 2015 | 40 | 9,720 | report |
| Measles, West Texas | JHU `CSSEGISandData/measles_data` (c361b38) | 19 Jan - 7 Sep 2025 | 34 | 802 | rash onset |
| Measles, Upstate SC | same (4 counties) | 7 Sep 2025 - 26 Apr 2026 | 34 | 990 | report |
| Measles, Short Creek UT-AZ | same (SW Utah HD + Mohave) | 22 Jun 2025 - 31 May 2026 | 50 | 550 | report |

### Data issues found and how they are handled

1. **Mpox counts are weekly, stored in a daily file.** `new_cases` is zero six
   days a week and holds the whole week on Mondays. The weekly series is the
   Monday-to-Monday change in `total_cases`. The wave ends at the first week
   after the peak below 2% of the peak (an objective rule).
2. **Ebola running totals fall** 7 / 4 / 3 times (Guinea / Liberia / Sierra
   Leone), from reclassified suspected cases. Fixed by **isotonic regression**:
   the closest non-decreasing curve (least squares). No report is deleted;
   final totals are unchanged; past 100 cases no value moves more than 7.3%.
3. **Ebola "Day" column** is one day off its own date on 8 rows; days are
   recomputed from dates.
4. **Ebola dates are irregular.** The cleaned running total is read every
   7 days by linear interpolation, then differenced. These weekly counts are
   interpolated, so Stage 5 must fit them with care.
5. **Flat starts.** Liberia and Sierra Leone have weeks of near-zero counts
   before the real epidemic. A take-off rule (first week after the last run of
   3+ weeks with < 1 case, before the peak) gives 14 Jun 2014 (Liberia) and
   24 May 2014 (Sierra Leone). Fitting from week 0 would fake "slow growth".
6. **Reporting batches.** Weeks above 3 x the median of their 4 neighbours
   (floor 5 cases) are flagged: 7 weeks. Clear batches: Liberia 25 Oct 2014
   (running total 4,665 -> 6,535 in a week), and the 28 Dec 2025 holiday
   backlog in South Carolina (174) and Short Creek (79). Flags are kept as a
   column, not deleted.

## Theory (Stage 2): main results

All proved in Section 3 of the paper and checked numerically by `05_check_theory.py`.

| Result | Statement |
|---|---|
| Theorem 3 | Caputo SEIR has a unique solution for all time that stays non-negative and bounded. The proof avoids the invalid argument "non-negative Caputo derivative implies non-decreasing" (Diethelm 2016). |
| Proposition 4 | R0 = beta^a sigma^a / ((sigma^a + mu^a)(gamma^a + mu^a)); for an outbreak (mu = 0) R0 = (beta/gamma)^a. Endemic equilibrium exists iff R0 > 1. |
| Theorem 5 | Disease-free state stable iff R0 < 1; endemic state stable whenever it exists; for every alpha (Matignon + Cong et al. 2016, 2017). |
| Proposition 6 | Early growth is E_a(lambda1 t^a): onset like t^a, then exponential at r = lambda1^(1/a) plus a t^(-a) memory term. |
| Propositions 7-8 | AHM model is well posed; with no competing removal and alpha < 1 the mean infectious period is infinite and there is NO epidemic threshold. |
| Proposition 10 | The growth rate r does not identify alpha: every alpha fits r with its own beta, and the implied R0 = (1 + (r/sigma)^a)(1 + (r/gamma)^a) changes with alpha (about 30% higher at alpha = 0.5 when r = gamma/2, sigma = gamma). |

## Numerics (Stage 3): main findings

| Finding | Detail |
|---|---|
| Two independent Caputo solvers | Predictor-corrector (order about 1+alpha) and implicit L1 (order 1), as predicted for non-smooth fractional solutions; they agree on the nonlinear model. |
| Automatic steps | PECE is stable only for h * lambda^(1/alpha) below about 1.6-2.0; the power 1/alpha makes small alpha much stiffer, so daily steps went unstable at alpha = 0.3. L1 loses positivity unless h^(-alpha)/Gamma(2-alpha) > lambda1 (weekly steps, large beta). The AHM solver needs h omega S0 / 2 small and about 80 steps per tau. All three now sub-step automatically and return values on the requested grid. |
| Same growth, different epidemics | Caputo SEIR epidemics tuned to the same early growth rate differ hugely after the peak (alpha = 0.6 still has 2e4 infectious at week 120). |
| Tails | Numerically, fractional declines approach power laws t^(-alpha) (by t ~ 6000 weeks), but the Caputo SEIR passes through a long transitional phase (slope from -4 to -0.3), so a tail slope from data does NOT estimate alpha. Integer models decline exponentially. |
| Independent review | Confirmed weights, L1 discretisation and conservation; found the two positivity problems above (now fixed). |
| For Stage 5 | Bound alpha from below (cost grows like lambda^(1/alpha)); fix the number of sub-steps per fit (avoids small jumps in the likelihood); use alpha <= 0.9999 or exactly 1 in the AHM model. |

## Synthetic experiments (Stage 4): main findings

Data from 8 known mechanisms, fitted with the Caputo SEIR (alpha free).
alpha* = where the fit converges with unlimited data (Poisson counts).

| Truth | Memory? | alpha* | Rejects alpha = 1 (k = 10 / 100, calibrated) |
|---|---|---|---|
| T0 ordinary SEIR | no | 1.000 | 3% / 3% |
| T1 Erlang stages | no | 0.986 | 17% / 3% |
| T2 declining transmission | **no** | **0.763** | 60% / 97% |
| T3 reporting delay | no | 0.983 | 13% / 27% |
| T4 Caputo alpha = 0.8 | yes | 0.800 | 37% / 97% |
| T5 Caputo alpha = 0.6 | yes | 0.600 | 100% / 100% |
| T6 AHM alpha = 0.7 | yes | 0.772 | 53% / 80% |
| T7 heterogeneous recovery | **no** | **0.810** | 40% / 73% |

- A fitted alpha < 1 is NOT evidence of memory: declining transmission and
  heterogeneous recovery (no memory) are reported as memory as often as real memory.
- Proposition 11: the AHM model's memory IS heterogeneity (its Mittag-Leffler
  survival is a mixture of exponentials).
- Remark 4: in the Caputo model cumulative infections can fall (S rises after the peak).
- With noisy counts (k = 10) the nominal 5% test rejects alpha = 1 in 12% of
  ordinary-SEIR data sets: tests must be calibrated (Stage 5: parametric bootstrap).
- The fractional fit reports an R0 18-30% larger than the ordinary fit.

## GitHub

The repository tracks code, processed data, results and figures. `paper/` and
`data/raw/` are excluded in `.gitignore`.
