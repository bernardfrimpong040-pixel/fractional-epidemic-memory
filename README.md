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
| 2 | Models and theory: Caputo SEIR (rates^alpha), Angstmann-Henry-McGann fractional-recovery SIR, integer-order alternatives; existence, uniqueness, positivity, R0, Matignon stability, identifiability proposition | next |
| 3 | Numerical solver: Garrappa predictor-corrector / L1 scheme, Mittag-Leffler convergence test | |
| 4 | Synthetic tests: does memoryless data give a spurious alpha < 1? | |
| 5 | Fitting to the seven series, profile likelihood for alpha | |
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
  data/
    raw/                      downloads (not on GitHub; re-created by script 01)
    processed/
      mpox_us_weekly.csv
      ebola_reports.csv       raw and cleaned cumulative counts per report
      ebola_weekly.csv
      measles_weekly.csv
      all_series_weekly.csv   all 7 series in one table (used by later stages)
  results/
    data_sources.csv          where each file came from, commit, fingerprint
    stage1_data_summary.csv   one line per series (Table 2 of the paper)
  figures/                    data_overview, ebola_cleaning, growth_shape (.png, .pdf)
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
```

The last line printed should be `All 12 checks passed.` Everything runs in
under a minute.

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

## GitHub

The repository tracks code, processed data, results and figures. `paper/` and
`data/raw/` are excluded in `.gitignore`.
