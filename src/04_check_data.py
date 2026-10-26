"""
04_check_data.py -- independent checks on everything Stage 1 produced.

Each check prints PASS or FAIL. Where possible a number is recomputed in a
DIFFERENT way from 02_clean_data.py, so a mistake in one method is caught
by the other. Run it after scripts 01-03; the last line should read
"All 12 checks passed."
"""

import hashlib
import sys

import numpy as np
import pandas as pd

import config

results = []   # list of (check name, True/False)


def check(name, condition, detail=""):
    """Record and print one check."""
    results.append((name, bool(condition)))
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {name}" + (f"  ({detail})" if detail else ""))


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ---- Load everything ----
P = config.PROCESSED_DIR
mpox = pd.read_csv(P / "mpox_us_weekly.csv", parse_dates=["date"])
ebola_r = pd.read_csv(P / "ebola_reports.csv", parse_dates=["date"])
ebola_w = pd.read_csv(P / "ebola_weekly.csv", parse_dates=["date"])
measles = pd.read_csv(P / "measles_weekly.csv", parse_dates=["date"])
alls = pd.read_csv(P / "all_series_weekly.csv", parse_dates=["date"])
summary = pd.read_csv(config.RESULTS_DIR / "stage1_data_summary.csv")

# 1. Raw files are exactly the pinned versions
ok = all(sha256(config.RAW_DIR / s["file"]) == s["sha256"] for s in config.SOURCES.values())
check("1. raw files match their pinned fingerprints", ok)

# 2. Seven series, no missing values
check("2. seven series present, no missing values",
      alls.groupby(["dataset", "series"]).ngroups == 7 and not alls[["cases", "cumulative"]].isna().any().any())

# 3. No negative weekly counts anywhere
check("3. no negative weekly counts", (alls["cases"] >= 0).all(),
      f"smallest = {alls['cases'].min()}")

# 4. Weeks are exactly 7 days apart in every series
gaps_ok = all((g["date"].diff().dropna() == pd.Timedelta(days=7)).all()
              for _, g in alls.groupby(["dataset", "series"]))
check("4. every series is on a regular 7-day grid", gaps_ok)

# 5. Running totals agree with the sum of weekly counts
cum_ok = all(np.isclose(g["cases"].sum(), g["cumulative"].iloc[-1] - (g["cumulative"].iloc[0] - g["cases"].iloc[0]))
             for _, g in alls.groupby(["dataset", "series"]))
check("5. weekly counts add up to the running totals", cum_ok)

# 6. Mpox, recomputed a different way: resample the OWID running total to
#    weeks ending on Monday with pandas, then difference.
raw = pd.read_csv(config.RAW_DIR / config.SOURCES["mpox"]["file"])
us = raw[raw["location"] == config.MPOX_LOCATION].copy()
us["date"] = pd.to_datetime(us["date"])
weekly2 = us.set_index("date")["total_cases"].resample("W-MON").last().diff()
weekly2.iloc[0] = us["total_cases"].iloc[0]
same = np.allclose(weekly2.loc[mpox["date"]].to_numpy(), mpox["cases"].to_numpy())
check("6. mpox weekly counts match an independent recomputation", same,
      f"{mpox['cases'].sum():,} cases in {len(mpox)} weeks")

# 7. Measles totals reproduce the measles paper
tot = measles.groupby("outbreak")["cases"].sum()
exp = {k: v["expected_total"] for k, v in config.MEASLES_OUTBREAKS.items()}
check("7. measles totals match the measles paper (802 / 990 / 550)",
      all(tot[k] == v for k, v in exp.items()), ", ".join(f"{k}: {tot[k]}" for k in exp))

# 8. Ebola cleaned curves never fall
mono = all((np.diff(g["cum_clean"]) >= 0).all() for _, g in ebola_r.groupby("country"))
check("8. Ebola cleaned running totals never decrease", mono)

# 9. Ebola cleaning is small: the largest change, relative to the reported value
ebola_r["rel_change"] = (ebola_r["cum_clean"] - ebola_r["cum_raw"]).abs() / ebola_r["cum_raw"].clip(lower=1)
big = ebola_r[ebola_r["cum_raw"] >= 100]["rel_change"].max()
check("9. once past 100 cases, no Ebola value moved by more than 10%", big <= 0.10,
      f"largest change = {100 * big:.1f}%")

# 10. Ebola final reported totals are unchanged by cleaning
last = ebola_r.groupby("country").tail(1)
check("10. Ebola final totals unchanged by cleaning",
      np.allclose(last["cum_raw"], last["cum_clean"]),
      ", ".join(f"{c}: {v:,.0f}" for c, v in zip(last["country"], last["cum_clean"])))

# 11. Ebola weekly grid reproduces the cleaned reports on the grid days
#     (linear interpolation is exact at a report day)
ok11 = True
for c, g in ebola_w.groupby("country"):
    r = ebola_r[ebola_r["country"] == c].set_index("day")["cum_clean"]
    shared = g[g["day"].isin(r.index)]
    ok11 &= np.allclose(shared["cumulative"], r.loc[shared["day"]].to_numpy(), atol=0.1)
check("11. Ebola weekly grid agrees with reports on shared days", ok11)

# 12. Take-off comes before the peak in every series
check("12. take-off week is before the peak in every series",
      (summary["takeoff_week"] < summary["peak_week_number"]).all())

n_pass = sum(ok for _, ok in results)
print()
if n_pass == len(results):
    print(f"All {len(results)} checks passed.")
else:
    print(f"{len(results) - n_pass} of {len(results)} checks FAILED.")
    sys.exit(1)
