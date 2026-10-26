"""
02_clean_data.py -- turn the raw downloads into clean, tidy weekly tables.

Inputs  (data/raw/, made by 01_download_data.py)
Outputs (data/processed/ and results/)
  mpox_us_weekly.csv       US mpox 2022 wave, one row per week
  ebola_reports.csv        Ebola cumulative counts on each report date, raw and cleaned
  ebola_weekly.csv         Ebola counts on a regular weekly grid, per country
  measles_weekly.csv       the three US measles outbreaks, one row per week
  all_series_weekly.csv    all 7 series stacked in one long table (used by later stages)
  results/stage1_data_summary.csv   one summary line per series (Table 1 in the paper)

Every step is checked with an `assert` (a line that stops the script with an
error if a condition is false), so bad data cannot pass silently.
"""

import numpy as np
import pandas as pd
from scipy.optimize import isotonic_regression

import config


# ===========================================================================
# 1. MPOX, United States, 2022
# ===========================================================================
def clean_mpox():
    raw = pd.read_csv(config.RAW_DIR / config.SOURCES["mpox"]["file"])
    us = raw[raw["location"] == config.MPOX_LOCATION].copy()
    us["date"] = pd.to_datetime(us["date"])
    us = us.sort_values("date")

    # Why not use the column `new_cases` directly?
    # OWID's US file is DAILY, but WHO sends US counts once a week. So
    # `new_cases` is 0 on six days and holds the whole week's count on Monday.
    # The safest weekly series is the change in the running total
    # (`total_cases`) from one Monday to the next.
    assert us["total_cases"].is_monotonic_increasing, "running total went down"

    mondays = us[us["date"].dt.dayofweek == 0][["date", "total_cases"]]
    mondays = mondays.reset_index(drop=True)
    weekly = pd.DataFrame({
        "date": mondays["date"],
        # first week: its count is the running total itself (counting starts here)
        "cases": mondays["total_cases"].diff().fillna(mondays["total_cases"]),
    })
    weekly["cases"] = weekly["cases"].astype(int)

    # The weekly series must add back up to the running total
    assert weekly["cases"].sum() == int(mondays["total_cases"].iloc[-1])

    # Choose the 2022 wave window with an objective rule (see config.py):
    # end = first week after the peak below 2% of the peak.
    peak_idx = weekly["cases"].idxmax()
    peak = weekly.loc[peak_idx, "cases"]
    after = weekly.loc[peak_idx:]
    below = after[after["cases"] < config.MPOX_END_FRACTION_OF_PEAK * peak]
    end_idx = below.index[0]
    wave = weekly.loc[:end_idx].copy()

    wave["week"] = np.arange(len(wave))              # 0, 1, 2, ...
    wave["cumulative"] = wave["cases"].cumsum()
    print(f"[mpox] {len(wave)} weeks, {wave['date'].min().date()} to "
          f"{wave['date'].max().date()}, {wave['cases'].sum():,} cases, "
          f"peak {peak:,} in week of {weekly.loc[peak_idx, 'date'].date()}")
    return wave[["week", "date", "cases", "cumulative"]]


# ===========================================================================
# 2. EBOLA 2014: Guinea, Liberia, Sierra Leone
# ===========================================================================
def clean_ebola():
    raw = pd.read_csv(config.RAW_DIR / config.SOURCES["ebola"]["file"])
    raw["Date"] = pd.to_datetime(raw["Date"], format="%m/%d/%Y")
    raw = raw.sort_values("Date")
    day0 = pd.Timestamp(config.EBOLA_DAY0)

    # The file has its own "Day" column, but on 8 rows it is one day off from
    # the date written next to it. Dates are the reliable field, so we
    # recompute the day number from the date and report how many differed.
    day_from_date = (raw["Date"] - day0).dt.days
    n_off = int((day_from_date != raw["Day"]).sum())
    print(f"[ebola] 'Day' column disagreed with the date on {n_off} rows; "
          f"day numbers recomputed from dates")
    assert n_off == 8, "the raw file is not the pinned version"
    raw["Day"] = day_from_date

    reports, weekly_parts = [], []

    for country, column in config.EBOLA_COUNTRIES.items():
        # Keep only dates on which this country actually reported (drop blanks)
        s = raw[["Date", "Day", column]].dropna().copy()
        s = s.rename(columns={"Date": "date", "Day": "day", column: "cum_raw"})

        # ---- The problem: the running total sometimes goes DOWN ----
        # A cumulative count can never fall in reality. Falls happen when
        # suspected cases are later ruled out ("reclassified").
        # ---- The fix: isotonic regression ----
        # It finds the non-decreasing curve that is as close as possible
        # (least squares) to the reported numbers. Where the data already
        # rise, nothing changes; around a drop, the neighbouring values are
        # pulled together to their average so the curve stays flat instead
        # of falling. No report is deleted.
        fit = isotonic_regression(s["cum_raw"].to_numpy(), increasing=True)
        s["cum_clean"] = np.round(fit.x, 1)
        s["adjusted"] = (s["cum_clean"] - s["cum_raw"]).abs() > 0.05
        s["country"] = country

        n_drops = int((s["cum_raw"].diff() < 0).sum())
        assert np.all(np.diff(s["cum_clean"]) >= 0), "cleaned series still falls"
        print(f"[ebola] {country}: {len(s)} reports, {n_drops} drops fixed, "
              f"{int(s['adjusted'].sum())} values adjusted, "
              f"final total {s['cum_raw'].iloc[-1]:,.0f}")
        reports.append(s)

        # ---- Put the irregular reports on a regular weekly grid ----
        # Reports arrive every 1-7 days. Later stages want equal steps, so we
        # read the cleaned running total every 7 days by drawing straight
        # lines between neighbouring reports (linear interpolation), then
        # take differences to get cases per week.
        first, last = s["day"].iloc[0], s["day"].iloc[-1]
        grid = np.arange(int(np.ceil(first / 7) * 7), last + 1, 7)
        cum_on_grid = np.interp(grid, s["day"], s["cum_clean"])
        w = pd.DataFrame({
            "country": country,
            "day": grid,
            "date": day0 + pd.to_timedelta(grid, unit="D"),
            "cumulative": np.round(cum_on_grid, 1),
        })
        w["cases"] = w["cumulative"].diff().round(1)  # first row has no week before it
        w = w.dropna(subset=["cases"]).reset_index(drop=True)
        w["week"] = np.arange(len(w))
        weekly_parts.append(w)

    reports = pd.concat(reports, ignore_index=True)
    weekly = pd.concat(weekly_parts, ignore_index=True)
    reports = reports[["country", "date", "day", "cum_raw", "cum_clean", "adjusted"]]
    weekly = weekly[["country", "week", "day", "date", "cases", "cumulative"]]
    return reports, weekly


# ===========================================================================
# 3. MEASLES: three US outbreaks (integer-order benchmark)
# ===========================================================================
def clean_measles():
    county = pd.read_csv(config.RAW_DIR / config.SOURCES["measles_county"]["file"])
    county["date"] = pd.to_datetime(county["date"])
    states = pd.read_csv(config.RAW_DIR / config.SOURCES["measles_states"]["file"])
    states["week_start"] = pd.to_datetime(states["week_start"])

    # In the county file each row is NEW cases announced on that date (not a
    # running total), so a week's count is simply the sum over that week.
    # Weeks run Sunday to Saturday (CDC "MMWR" weeks). The line below moves
    # every date back to the Sunday that starts its week:
    #   dayofweek is 0 for Monday ... 6 for Sunday, so (dayofweek+1) % 7 is the
    #   number of days since the last Sunday.
    county["week_start"] = county["date"] - pd.to_timedelta(
        (county["date"].dt.dayofweek + 1) % 7, unit="D")

    parts = []
    for name, spec in config.MEASLES_OUTBREAKS.items():
        weeks = pd.date_range(spec["start"], spec["end"], freq="7D")
        if spec["source"] == "states":
            s = states.set_index("week_start")[spec["column"]]
        else:
            rows = county[county["location_name"].isin(spec["counties"])]
            s = rows.groupby("week_start")["value"].sum()
        s = s.reindex(weeks, fill_value=0)              # weeks with no reports = 0

        df = pd.DataFrame({"outbreak": name, "week": np.arange(len(weeks)),
                           "date": weeks, "cases": s.to_numpy().astype(int)})
        df["cumulative"] = df["cases"].cumsum()
        df["date_basis"] = spec["date_basis"]

        # Must reproduce the series used in the measles paper exactly
        assert len(df) == spec["expected_weeks"], name
        assert df["cases"].sum() == spec["expected_total"], name
        print(f"[measles] {name}: {len(df)} weeks, {df['cases'].sum()} cases "
              f"(by {spec['date_basis']} date) -- matches the measles paper")
        parts.append(df)
    return pd.concat(parts, ignore_index=True)


# ===========================================================================
# 4. One long table with all seven series, and a summary table
# ===========================================================================
def stack_all(mpox, ebola_weekly, measles):
    a = mpox.assign(dataset="mpox", series="United States")
    b = ebola_weekly.rename(columns={"country": "series"}).assign(dataset="ebola")
    c = measles.rename(columns={"outbreak": "series"}).assign(dataset="measles")
    cols = ["dataset", "series", "week", "date", "cases", "cumulative"]
    out = pd.concat([a[cols], b[cols], c[cols]], ignore_index=True)

    # Flag reporting spikes (rule in config.py), series by series
    out["spike_flag"] = False
    for _, g in out.groupby(["dataset", "series"], sort=False):
        out.loc[g.index, "spike_flag"] = flag_spikes(g["cases"].to_numpy())
    flagged = out[out["spike_flag"]]
    for _, r in flagged.iterrows():
        print(f"[spike] {r['series']}: week of {r['date'].date()} has "
              f"{r['cases']:,.0f} cases -- flagged for review (possible reporting batch)")
    return out


def flag_spikes(cases):
    """True for weeks whose count is > SPIKE_FACTOR x median of the 4 neighbours.

    The neighbours are the 2 weeks before and the 2 weeks after (fewer at the
    ends of the series). The median is floored at SPIKE_MIN_TYPICAL (5 cases)
    so that small-number noise, such as a jump from 1 to 10 cases, is not
    called a spike.
    """
    cases = np.asarray(cases, dtype=float)
    flags = np.zeros(len(cases), dtype=bool)
    for i in range(len(cases)):
        neighbours = np.r_[cases[max(0, i - 2):i], cases[i + 1:i + 3]]
        typical = max(np.median(neighbours), config.SPIKE_MIN_TYPICAL)
        flags[i] = cases[i] > config.SPIKE_FACTOR * typical
    return flags


def takeoff_week(cases):
    """Return the week the epidemic took off (rule explained in config.py).

    cases: weekly counts in time order (a list or array).
    Example: Liberia's counts start 11, 0, 0, 0, 0, 0, 0, 0, 0, 0, 10, 20, ...
    The quiet run is weeks 1-9 (nine weeks under 1 case), so take-off = week 10.
    """
    cases = np.asarray(cases, dtype=float)
    peak = int(np.argmax(cases))
    quiet = cases[:peak] < config.TAKEOFF_QUIET_CASES   # True/False for each week before the peak
    start, run = 0, 0
    for i, q in enumerate(quiet):
        run = run + 1 if q else 0          # length of the current quiet run
        if run >= config.TAKEOFF_QUIET_WEEKS:
            start = i + 1                   # first week after this (long enough) quiet run
    # If the run continues, move `start` past its end
    while start < peak and cases[start] < config.TAKEOFF_QUIET_CASES:
        start += 1
    return start


def summarise(all_series):
    notes = {
        "mpox": "WHO confirmed cases, weekly (via OWID)",
        "ebola": "suspected+probable+confirmed, interpolated to weeks",
        "measles": "lab-confirmed (JHU)",
    }
    rows = []
    for (dataset, series), g in all_series.groupby(["dataset", "series"], sort=False):
        g = g.reset_index(drop=True)
        k = g["cases"].idxmax()
        t0 = takeoff_week(g["cases"])
        rows.append({
            "dataset": dataset,
            "series": series,
            "first_week": g["date"].min().date(),
            "last_week": g["date"].max().date(),
            "n_weeks": len(g),
            "total_cases": round(g["cases"].sum()),
            "peak_weekly_cases": round(g.loc[k, "cases"]),
            "peak_week_number": int(g.loc[k, "week"]),
            "peak_is_spike": bool(g.loc[k, "spike_flag"]),
            "n_spike_weeks": int(g["spike_flag"].sum()),
            "peak_is_last_week": k == len(g) - 1,
            "takeoff_week": t0,
            "takeoff_date": g.loc[t0, "date"].date(),
            "weeks_takeoff_to_peak": int(g.loc[k, "week"]) - t0,
            "note": notes[dataset],
        })
    return pd.DataFrame(rows)


def main():
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    mpox = clean_mpox()
    ebola_reports, ebola_weekly = clean_ebola()
    measles = clean_measles()
    all_series = stack_all(mpox, ebola_weekly, measles)
    summary = summarise(all_series)

    out = config.PROCESSED_DIR
    mpox.to_csv(out / "mpox_us_weekly.csv", index=False)
    ebola_reports.to_csv(out / "ebola_reports.csv", index=False)
    ebola_weekly.to_csv(out / "ebola_weekly.csv", index=False)
    measles.to_csv(out / "measles_weekly.csv", index=False)
    all_series.to_csv(out / "all_series_weekly.csv", index=False)
    summary.to_csv(config.RESULTS_DIR / "stage1_data_summary.csv", index=False)

    print("\nSummary (results/stage1_data_summary.csv):")
    print(summary.drop(columns="note").to_string(index=False))
    print("\nSaved 5 files in data/processed/ and the summary in results/.")


if __name__ == "__main__":
    main()
