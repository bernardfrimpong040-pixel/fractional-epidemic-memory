"""
outbreaks.py -- the seven real outbreak series and the disease parameters
used when fitting them (Stage 5).

Nothing runs here; 12_check_real_fitting.py, 13_fit_outbreaks.py and
14_plot_outbreaks.py import these definitions.

TIME UNIT: weeks (the data are weekly).

LATENT AND INFECTIOUS PERIODS (held fixed in every fit)
    A weekly case series cannot tell the latent period, the infectious period
    and alpha apart (Section 3), so sigma and gamma are taken from the
    literature. In the SEIR model the mean generation interval (time from
    being infected to infecting someone) is 1/sigma + 1/gamma. When a study
    reports the incubation period and the serial interval, we set
        1/sigma = mean incubation period,
        1/gamma = mean serial interval - mean incubation period,
    so that the model's mean generation interval equals the observed mean
    serial interval (which controls how fast an outbreak grows).

    mpox (US 2022): incubation 5.6 days, serial interval 8.5 days
        (Madewell et al., Emerg Infect Dis 29(4), 2023)
        -> 1/sigma = 5.6 d, 1/gamma = 2.9 d
    Ebola (West Africa 2014): incubation 11.4 days, serial interval 15.3 days
        (WHO Ebola Response Team, N Engl J Med 371, 2014)
        -> 1/sigma = 11.4 d, 1/gamma = 3.9 d
    measles: people are infectious from 4 days before to 4 days after rash
        onset, and the rash appears about 14 days after exposure (CDC), so
        1/sigma = 14 - 4 = 10 d and 1/gamma = 8 d (as in the earlier
        measles paper that used these three outbreaks).

    The sensitivity analysis multiplies gamma by 0.8 and 1.2.
"""

import numpy as np
import pandas as pd

import config

DAYS = 7.0   # days per week

DISEASES = {
    "mpox":    {"latent_days": 5.6,  "infectious_days": 2.9,
                "source": "Madewell et al. (2023)"},
    "ebola":   {"latent_days": 11.4, "infectious_days": 3.9,
                "source": "WHO Ebola Response Team (2014)"},
    "measles": {"latent_days": 10.0, "infectious_days": 8.0,
                "source": "CDC; earlier measles paper"},
}

# Short names used in file names and tables, in the order used in the paper.
SERIES = [
    ("mpox", "United States", "mpox-US"),
    ("ebola", "Guinea", "ebola-GN"),
    ("ebola", "Liberia", "ebola-LR"),
    ("ebola", "Sierra Leone", "ebola-SL"),
    ("measles", "West Texas", "measles-TX"),
    ("measles", "Upstate South Carolina", "measles-SC"),
    ("measles", "Short Creek (UT-AZ)", "measles-SCk"),
]
SHORT_NAMES = [s[2] for s in SERIES]


def rates(disease, gamma_factor=1.0):
    """(sigma, gamma) per WEEK for a disease; gamma_factor scales gamma."""
    d = DISEASES[disease]
    return DAYS / d["latent_days"], gamma_factor * DAYS / d["infectious_days"]


def load_series(short, mask_flagged=False):
    """Weekly counts of one series, from its take-off week to its last week.

    Returns a dict with:
        y      counts as floats (rounded to whole numbers: the Ebola weekly
               counts were interpolated from irregular reports, so they are
               not integers; the likelihood needs counts)
        dates  week-start dates
        flag   True for the weeks flagged as reporting spikes in Stage 1
        weeks  week numbers counted from take-off (0, 1, 2, ...)
    With mask_flagged=True the flagged weeks are set to NaN, which the
    fitting code skips (the week still exists in the model, it simply does
    not enter the likelihood).
    """
    disease, name, _ = next(s for s in SERIES if s[2] == short)
    d = pd.read_csv(config.PROCESSED_DIR / "all_series_weekly.csv")
    d = d[(d.dataset == disease) & (d.series == name)].sort_values("week")
    summ = pd.read_csv(config.RESULTS_DIR / "stage1_data_summary.csv")
    take = int(summ[(summ.dataset == disease) & (summ.series == name)].takeoff_week.iloc[0])
    d = d[d.week >= take]
    y = np.round(d.cases.to_numpy(dtype=float))
    flag = d.spike_flag.to_numpy(dtype=bool)
    if mask_flagged:
        y = np.where(flag, np.nan, y)
    return {"disease": disease, "name": name, "short": short, "y": y,
            "dates": d.date.to_numpy(), "flag": flag,
            "weeks": np.arange(len(y)), "takeoff_week": take}


def truncate_after_peak(y, weeks_after=4, flag=None):
    """Keep the data up to `weeks_after` weeks after the (first) peak week.

    flag: weeks flagged as reporting spikes. They are ignored when looking
    for the peak (in three series the highest week is a spike, which is a
    reporting artefact, not the epidemic peak), but they are kept in the data.
    """
    look = np.asarray(y, dtype=float).copy()
    if flag is not None:
        look[np.asarray(flag, dtype=bool)] = np.nan
    k = int(np.nanargmax(look))
    return y[: min(len(y), k + weeks_after + 1)]
