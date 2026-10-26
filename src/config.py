"""
config.py -- settings shared by every script in this project.

Nothing runs here. Other scripts do `import config` and read the values below.
Keeping all paths, web addresses and date windows in ONE place means that if
something changes (for example a new data version), we edit it here only.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# 1. Folder layout
# ---------------------------------------------------------------------------
# __file__ is the path of this file (src/config.py).
# .resolve().parent is the src folder; one more .parent is the project folder.
PROJECT_DIR = Path(__file__).resolve().parent.parent

RAW_DIR = PROJECT_DIR / "data" / "raw"              # downloaded files (not on GitHub)
PROCESSED_DIR = PROJECT_DIR / "data" / "processed"  # cleaned files we analyse
FIGURES_DIR = PROJECT_DIR / "figures"               # figures for GitHub
RESULTS_DIR = PROJECT_DIR / "results"               # summary tables
PAPER_FIG_DIR = PROJECT_DIR / "paper" / "figures"   # copy of figures for Overleaf

# ---------------------------------------------------------------------------
# 2. Data sources, each PINNED to one fixed GitHub commit
# ---------------------------------------------------------------------------
# A "commit" is a frozen snapshot of a repository. Downloading from a commit
# (instead of from "main") means the file can never change under our feet,
# so every number in the paper can be reproduced exactly, years from now.
#
# sha256 is a "fingerprint" of each file: if even one byte differs,
# the fingerprint changes. 04_check_data.py compares the two.

SOURCES = {
    "mpox": {
        "file": "owid-monkeypox-data.csv",
        "url": "https://raw.githubusercontent.com/owid/monkeypox/"
               "08f97c122304a99496ff127ca0be17929084fe3c/owid-monkeypox-data.csv",
        "repo": "github.com/owid/monkeypox",
        "commit": "08f97c1",
        "sha256": "2764761fd455f9fb295101129b10e37cf1f4e7acf1ae3c779b6f6d2b25b2c927",
    },
    "ebola": {
        "file": "ebola_country_timeseries.csv",
        "url": "https://raw.githubusercontent.com/cmrivers/ebola/"
               "ddcbf44475a84b324ad8ca0c075784d070e70d37/country_timeseries.csv",
        "repo": "github.com/cmrivers/ebola",
        "commit": "ddcbf44",
        "sha256": "fda0a9972ec3bc827c7ccdae688acb85dc7c36581efee261c4bb439605d96be4",
    },
    "measles_county": {
        "file": "measles_county_all_updates.csv",
        "url": "https://raw.githubusercontent.com/CSSEGISandData/measles_data/"
               "c361b387631ce7e31f9d9b28eb36b7642b8b9892/measles_county_all_updates.csv",
        "repo": "github.com/CSSEGISandData/measles_data",
        "commit": "c361b38",
        "sha256": "4f98e2daaefa0cfb996e2c833df27be6670a288f8cdb386ed816b5d314800971",
    },
    "measles_states": {
        "file": "measles_Top_states_time_series.csv",
        "url": "https://raw.githubusercontent.com/CSSEGISandData/measles_data/"
               "c361b387631ce7e31f9d9b28eb36b7642b8b9892/Top_states_time_series.csv",
        "repo": "github.com/CSSEGISandData/measles_data",
        "commit": "c361b38",
        "sha256": "96b60e410f0c033ae170bc6df9dfba8384324d14ee26138c0fb84a7b19035fb1",
    },
}

# ---------------------------------------------------------------------------
# 3. Choices about each dataset
# ---------------------------------------------------------------------------

# --- Mpox, United States, 2022 wave ---
MPOX_LOCATION = "United States"
# The wave starts at the first reported week. It ends at the first week AFTER
# the peak whose count falls below this fraction of the peak (an objective rule,
# so we do not hand-pick the end date).
MPOX_END_FRACTION_OF_PEAK = 0.02

# --- Ebola 2014, West Africa ---
EBOLA_COUNTRIES = {"Guinea": "Cases_Guinea",
                   "Liberia": "Cases_Liberia",
                   "Sierra Leone": "Cases_SierraLeone"}
# In this file "Day 0" is 22 March 2014 (the first WHO report).
EBOLA_DAY0 = "2014-03-22"

# Take-off rule (used for every series): a model should be fitted from the week
# the epidemic took off, not from a flat early stretch. Take-off = the first
# week after the LAST run of at least TAKEOFF_QUIET_WEEKS weeks with fewer than
# TAKEOFF_QUIET_CASES cases, looking only before the peak. No such run -> week 0.
TAKEOFF_QUIET_WEEKS = 3
# Reporting-spike flag: a week is flagged when its count is more than
# SPIKE_FACTOR times the median of the 4 weeks around it (2 before, 2 after).
# Flagged weeks are kept; later stages decide how to treat them.
SPIKE_FACTOR = 3.0
SPIKE_MIN_TYPICAL = 5.0   # floor for the neighbours' median (avoids small-number noise)
TAKEOFF_QUIET_CASES = 1.0

# --- Measles, three US outbreaks (same windows as the measles paper) ---
# date_basis tells us whether a case is dated by rash onset or by report date.
MEASLES_OUTBREAKS = {
    "West Texas": {
        "source": "states", "column": "TX_cases",
        "start": "2025-01-19", "end": "2025-09-07",
        "date_basis": "rash onset",
        "expected_total": 802, "expected_weeks": 34,
    },
    "Upstate South Carolina": {
        "source": "county",
        "counties": ["Spartanburg, South Carolina", "Cherokee, South Carolina",
                     "Greenville, South Carolina", "Pickens, South Carolina"],
        "start": "2025-09-07", "end": "2026-04-26",
        "date_basis": "report",
        "expected_total": 990, "expected_weeks": 34,
    },
    "Short Creek (UT-AZ)": {
        "source": "county",
        "counties": ["Southwest Health District, Utah", "Mohave, Arizona"],
        "start": "2025-06-22", "end": "2026-05-31",
        "date_basis": "report",
        "expected_total": 550, "expected_weeks": 50,
    },
}

# ---------------------------------------------------------------------------
# 4. Figure colours (validated colour-blind-safe palette, used in fixed order)
# ---------------------------------------------------------------------------
COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]   # blue, orange, aqua
MARKERS = ["o", "s", "^"]                    # shapes, so colour is never the only cue
TEXT_DARK = "#0b0b0b"
TEXT_SOFT = "#52514e"
GRID = "#e4e3df"
