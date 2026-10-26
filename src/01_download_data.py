"""
01_download_data.py -- download the four public data files into data/raw/.

What it does, step by step
  1. Makes the folder data/raw/ if it does not exist yet.
  2. For each source listed in config.SOURCES:
       a. downloads the file from its pinned GitHub commit,
       b. computes the file's SHA-256 fingerprint,
       c. compares it with the fingerprint stored in config.py.
  3. Writes results/data_sources.csv: where each file came from and whether
     the fingerprint matched. (data/raw/ is not on GitHub, but this table is.)

Run it from the project folder:
    python src/01_download_data.py
If a file is already downloaded and its fingerprint is correct, it is skipped.
"""

import hashlib
import sys
import urllib.request
from datetime import date

import pandas as pd

import config


def sha256_of_file(path):
    """Return the SHA-256 fingerprint of a file as a 64-character text string."""
    h = hashlib.sha256()
    with open(path, "rb") as f:                 # "rb" = read the raw bytes
        for block in iter(lambda: f.read(1 << 20), b""):   # 1 MB at a time
            h.update(block)
    return h.hexdigest()


def download(url, path):
    """Download `url` and save it to `path`."""
    # A browser-like "User-Agent" header avoids rare refusals from servers.
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request, timeout=120) as response:
        data = response.read()
    with open(path, "wb") as f:                 # "wb" = write raw bytes
        f.write(data)


def main():
    config.RAW_DIR.mkdir(parents=True, exist_ok=True)
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    rows = []          # one row per file, for the data_sources.csv table
    all_ok = True

    for name, src in config.SOURCES.items():
        path = config.RAW_DIR / src["file"]

        # Step 2a: download, unless a correct copy is already there
        if path.exists() and sha256_of_file(path) == src["sha256"]:
            print(f"[skip] {src['file']} already downloaded and verified")
        else:
            print(f"[download] {src['file']} ...")
            download(src["url"], path)

        # Steps 2b and 2c: fingerprint check
        fingerprint = sha256_of_file(path)
        matches = fingerprint == src["sha256"]
        all_ok = all_ok and matches
        size_kb = path.stat().st_size / 1024
        print(f"    size {size_kb:,.0f} KB, fingerprint "
              f"{'MATCHES' if matches else 'DOES NOT MATCH'} the pinned version")

        rows.append({
            "dataset": name,
            "file": src["file"],
            "repository": src["repo"],
            "commit": src["commit"],
            "url": src["url"],
            "sha256": fingerprint,
            "fingerprint_ok": matches,
            "downloaded_on": date.today().isoformat(),
        })

    # Step 3: save the provenance table
    out = config.RESULTS_DIR / "data_sources.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\nSaved {out.relative_to(config.PROJECT_DIR)}")

    if not all_ok:
        print("WARNING: at least one file differs from the pinned version. "
              "Delete data/raw/ and run this script again.")
        sys.exit(1)
    print("All four files downloaded and verified.")


if __name__ == "__main__":
    main()
