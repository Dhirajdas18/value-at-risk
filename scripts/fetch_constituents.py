"""Refresh docs/data/nifty50.csv (symbol, industry) from NSE's published NIFTY 50 list."""

import csv
import io
import urllib.request

from var.data import CONSTITUENTS

URL = "https://nsearchives.nseindia.com/content/indices/ind_nifty50list.csv"


def main() -> None:
    req = urllib.request.Request(URL, headers={"User-Agent": "value-at-risk/0.1 (research)"})
    with urllib.request.urlopen(req, timeout=60) as r:
        rows = list(csv.DictReader(io.StringIO(r.read().decode("utf-8"))))
    if len(rows) != 50:
        raise SystemExit(f"expected 50 constituents, got {len(rows)}; not overwriting")
    with open(CONSTITUENTS, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["symbol", "industry"])
        w.writerows([r["Symbol"].strip(), r["Industry"].strip()] for r in rows)
    print(f"{len(rows)} constituents -> {CONSTITUENTS}")


if __name__ == "__main__":
    main()
