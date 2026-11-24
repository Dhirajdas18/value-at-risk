"""Build data/prices.parquet from Yahoo Finance (no setup needed)."""

import sys
import time
from pathlib import Path

from var.data import (
    DATA_PATH,
    INDEX,
    SYMBOLS,
    fetch_yfinance,
    flag_adjusted_drops,
    missing_symbols,
)


def download(fetch, find_missing, attempts: int = 4):
    """Call `fetch` until `find_missing` reports nothing absent; exit rather than write a gap.

    Yahoo occasionally drops a ticker or errors, so each download gets a few tries."""
    missing = ["everything"]
    for attempt in range(attempts):
        if attempt:
            time.sleep(5 * attempt)
        try:
            df = fetch()
        except Exception as exc:  # network or Yahoo error: retry
            print(f"  download attempt {attempt + 1} failed: {exc}")
            continue
        if not (missing := find_missing(df)):
            return df
    sys.exit(f"download failed for {missing}; not writing an incomplete table")


def main() -> None:
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else DATA_PATH
    df = download(fetch_yfinance, missing_symbols)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out)
    span = f"{df.index.min():%Y-%m-%d} to {df.index.max():%Y-%m-%d}"
    print(f"Yahoo Finance: {len(df)} rows, {span} -> {out}")
    for sym, date in flag_adjusted_drops(df["adj_close"][SYMBOLS + [INDEX]]):
        print(f"  check {sym} {date:%Y-%m-%d}: adjusted price fell >20% in a day")


if __name__ == "__main__":
    main()
