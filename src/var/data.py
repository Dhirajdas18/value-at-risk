"""Daily OHLC + adjusted closes from Yahoo Finance for today's NIFTY 50 and the index.

The constituent list (docs/data/nifty50.csv) is a snapshot of NSE's published list; refresh it with
scripts/fetch_constituents.py. Using today's members over past years is survivorship-biased: the
point-in-time walk-forward studies use var.universe instead.


Columns are a (field, symbol) MultiIndex. Fields: open, high, low, close, adj_close.
India VIX only has a meaningful close.
"""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
CONSTITUENTS = ROOT / "docs" / "data" / "nifty50.csv"
SYMBOLS = pd.read_csv(CONSTITUENTS)["symbol"].tolist()
SECTORS = dict(pd.read_csv(CONSTITUENTS).set_index("symbol")["industry"])
INDEX = "NIFTY"
VIX = "INDIAVIX"
ETFS = ["NIFTYBEES", "GOLDBEES"]  # NIFTY 50 with dividends; gold in rupees
START = "2007-01-01"  # includes 2008
DATA_PATH = ROOT / "data" / "prices.parquet"
FIELDS = {"Open": "open", "High": "high", "Low": "low", "Close": "close", "Adj Close": "adj_close"}


def fetch_yfinance(start: str = START, end: str | None = None) -> pd.DataFrame:
    """Download every symbol. Yahoo's open/high/low/close are split-adjusted, not dividend-adjusted;
    that is fine for intraday ranges (ratios within one day). Returns use adj_close."""
    import yfinance as yf

    tickers = ({f"{s}.NS": s for s in SYMBOLS} | {"^NSEI": INDEX, "^INDIAVIX": VIX}
               | {f"{e}.NS": e for e in ETFS})
    # threads=False: parallel downloads can lock yfinance's sqlite cache and drop a ticker
    raw = yf.download(list(tickers), start=start, end=end, auto_adjust=False, progress=False,
                      threads=False)
    raw = raw[list(FIELDS)].rename(columns=FIELDS, level=0).rename(columns=tickers, level=1)
    raw.columns.names = ["field", "symbol"]
    raw.index = pd.DatetimeIndex(raw.index.tz_localize(None), name="date")
    return raw.sort_index(axis=1)


def missing_symbols(table: pd.DataFrame, expected: list[str] | None = None) -> list[str]:
    """Symbols with no adjusted close at all: an all-empty column or one Yahoo left out.

    `expected` defaults to every symbol we download (the stocks, NIFTY and India VIX).
    """
    adj = table["adj_close"] if "adj_close" in table.columns.get_level_values(0) else pd.DataFrame()
    want = expected if expected is not None else SYMBOLS + [INDEX, VIX] + ETFS
    return [s for s in want if s not in adj.columns or adj[s].isna().all()]


def load_table(symbols: list[str] | None = None, path: Path = DATA_PATH) -> pd.DataFrame:
    """The whole (field, symbol) table, limited to `symbols` (default: the stocks + NIFTY)."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `uv run python scripts/fetch_data.py` first."
        )
    keep = symbols or [INDEX] + SYMBOLS
    table = pd.read_parquet(path)
    table = table.loc[:, table.columns.get_level_values("symbol").isin(keep)]
    return table.reindex(columns=keep, level="symbol").sort_index()


def load(field: str, symbols: list[str] | None = None, path: Path = DATA_PATH) -> pd.DataFrame:
    """One field as a wide date x symbol frame.

    Keeps only rows where every requested symbol (all, if None) has data.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `uv run python scripts/fetch_data.py` first."
        )
    table = pd.read_parquet(path)
    if field not in table.columns.get_level_values("field"):
        raise KeyError(f"field not in price table: {field}")
    df = table[field]
    if symbols is not None:
        missing = [s for s in symbols if s not in df.columns]
        if missing:
            raise KeyError(f"symbols not in price table: {missing}")
        df = df[symbols]
    df.columns.name = None
    return df.sort_index().dropna(how="any")


def flag_adjusted_drops(
    adj: pd.DataFrame, threshold: float = 0.2
) -> list[tuple[str, pd.Timestamp]]:
    """(symbol, date) pairs where an adjusted price falls more than `threshold` in a day.

    Large caps rarely do that for real; each hit is worth checking for an unadjusted split or bonus.
    """
    moves = adj.pct_change().stack()
    hits = moves[moves < -threshold]
    return [(sym, date) for date, sym in hits.index]


def load_bars(symbol: str, path: Path = DATA_PATH) -> pd.DataFrame:
    """Dividend- and split-adjusted open and close for one symbol.

    Yahoo adjusts only the close; the open is scaled by the same day's factor (adj_close / close).
    """
    fields = {f: load(f, [symbol], path=path)[symbol] for f in ("open", "close", "adj_close")}
    df = pd.concat(fields, axis=1).dropna()
    factor = df["adj_close"] / df["close"]
    return pd.DataFrame({"open": df["open"] * factor, "close": df["adj_close"]})


FIXES = ROOT / "docs" / "data" / "yahoo_fixes.csv"


def daily_returns(prices: pd.DataFrame, fixes: Path = FIXES) -> pd.DataFrame:
    """Simple daily returns of adjusted closes, with known Yahoo errors repaired from
    docs/data/yahoo_fixes.csv (each with its source):
    - zero: a corporate action Yahoo did not adjust for; that day's return is set to 0.
    - blank: a bad print; the price is dropped and the previous price carried forward.
    - start: the series is not this company before the date; earlier prices are dropped.
    """
    px = prices.copy()
    table = pd.read_csv(fixes, parse_dates=["date"])
    if "action" not in table:
        table["action"] = "zero"
    for row in table[table["action"] == "start"].itertuples():
        if row.symbol in px.columns:
            px.loc[px.index < row.date, row.symbol] = float("nan")
    for row in table[table["action"] == "blank"].itertuples():
        if row.symbol in px.columns and row.date in px.index:
            px.loc[row.date, row.symbol] = float("nan")
    started = px.notna().cummax()  # forward-fill only inside a stock's own history
    px = px.ffill().where(started)
    r = px.pct_change(fill_method=None)
    for row in table[table["action"] == "zero"].itertuples():
        if row.symbol in r.columns and row.date in r.index:
            r.loc[row.date, row.symbol] = 0.0
    return r
