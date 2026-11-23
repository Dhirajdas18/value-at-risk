import numpy as np
import pandas as pd
import pytest

from var.data import SECTORS, SYMBOLS, daily_returns


def test_constituent_snapshot_has_fifty_stocks_with_sectors():
    assert len(SYMBOLS) == 50 and set(SECTORS) == set(SYMBOLS)


def _fixes(tmp_path, rows):
    p = tmp_path / "fixes.csv"
    pd.DataFrame(rows, columns=["symbol", "date", "action", "reason"]).to_csv(p, index=False)
    return p


def test_zero_blank_and_start_fixes(tmp_path):
    idx = pd.bdate_range("2020-01-01", periods=5)
    px = pd.DataFrame({"A": [100.0, 50.0, 51.0, 52.0, 53.0],  # day 1: unadjusted split -> zero
                       "B": [10.0, 1.0, 10.5, 10.6, 10.7],    # day 1: bad print -> blank
                       "C": [99.0, 98.0, 5.0, 5.1, 5.2]},     # real history starts on day 2
                      index=idx)
    fixes = _fixes(tmp_path, [["A", idx[1].date(), "zero", ""], ["B", idx[1].date(), "blank", ""],
                              ["C", idx[2].date(), "start", ""]])
    r = daily_returns(px, fixes)
    assert r.loc[idx[1], "A"] == 0.0
    assert r.loc[idx[1], "B"] == 0.0 and r.loc[idx[2], "B"] == pytest.approx(0.05)
    assert r.loc[: idx[2], "C"].isna().all() and r.loc[idx[3], "C"] == pytest.approx(0.02)
    assert np.isfinite(r.loc[idx[3]]).all()
