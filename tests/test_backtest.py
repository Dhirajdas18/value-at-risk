import numpy as np
import pandas as pd
import pytest
from scipy.stats import chi2

from var.backtest import christoffersen, kupiec, rolling_var, traffic_light


def test_kupiec_is_zero_when_breaches_match_exactly():
    lr, p = kupiec(10, 1000, 0.01)
    assert lr == pytest.approx(0.0, abs=1e-9) and p == pytest.approx(1.0)


def test_kupiec_matches_the_textbook_formula():
    x, n, q = 20, 1000, 0.01
    pi = x / n
    expected = -2 * ((n - x) * np.log(1 - q) + x * np.log(q)
                     - (n - x) * np.log(1 - pi) - x * np.log(pi))
    lr, p = kupiec(x, n, q)
    assert lr == pytest.approx(expected) and p == pytest.approx(chi2.sf(expected, 1))


def test_christoffersen_flags_clustered_breaches():
    spread = np.zeros(1000, dtype=bool)
    spread[::100] = True  # 10 isolated breaches
    clustered = np.zeros(1000, dtype=bool)
    clustered[500:510] = True  # the same 10, all in a row
    assert christoffersen(spread)[1] > 0.5
    assert christoffersen(clustered)[1] < 0.001


def test_traffic_light_zones():
    assert [traffic_light(k) for k in (0, 4, 5, 9, 10)] == ["green", "green", "yellow", "yellow",
                                                           "red"]


def test_rolling_var_uses_only_past_days():
    r = pd.Series(np.random.default_rng(0).normal(0, 0.01, 600),
                  index=pd.bdate_range("2020-01-01", periods=600))
    base = rolling_var(r, "historical", window=250)
    later = r.copy()
    later.iloc[400:] *= 5
    moved = rolling_var(later, "historical", window=250)
    assert base.index[0] == r.index[250]  # first forecast needs a full window
    pd.testing.assert_series_equal(base.loc[: r.index[400]], moved.loc[: r.index[400]])
