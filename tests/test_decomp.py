import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from var.decomp import (
    SCENARIOS,
    component_es,
    component_var_normal,
    incremental_es,
    proxy_beta,
    stress,
)
from var.var import historical, portfolio_returns


def _returns(n=4000, seed=0):
    rng = np.random.default_rng(seed)
    cov = np.array([[4, 2, 0], [2, 9, 0], [0, 0, 1]]) * 1e-4
    x = rng.multivariate_normal(np.zeros(3), cov, n)
    return pd.DataFrame(x, columns=["A", "B", "G"], index=pd.bdate_range("2010-01-01", periods=n))


def test_component_es_adds_up_to_historical_es():
    R, h = _returns(), {"A": 400_000, "B": 300_000, "G": 300_000}
    c = component_es(R, h, 0.975)
    es = historical(portfolio_returns(R, h), 0.975)[1] * 1_000_000
    assert c.sum() == pytest.approx(es, rel=1e-9)
    assert c["B"] > c["A"] > c["G"]  # riskiest holding carries most; uncorrelated G least


def test_component_var_normal_is_euler_split_of_normal_var():
    R, h = _returns(), {"A": 400_000, "B": 300_000, "G": 300_000}
    c = component_var_normal(R, h, 0.99)
    w = np.array([0.4, 0.3, 0.3])
    sd = float(np.sqrt(w @ R.cov().to_numpy() @ w))
    assert c["component"].sum() == pytest.approx(norm.ppf(0.99) * sd * 1_000_000, rel=1e-9)
    # marginal: ₹ of VaR per extra ₹ held; matches a small finite difference
    bumped = dict(h, G=h["G"] + 1000)
    wb = np.array([bumped[k] for k in "ABG"], dtype=float)
    var_b = norm.ppf(0.99) * float(np.sqrt(wb @ R.cov().to_numpy() @ wb))
    assert (var_b - c["component"].sum()) / 1000 == pytest.approx(c.loc["G", "marginal"], rel=0.01)


def test_incremental_es_is_full_revaluation():
    R, h = _returns(), {"A": 700_000, "G": 300_000}
    inc = incremental_es(R, h, "G", 200_000, 0.975)
    before = historical(portfolio_returns(R, h), 0.975)[1] * 1_000_000
    after = historical(portfolio_returns(R, {"A": 700_000, "G": 500_000}), 0.975)[1] * 1_200_000
    assert inc == pytest.approx(after - before, rel=1e-9)


def test_stress_compounds_each_holding_over_the_window():
    idx = pd.bdate_range("2020-02-17", periods=6)
    R = pd.DataFrame({"A": [0.0, -0.1, -0.1, 0.0, 0.0, 0.0], "NIFTY": [0, -0.05, -0.05, 0, 0, 0]},
                     index=idx, dtype=float)
    out = stress(R, {"A": 100_000}, idx[0], idx[2])
    assert out.loc["A", "return"] == pytest.approx(0.9 * 0.9 - 1)
    assert out.loc["A", "pnl"] == pytest.approx(-19_000)
    assert out.loc["A", "source"] == "history"


def test_stress_proxies_a_missing_holding_with_its_index_beta():
    R = _returns(2000)
    R["NIFTY"] = R["A"]
    R["X"] = 0.5 * R["A"] + 0.001 * np.random.default_rng(1).standard_normal(len(R))
    start, end = R.index[100], R.index[110]
    R.loc[: R.index[300], "X"] = np.nan  # X didn't trade yet
    b = proxy_beta(R["X"], R["NIFTY"])
    assert b == pytest.approx(0.5, abs=0.03)
    out = stress(R, {"X": 100_000}, start, end)
    nifty = float((1 + R.loc[(R.index > start) & (R.index <= end), "NIFTY"]).prod() - 1)
    assert out.loc["X", "source"].startswith("proxy")
    assert out.loc["X", "return"] == pytest.approx(b * nifty)
    R["NIFTYBEES"] = np.nan  # an index fund maps 1:1 to its index
    etf = stress(R, {"NIFTYBEES": 100_000}, start, end)
    assert etf.loc["NIFTYBEES", "return"] == pytest.approx(nifty)


def test_scenarios_are_dated_windows():
    assert {"2008 crash", "Demonetisation", "COVID crash", "2024 election result"} <= set(SCENARIOS)
    for start, end in SCENARIOS.values():
        assert pd.Timestamp(start) < pd.Timestamp(end)
