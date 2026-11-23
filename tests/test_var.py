import numpy as np
import pandas as pd
import pytest

from var.var import (
    cornish_fisher,
    historical,
    horizon_returns,
    monte_carlo,
    normal,
    portfolio_returns,
)

RNG = np.random.default_rng(0)


def test_normal_var_and_es_formulas():
    v, es = normal(mu=0.0005, sigma=0.01, alpha=0.99)
    assert v == pytest.approx(2.32635 * 0.01 - 0.0005, rel=1e-4)
    assert es == pytest.approx(0.01 * 2.66521 - 0.0005, rel=1e-4)  # phi(z) / (1 - alpha)


def test_historical_var_on_a_known_sample():
    r = np.arange(-50, 50) / 1000  # -5.0% ... 4.9%, 100 days
    v, es = historical(r, alpha=0.95)
    assert v == pytest.approx(0.04505, abs=1e-4)  # 5th percentile, interpolated
    assert es == pytest.approx(np.mean(np.arange(-50, -45) / -1000), abs=1e-9)


def test_cornish_fisher_equals_normal_without_skew_or_fat_tails():
    v_cf, es_cf = cornish_fisher(0.0, 0.01, skew=0.0, excess_kurt=0.0, alpha=0.99)
    v_n, es_n = normal(0.0, 0.01, 0.99)
    assert v_cf == pytest.approx(v_n) and es_cf == pytest.approx(es_n, rel=1e-3)


def test_cornish_fisher_raises_var_for_fat_left_tails():
    v_cf, _ = cornish_fisher(0.0, 0.01, skew=-0.5, excess_kurt=5.0, alpha=0.99)
    assert v_cf > normal(0.0, 0.01, 0.99)[0]


def test_monte_carlo_normal_matches_parametric():
    cov = np.array([[1.0, 0.3], [0.3, 2.0]]) * 1e-4
    w = np.array([0.6, 0.4])
    v, es = monte_carlo(np.zeros(2), cov, w, alpha=0.99, n=400_000, seed=1)
    sigma = np.sqrt(w @ cov @ w)
    assert v == pytest.approx(normal(0.0, sigma, 0.99)[0], rel=0.02)
    assert es >= v


def test_es_is_at_least_var_for_every_method():
    r = RNG.standard_t(4, 2000) * 0.01
    for v, es in (historical(r, 0.99), normal(r.mean(), r.std(), 0.99),
                  cornish_fisher(r.mean(), r.std(), -0.3, 3.0, 0.99)):
        assert es >= v > 0


def test_portfolio_returns_use_todays_weights():
    rets = pd.DataFrame({"A": [0.01, -0.02], "B": [0.03, 0.0]})
    p = portfolio_returns(rets, {"A": 300_000, "B": 100_000})
    assert list(p) == pytest.approx([0.75 * 0.01 + 0.25 * 0.03, 0.75 * -0.02])


def test_ten_day_returns_compound_overlapping_windows():
    r = pd.Series([0.01] * 12)
    h = horizon_returns(r, 10)
    assert len(h) == 3 and h.iloc[0] == pytest.approx(1.01**10 - 1)


def test_cornish_fisher_validity_check():
    from var.var import cornish_fisher_valid

    assert cornish_fisher_valid(0.0, 0.0) and cornish_fisher_valid(-0.3, 3.0)
    assert not cornish_fisher_valid(0.13, 17.7)  # NIFTY's daily returns 2009-2026
