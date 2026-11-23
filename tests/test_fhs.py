import numpy as np
import pandas as pd
import pytest

from var.fhs import ewma_vol, fhs_var, garch_vol


def test_ewma_vol_uses_only_past_returns():
    r = pd.Series([0.01, -0.02, 0.03, 0.0, 0.01])
    v = ewma_vol(r, lam=0.94, seed_days=2)
    var2 = r.iloc[:2].pow(2).mean()  # seed from the first two days
    var3 = 0.94 * var2 + 0.06 * r.iloc[2] ** 2
    assert v.iloc[2] == pytest.approx(np.sqrt(var2))  # forecast for day 2 known before day 2
    assert v.iloc[3] == pytest.approx(np.sqrt(var3))
    assert v.iloc[:2].isna().all()


def test_fhs_recovers_the_shock_quantile_times_tomorrows_vol():
    rng = np.random.default_rng(0)
    n = 3000
    sigma = np.where(np.arange(n) < 1500, 0.01, 0.03)  # a regime change
    z = rng.standard_t(5, n) / np.sqrt(5 / 3)
    r = pd.Series(sigma * z)
    out = fhs_var(r, pd.Series(sigma), window=1000, alpha=0.99)
    true_q = -np.quantile(z, 0.01)
    assert out.iloc[-1] == pytest.approx(0.03 * true_q, rel=0.15)


def test_fhs_reacts_at_once_to_a_volatility_jump():
    rng = np.random.default_rng(1)
    r = pd.Series(np.concatenate([rng.normal(0, 0.01, 1200), rng.normal(0, 0.04, 30)]))
    vol = ewma_vol(r)
    fhs = fhs_var(r, vol, window=1000)
    hist = r.rolling(1000).quantile(0.01).shift(1).abs()
    assert fhs.iloc[-1] > 2.5 * fhs.iloc[1150] and hist.iloc[-1] < 1.5 * hist.iloc[1150]


def test_garch_vol_is_positive_and_forecasts_ahead():
    rng = np.random.default_rng(2)
    r = pd.Series(rng.normal(0, 0.01, 1300), index=pd.bdate_range("2015-01-01", periods=1300))
    v = garch_vol(r, window=1000, refit_every=100)
    assert v.dropna().gt(0).all() and v.index.equals(r.index) and v.iloc[:1000].isna().all()


def test_fhs_today_gives_tomorrows_var_and_es():
    from var.fhs import fhs_today

    rng = np.random.default_rng(4)
    r = pd.Series(rng.normal(0, 0.01, 800))
    v, es = fhs_today(r, window=500, alpha=0.99)
    vol = ewma_vol(r)
    lam_next = np.sqrt(0.94 * vol.iloc[-1] ** 2 + 0.06 * r.iloc[-1] ** 2)
    z = (r / vol).iloc[-500:]
    assert v == pytest.approx(-lam_next * np.quantile(z, 0.01))
    assert es >= v
