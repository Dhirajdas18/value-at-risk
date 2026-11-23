import numpy as np
import pandas as pd
import pytest

from var.es import acerbi_szekely, es_forecasts, z2_pvalue
from var.var import normal


def test_normal_es_at_97_5_is_close_to_var_at_99():
    es975 = normal(0.0, 1.0, 0.975)[1]
    var99 = normal(0.0, 1.0, 0.99)[0]
    assert es975 == pytest.approx(2.3378, abs=1e-3) and var99 == pytest.approx(2.3263, abs=1e-3)


def test_z2_is_near_zero_when_es_is_right_and_negative_when_too_small():
    rng = np.random.default_rng(0)
    x = rng.normal(0, 0.01, 20000)
    v, es = normal(0.0, 0.01, 0.975)
    right = acerbi_szekely(x, np.full(len(x), v), np.full(len(x), es), 0.975)
    low = acerbi_szekely(x, np.full(len(x), v * 0.8), np.full(len(x), es * 0.8), 0.975)
    assert abs(right) < 0.1 and low < -0.4


def test_z2_pvalue_by_simulating_from_the_forecasts():
    rng = np.random.default_rng(1)
    n = 3000
    x = rng.standard_t(3, n) * 0.013 / np.sqrt(3)  # fatter tails and 30% more volatile
    v, es = normal(0.0, 0.01, 0.975)
    samples = lambda k: rng.normal(0, 0.01, (k, n))  # noqa: E731  model believes normal
    vv, ee = np.full(n, v), np.full(n, es)
    z, p = z2_pvalue(x, vv, ee, 0.975, samples, sims=500)
    assert z == pytest.approx(acerbi_szekely(x, vv, ee, 0.975)) and p < 0.05
    good = rng.normal(0, 0.01, n)
    assert z2_pvalue(good, vv, ee, 0.975, samples, sims=500)[1] > 0.05


def test_es_forecasts_use_only_past_days():
    r = pd.Series(np.random.default_rng(2).normal(0, 0.01, 800),
                  index=pd.bdate_range("2020-01-01", periods=800))
    a = es_forecasts(r, "historical", 250, 0.975)
    later = r.copy()
    later.iloc[600:] *= 4
    b = es_forecasts(later, "historical", 250, 0.975)
    pd.testing.assert_frame_equal(a.loc[: r.index[600]], b.loc[: r.index[600]])
    assert (a["es"] >= a["var"]).all()
