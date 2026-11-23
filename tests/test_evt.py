import numpy as np
import pytest
from scipy.stats import genpareto

from var.evt import evt_es, evt_var, exceed_probability, fit_tail


def _pareto_losses(xi=0.3, n=200_000, seed=0):
    rng = np.random.default_rng(seed)
    # body: small losses; tail: exactly GPD above u = 0.02 for 5% of days
    body = rng.uniform(-0.02, 0.02, int(n * 0.95))
    tail = 0.02 + genpareto.rvs(xi, scale=0.01, size=n - len(body), random_state=rng)
    return -np.concatenate([body, tail])  # returns: losses are negative


def test_fit_recovers_the_tail_shape_and_scale():
    t = fit_tail(_pareto_losses(), threshold_q=0.95)
    assert t["xi"] == pytest.approx(0.3, abs=0.03)
    assert t["beta"] == pytest.approx(0.01, rel=0.05)
    assert t["share"] == pytest.approx(0.05, rel=0.01)


def test_evt_var_and_es_match_the_true_quantiles():
    t = fit_tail(_pareto_losses(), threshold_q=0.95)
    q = 0.999  # tail probability 0.001 = 0.05 × 0.02, so GPD quantile at 0.98
    true_var = 0.02 + genpareto.ppf(0.98, 0.3, scale=0.01)
    assert evt_var(t, q) == pytest.approx(true_var, rel=0.05)
    true_es = (true_var + 0.01 - 0.3 * 0.02) / (1 - 0.3)
    assert evt_es(t, q) == pytest.approx(true_es, rel=0.07)
    assert evt_es(t, q) > evt_var(t, q)


def test_exceed_probability_inverts_var():
    t = fit_tail(_pareto_losses(), threshold_q=0.95)
    assert exceed_probability(t, evt_var(t, 0.999)) == pytest.approx(0.001, rel=1e-6)
