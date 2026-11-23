"""Extreme value theory, peaks over threshold (McNeil, Frey & Embrechts, Quantitative Risk
Management, ch. 5).

Losses beyond a high threshold u follow, approximately, a generalised Pareto distribution (GPD)
with shape xi (tail fatness: 0 = exponential like a normal's tail, > 0 = power law with
1/xi finite moments) and scale beta. With a share Nu/n of days beyond u:

    P(loss > x) = (Nu/n) (1 + xi (x - u) / beta)^(-1/xi)
    VaR_q = u + beta/xi [ ((n/Nu)(1 - q))^(-xi) - 1 ]
    ES_q  = (VaR_q + beta - xi u) / (1 - xi)          (xi < 1)
"""

import numpy as np
from scipy.stats import genpareto

SMALL = 1e-6  # |xi| below this uses the exponential (xi -> 0) formulas


def fit_tail(returns, threshold_q: float = 0.95) -> dict:
    """Fit a GPD to the losses beyond the `threshold_q` quantile of losses."""
    loss = -np.asarray(returns, dtype=float)
    loss = loss[np.isfinite(loss)]
    u = float(np.quantile(loss, threshold_q))
    excess = loss[loss > u] - u
    xi, _, beta = genpareto.fit(excess, floc=0)
    return {"xi": float(xi), "beta": float(beta), "u": u, "share": len(excess) / len(loss),
            "n_tail": len(excess)}


def evt_var(t: dict, q: float) -> float:
    xi, beta, u, share = t["xi"], t["beta"], t["u"], t["share"]
    if abs(xi) < SMALL:  # exponential limit
        return float(u - beta * np.log((1 - q) / share))
    return float(u + beta / xi * (((1 - q) / share) ** (-xi) - 1))


def evt_es(t: dict, q: float) -> float:
    xi = t["xi"]
    return float((evt_var(t, q) + t["beta"] - xi * t["u"]) / (1 - xi))


def exceed_probability(t: dict, loss: float) -> float:
    """Daily probability of a loss bigger than `loss` (a positive fraction, at or above u)."""
    xi, beta, u, share = t["xi"], t["beta"], t["u"], t["share"]
    if abs(xi) < SMALL:  # exponential limit
        return float(share * np.exp(-(loss - u) / beta))
    return float(share * (1 + xi * (loss - u) / beta) ** (-1 / xi))
