"""Value at Risk and expected shortfall. Losses are positive fractions of the portfolio's value.

VaR at confidence alpha: the loss exceeded on only (1 - alpha) of days. Expected shortfall (ES):
the average loss on those worst days.
- historical: quantiles of past portfolio returns.
- normal: from the mean and standard deviation, assuming a bell curve.
- cornish_fisher: the normal quantile corrected for skew and fat tails (excess kurtosis).
- monte_carlo: simulate correlated asset returns (normal or Student-t), then read the quantile.
"""

import numpy as np
import pandas as pd
from scipy.stats import norm


def historical(r, alpha: float = 0.99) -> tuple[float, float]:
    r = np.asarray(r, dtype=float)
    r = r[np.isfinite(r)]
    q = float(np.quantile(r, 1 - alpha))
    return -q, float(-r[r <= q].mean())


def normal(mu: float, sigma: float, alpha: float = 0.99) -> tuple[float, float]:
    z = norm.ppf(1 - alpha)
    return float(-(mu + sigma * z)), float(-mu + sigma * norm.pdf(z) / (1 - alpha))


def _cf_z(z, skew, k):
    return (z + (z**2 - 1) * skew / 6 + (z**3 - 3 * z) * k / 24
            - (2 * z**3 - 5 * z) * skew**2 / 36)


def cornish_fisher(mu: float, sigma: float, skew: float, excess_kurt: float,
                   alpha: float = 0.99) -> tuple[float, float]:
    """Cornish-Fisher VaR; ES by averaging the corrected quantile over the tail."""
    var = -(mu + sigma * _cf_z(norm.ppf(1 - alpha), skew, excess_kurt))
    u = (np.arange(2000) + 0.5) / 2000 * (1 - alpha)
    es = -(mu + sigma * _cf_z(norm.ppf(u), skew, excess_kurt)).mean()
    return float(var), float(es)


def monte_carlo(mu, cov, weights, alpha: float = 0.99, n: int = 100_000, seed=None,
                df: float | None = None) -> tuple[float, float]:
    """Simulate `n` days of asset returns (multivariate normal, or Student-t with `df` degrees of
    freedom scaled to the same covariance) and take the portfolio's historical VaR and ES."""
    rng = np.random.default_rng(seed)
    mu, cov, w = (np.asarray(x, dtype=float) for x in (mu, cov, weights))
    z = rng.multivariate_normal(np.zeros(len(mu)), cov, n)
    if df is not None:
        z *= np.sqrt((df - 2) / rng.chisquare(df, n))[:, None]
    return historical((mu + z) @ w, alpha)


def portfolio_returns(returns: pd.DataFrame, holdings: dict) -> pd.Series:
    """Daily returns of today's holdings (₹ by symbol) under each past day's asset returns."""
    w = pd.Series(holdings, dtype=float)
    w = w / w.sum()
    return returns[w.index] @ w


def horizon_returns(r: pd.Series, days: int) -> pd.Series:
    """Compound returns over every overlapping window of `days` trading days."""
    growth = (1 + r).cumprod()
    out = growth / growth.shift(days) - 1
    out.iloc[days - 1] = growth.iloc[days - 1] - 1
    return out.dropna()


def cornish_fisher_valid(skew: float, excess_kurt: float) -> bool:
    """Cornish-Fisher is only a valid quantile function where it keeps rising with z. With fat
    enough tails it bends back on itself and its VaR is meaningless (Maillard 2012)."""
    z = np.linspace(-4, 4, 801)
    return bool(np.all(np.diff(_cf_z(z, skew, excess_kurt)) > 0))
