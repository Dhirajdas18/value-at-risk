"""Backtesting VaR: did the forecast loss get exceeded as often as promised, and at random?

- rolling_var: each day's VaR forecast from the previous `window` days only.
- kupiec: proportion-of-failures test (Kupiec 1995): breach rate vs 1 - alpha. LR ~ chi2(1).
- christoffersen: independence test (Christoffersen 1998): is a breach more likely right after a
  breach? LR ~ chi2(1).
- traffic_light: Basel zones from breaches of 99% VaR in the last 250 days.
"""

import numpy as np
import pandas as pd
from scipy.stats import chi2, kurtosis, t

from var.var import historical, normal


def rolling_var(r: pd.Series, method: str = "historical", window: int = 250,
                alpha: float = 0.99) -> pd.Series:
    """VaR forecast for each day from the `window` days before it (losses positive)."""
    x = r.dropna().to_numpy()
    out = []
    for i in range(window, len(x)):
        past = x[i - window:i]
        if method == "historical":
            out.append(historical(past, alpha)[0])
        elif method == "normal":
            out.append(normal(past.mean(), past.std(ddof=1), alpha)[0])
        elif method == "student-t":
            k = kurtosis(past)
            df = max(4 + 6 / k, 2.5) if k > 0 else 100.0
            scale = past.std(ddof=1) * np.sqrt((df - 2) / df)
            out.append(-(past.mean() + scale * t.ppf(1 - alpha, df)))
        else:
            raise ValueError(method)
    return pd.Series(out, index=r.dropna().index[window:])


def kupiec(breaches: int, days: int, q: float) -> tuple[float, float]:
    """LR statistic and p-value that the true breach probability is q."""
    pi = breaches / days

    def ll(p):
        return ((days - breaches) * np.log1p(-p) if breaches < days else 0.0) + (
            breaches * np.log(p) if breaches > 0 else 0.0)

    lr = float(-2 * (ll(q) - ll(pi)))
    return max(lr, 0.0), float(chi2.sf(max(lr, 0.0), 1))


def christoffersen(hits) -> tuple[float, float]:
    """LR statistic and p-value that breaches are independent from one day to the next."""
    h = np.asarray(hits, dtype=bool)
    prev, now = h[:-1], h[1:]
    n00, n01 = int((~prev & ~now).sum()), int((~prev & now).sum())
    n10, n11 = int((prev & ~now).sum()), int((prev & now).sum())
    p01 = n01 / max(n00 + n01, 1)
    p11 = n11 / max(n10 + n11, 1)
    p = (n01 + n11) / max(n00 + n01 + n10 + n11, 1)

    def term(n, prob):
        return n * np.log(prob) if n > 0 else 0.0

    l_ind = term(n00, 1 - p01) + term(n01, p01) + term(n10, 1 - p11) + term(n11, p11)
    l_null = term(n00 + n10, 1 - p) + term(n01 + n11, p)
    lr = max(float(-2 * (l_null - l_ind)), 0.0)
    return lr, float(chi2.sf(lr, 1))


def traffic_light(breaches_250: int) -> str:
    """Basel zone for the number of 99% VaR breaches in 250 days."""
    return "green" if breaches_250 <= 4 else "yellow" if breaches_250 <= 9 else "red"
