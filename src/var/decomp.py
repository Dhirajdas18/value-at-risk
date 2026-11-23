"""Where a portfolio's risk comes from, and what history's worst episodes would do to it today.

Holdings are ₹ amounts by symbol; results are in ₹.
- component_es: each holding's share of historical expected shortfall (average loss on the
  portfolio's worst days). The shares add up exactly to the portfolio's ES.
- component_var_normal: the same split for normal-curve VaR (Euler allocation), with each holding's
  marginal VaR: the extra ₹ of VaR per extra ₹ held.
- incremental_es: the change in ES from a trade, by recomputing the whole portfolio.
- stress: replay a dated episode on today's holdings; a holding without prices then is proxied by
  its beta to the NIFTY index.
"""

import numpy as np
import pandas as pd
from scipy.stats import norm

from var.var import historical, portfolio_returns

INDEX = "NIFTY"
TRACKERS = {"NIFTYBEES": 1.0}  # an index fund maps 1:1 to its index, not by a noisy beta
# (last close before, last close of the episode): NIFTY's fall between those closes
SCENARIOS = {
    "2008 crash": ("2008-01-08", "2008-10-27"),  # peak to the October low
    "Demonetisation": ("2016-11-08", "2016-12-26"),  # announced after the close on 8 Nov
    "COVID crash": ("2020-02-19", "2020-03-23"),
    "2024 election result": ("2024-06-03", "2024-06-04"),  # one day
}


def _weights(holdings: dict) -> pd.Series:
    h = pd.Series(holdings, dtype=float)
    return h / h.sum()


def component_es(R: pd.DataFrame, holdings: dict, alpha: float = 0.975) -> pd.Series:
    """₹ of historical ES carried by each holding: its average loss on the portfolio's tail days."""
    R = R[list(holdings)].dropna()
    p = portfolio_returns(R, holdings)
    tail = p <= np.quantile(p, 1 - alpha)
    w = _weights(holdings)
    return -(R[tail].mean() * w) * sum(holdings.values())


def component_var_normal(R: pd.DataFrame, holdings: dict, alpha: float = 0.99) -> pd.DataFrame:
    """Normal VaR (zero mean) split by holding: marginal = z·(Σw)ᵢ/σ, component = hᵢ·marginal."""
    R = R[list(holdings)].dropna()
    w = _weights(holdings)
    sw = R.cov().to_numpy() @ w.to_numpy()
    marginal = norm.ppf(alpha) * sw / float(np.sqrt(w.to_numpy() @ sw))
    h = pd.Series(holdings, dtype=float)
    return pd.DataFrame({"marginal": marginal, "component": h * marginal}, index=w.index)


def incremental_es(R: pd.DataFrame, holdings: dict, symbol: str, amount: float,
                   alpha: float = 0.975) -> float:
    """₹ change in historical ES from buying `amount` more of `symbol` (full recomputation)."""
    after = dict(holdings)
    after[symbol] = after.get(symbol, 0.0) + amount
    keep = list(after)
    rows = R[keep].dropna()

    def es(h):
        return historical(portfolio_returns(rows, h), alpha)[1] * sum(h.values())

    return es(after) - es({k: v for k, v in holdings.items()})


def proxy_beta(asset: pd.Series, index: pd.Series) -> float:
    """Beta to the index from weekly returns (daily ones understate it for thinly traded days)."""
    pair = pd.concat([asset, index], axis=1).dropna()
    weekly = (1 + pair).cumprod().resample("W-FRI").last().pct_change().dropna()
    c = weekly.cov().to_numpy()
    return float(c[0, 1] / c[1, 1])


def stress(R: pd.DataFrame, holdings: dict, start, end, index: str = INDEX) -> pd.DataFrame:
    """Each holding's return and ₹ P&L over the closes after `start` up to `end`."""
    window = R.loc[(R.index > pd.Timestamp(start)) & (R.index <= pd.Timestamp(end))]
    idx_ret = float((1 + window[index]).prod() - 1)
    rows = {}
    for s, amount in holdings.items():
        x = window[s]
        if len(x) and x.notna().all():
            ret, source = float((1 + x).prod() - 1), "history"
        else:
            if s in TRACKERS:
                b, source = TRACKERS[s], "proxy: NIFTY index"
            else:
                b = proxy_beta(R[s], R[index])
                source = f"proxy: β {b:.2f} × NIFTY"
            ret = b * idx_ret
        rows[s] = {"return": ret, "pnl": ret * amount, "source": source}
    return pd.DataFrame(rows).T.astype({"return": float, "pnl": float})
