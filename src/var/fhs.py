"""Filtered historical simulation (Barone-Adesi et al. 1999; Hull & White 1998).

Divide each past return by the volatility forecast for its day to get standardised shocks, then
scale their historical quantile by the volatility forecast for tomorrow. The shape of the tail
comes from history; its size follows today's volatility.

Volatility forecasts (each uses only earlier returns; the value on day t is the forecast for t):
- ewma_vol: RiskMetrics exponentially weighted volatility (lambda 0.94).
- garch_vol: GARCH(1,1), refitted every `refit_every` days on the previous `window` days and run
  forward day by day in between.
"""

import numpy as np
import pandas as pd


def ewma_vol(r: pd.Series, lam: float = 0.94, seed_days: int = 30) -> pd.Series:
    x = r.to_numpy(dtype=float)
    out = np.full(len(x), np.nan)
    var = float(np.mean(x[:seed_days] ** 2))
    for t in range(seed_days, len(x)):
        out[t] = np.sqrt(var)
        var = lam * var + (1 - lam) * x[t] ** 2
    return pd.Series(out, index=r.index)


def garch_vol(r: pd.Series, window: int = 1000, refit_every: int = 21) -> pd.Series:
    from arch import arch_model

    x = r.to_numpy(dtype=float)
    out = np.full(len(x), np.nan)
    for start in range(window, len(x), refit_every):
        past = x[start - window:start] * 100  # arch prefers percent returns
        res = arch_model(past, mean="Zero", vol="GARCH", p=1, q=1, rescale=False).fit(
            disp="off")
        omega, alpha, beta = (res.params[k] for k in ("omega", "alpha[1]", "beta[1]"))
        var = omega + alpha * past[-1] ** 2 + beta * res.conditional_volatility[-1] ** 2
        for t in range(start, min(start + refit_every, len(x))):
            out[t] = np.sqrt(var) / 100
            var = omega + alpha * (x[t] * 100) ** 2 + beta * var
    return pd.Series(out, index=r.index)


def fhs_var(r: pd.Series, vol: pd.Series, window: int = 1000, alpha: float = 0.99) -> pd.Series:
    """VaR for each day: its volatility forecast × the (1 - alpha) quantile of the standardised
    shocks of the previous `window` days (losses positive)."""
    z = (r / vol).to_numpy(dtype=float)
    v = vol.to_numpy(dtype=float)
    out = np.full(len(z), np.nan)
    for t in range(len(z)):
        past = z[max(0, t - window):t]
        past = past[np.isfinite(past)]
        if len(past) >= window and np.isfinite(v[t]):
            out[t] = -v[t] * np.quantile(past, 1 - alpha)
    return pd.Series(out, index=r.index)


def fhs_today(r: pd.Series, lam: float = 0.94, window: int = 500,
              alpha: float = 0.99) -> tuple[float, float]:
    """Tomorrow's FHS VaR and ES from EWMA volatility: the next volatility forecast times the
    quantile (and tail mean) of the last `window` standardised shocks."""
    vol = ewma_vol(r, lam)
    x = r.to_numpy(dtype=float)
    nxt = np.sqrt(lam * vol.iloc[-1] ** 2 + (1 - lam) * x[-1] ** 2)
    z = (r / vol).to_numpy(dtype=float)[-window:]
    z = z[np.isfinite(z)]
    q = np.quantile(z, 1 - alpha)
    return float(-nxt * q), float(-nxt * z[z <= q].mean())
