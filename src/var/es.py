"""Expected shortfall forecasts and their backtest.

- es_forecasts: each day's VaR and ES at `alpha` from past data only (historical, normal, or
  filtered historical with EWMA volatility), with the forecast distribution kept for simulation.
- acerbi_szekely: Z2 = sum(X_t · I_t / ((1 - alpha) · ES_t)) / T + 1, where X_t is the return and
  I_t marks a VaR breach (Acerbi & Szekely 2014). Zero for a correct ES; negative when ES is too
  small.
- z2_pvalue: the share of Z2 values simulated from the forecasts that are at or below the real one.
"""

import numpy as np
import pandas as pd

from var.fhs import ewma_vol


def es_forecasts(r: pd.Series, method: str = "historical", window: int = 250,
                 alpha: float = 0.975) -> pd.DataFrame:
    """Columns var, es (losses positive) for each day from the `window` days before it. The
    forecast distribution for each day is stored in `.attrs["draw"]`, a function of (rng, k)
    returning a (k, days) array of simulated returns."""
    x = r.dropna()
    v = x.to_numpy(dtype=float)
    vol = ewma_vol(x).to_numpy() if method == "filtered" else None
    rows, pools = [], []
    for t in range(window, len(v)):
        if method == "historical":
            pool = v[t - window:t]
        elif method == "filtered":
            z = (v[t - window:t] / vol[t - window:t])
            pool = vol[t] * z[np.isfinite(z)]
        elif method == "normal":
            past = v[t - window:t]
            pool = (past.mean(), past.std(ddof=1))
        else:
            raise ValueError(method)
        if method == "normal":
            from var.var import normal

            rows.append(normal(pool[0], pool[1], alpha))
        else:
            q = np.quantile(pool, 1 - alpha)
            rows.append((-q, -pool[pool <= q].mean()))
        pools.append(pool)
    out = pd.DataFrame(rows, columns=["var", "es"], index=x.index[window:])

    def draw(rng, k):
        if method == "normal":
            mu = np.array([p[0] for p in pools])
            sd = np.array([p[1] for p in pools])
            return mu + sd * rng.standard_normal((k, len(pools)))
        return np.stack([rng.choice(p, k) for p in pools], axis=1)

    out.attrs["draw"] = draw
    return out


def acerbi_szekely(x, var, es, alpha: float = 0.975) -> float:
    x, var, es = (np.asarray(a, dtype=float) for a in (x, var, es))
    hit = x < -var
    return float(np.sum(x * hit / ((1 - alpha) * es)) / len(x) + 1)


def z2_pvalue(x, var, es, alpha, draw, sims: int = 1000) -> tuple[float, float]:
    """Z2 and its one-sided p-value: how often returns drawn from the forecasts themselves give a
    Z2 at least as negative. `draw(k)` returns a (k, days) array of simulated returns."""
    z = acerbi_szekely(x, var, es, alpha)
    sim = draw(sims)
    hit = sim < -np.asarray(var)[None, :]
    zs = np.sum(sim * hit / ((1 - alpha) * np.asarray(es)[None, :]), axis=1) / sim.shape[1] + 1
    return z, float(np.mean(zs <= z))
