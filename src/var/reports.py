"""README findings, one function per topic: `var report <topic>` writes charts to out/."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import kurtosis, norm, skew

from var import plots
from var.data import DATA_PATH, daily_returns
from var.var import (
    cornish_fisher,
    cornish_fisher_valid,
    historical,
    horizon_returns,
    monte_carlo,
    normal,
    portfolio_returns,
)

CAPITAL = 1_000_000.0  # ₹10 lakh
START = "2009-01-02"  # first day both ETFs trade on Yahoo
PORTFOLIOS = {
    "NIFTY ETF": {"NIFTYBEES": 1.0},
    "5 large stocks": {s: 0.2 for s in ("RELIANCE", "HDFCBANK", "INFY", "TCS", "ITC")},
    "70% NIFTY / 30% gold": {"NIFTYBEES": 0.7, "GOLDBEES": 0.3},
}


def returns() -> pd.DataFrame:
    return daily_returns(pd.read_parquet(DATA_PATH)["adj_close"])


def method_table(r: pd.DataFrame, holdings: dict, alpha: float = 0.99, seed: int = 0) -> dict:
    """VaR and ES (fractions) by method for one portfolio over the rows of `r`."""
    r = r[list(holdings)].dropna()
    p = portfolio_returns(r, holdings)
    w = (pd.Series(holdings) / sum(holdings.values())).to_numpy()
    mu, sd, sk, ku = float(p.mean()), float(p.std()), float(skew(p)), float(kurtosis(p))
    df = max(4 + 6 / ku, 2.5) if ku > 0 else None  # Student-t matched to the excess kurtosis
    return {
        "historical": historical(p, alpha),
        "normal": normal(mu, sd, alpha),
        "Cornish-Fisher": cornish_fisher(mu, sd, sk, ku, alpha),
        "Monte Carlo, normal": monte_carlo(r.mean(), r.cov(), w, alpha, 200_000, seed),
        "Monte Carlo, Student-t": monte_carlo(r.mean(), r.cov(), w, alpha, 200_000, seed, df),
        "_moments": (mu, sd, sk, ku, df, len(p), cornish_fisher_valid(sk, ku)),
    }


def report_methods(out_dir: Path) -> dict:
    """Four ways to compute VaR and ES on three ₹10 lakh portfolios, and the √10 shortcut."""
    R = returns()
    R = R[R.index >= START]
    tables = {name: method_table(R, h) for name, h in PORTFOLIOS.items()}
    nifty = R["NIFTYBEES"].dropna()
    t = tables["NIFTY ETF"]
    methods = [m for m in t if not m.startswith("_")]

    fig, ax = plt.subplots(figsize=(9, 4.6))
    pnl = nifty * CAPITAL / 1000
    bins = np.linspace(-60, 60, 121)
    ax.hist(pnl.clip(-60, 60), bins=bins, color=plots.MUTED, density=True, label="daily P&L")
    mu, sd = pnl.mean(), pnl.std()
    x = np.linspace(-60, 60, 400)
    ax.plot(x, norm.pdf(x, mu, sd), color=plots.SERIES[0], lw=1.5, label="normal curve")
    marks = {"historical": plots.TEXT, "normal": plots.SERIES[0],
             "Monte Carlo, Student-t": plots.SERIES[1]}
    for m, color in marks.items():
        v = -t[m][0] * CAPITAL / 1000
        ax.axvline(v, color=color, ls="--", lw=1.2, label=f"99% VaR, {m}: ₹{-v:.1f}k")
    ax.set_xlim(-60, 25)
    ax.annotate("days worse than −₹60k\npiled at the edge", (-59, 0.002), xytext=(-57, 0.012),
                fontsize=8, color=plots.TEXT_2, arrowprops=dict(arrowstyle="-", color=plots.MUTED))
    ax.set_xlabel("daily profit or loss on ₹10 lakh in the NIFTY ETF, ₹ thousand (2009–2026)")
    ax.set_ylabel("share of days (density)")
    ax.legend(loc="upper left", fontsize=8)
    ax.set_title("The worst days sit far beyond where a normal curve puts them")
    plots.save(fig, out_dir, "methods_distribution")

    fig, axes = plt.subplots(1, 2, figsize=(9, 4.8), sharey=True)
    show = [m for m in methods if m != "Cornish-Fisher"]
    colors = [plots.TEXT, plots.SERIES[0], plots.SERIES[2], plots.SERIES[1]]
    ys = np.arange(len(PORTFOLIOS))[::-1]
    for ax, k, label in ((axes[0], 0, "99% VaR"), (axes[1], 1, "99% expected shortfall")):
        for j, (m, c) in enumerate(zip(show, colors, strict=True)):
            vals = [tables[p][m][k] * CAPITAL / 1000 for p in PORTFOLIOS]
            ax.barh(ys + 0.3 - j * 0.2, vals, 0.19, color=c, label=m)
        ax.set_title(f"{label}, ₹ thousand", fontsize=10, fontweight="normal",
                     color=plots.TEXT_2)
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(ys, list(PORTFOLIOS))
    axes[0].legend(loc="upper center", bbox_to_anchor=(1.05, -0.08), ncol=4, fontsize=8,
                   frameon=False)
    fig.suptitle("A normal curve understates the loss, and the shortfall most of all", x=0.06,
                 ha="left", fontsize=13, fontweight="semibold", color=plots.TEXT)
    plots.save(fig, out_dir, "methods_compare")

    levels = [0.90, 0.95, 0.975, 0.99, 0.995]
    scaled = [historical(nifty, a)[0] * np.sqrt(10) * 100 for a in levels]
    actual = [historical(horizon_returns(nifty, 10), a)[0] * 100 for a in levels]
    fig, ax = plt.subplots(figsize=(9, 4.6))
    xs = np.arange(len(levels))
    ax.bar(xs - 0.2, scaled, 0.38, color=plots.MUTED, label="1-day VaR × √10")
    ax.bar(xs + 0.2, actual, 0.38, color=plots.SERIES[0], label="actual 10-day VaR")
    for i in range(len(levels)):
        ax.text(i - 0.2, scaled[i] + 0.1, f"{scaled[i]:.1f}", ha="center", fontsize=8)
        ax.text(i + 0.2, actual[i] + 0.1, f"{actual[i]:.1f}", ha="center", fontsize=8)
    ax.set_xticks(xs, [f"{a:.1%}" for a in levels])
    ax.set_xlabel("confidence level")
    ax.set_ylabel("10-day VaR, % of the NIFTY ETF")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper left")
    ax.set_title("√10 scaling works at 95% and overstates the extreme 10-day loss")
    plots.save(fig, out_dir, "methods_horizon")

    g = (1 + nifty).cumprod()
    kurt_by_h = {h: float(kurtosis(g.iloc[::h].pct_change().dropna())) for h in (1, 5, 10, 20)}
    rupees = lambda f: f"₹{f * CAPITAL:,.0f}"  # noqa: E731
    out = {}
    for name, tb in tables.items():
        mu, sd, sk, ku, df, n, valid = tb["_moments"]
        out[name] = {m: (rupees(tb[m][0]), rupees(tb[m][1])) for m in methods}
        out[name]["days, skew, excess kurtosis, t df, CF valid"] = (n, round(sk, 2), round(ku, 1),
                                                                   round(df, 2), valid)
        breach = float((portfolio_returns(R[list(PORTFOLIOS[name])].dropna(), PORTFOLIOS[name])
                        < -tb["normal"][0]).mean())
        out[name]["share of days beyond normal 99% VaR"] = f"{breach:.2%}"
    out["NIFTY ETF 10-day: √10 vs actual by level"] = {
        f"{a:.1%}": (f"{s:.2f}%", f"{a_:.2f}%") for a, s, a_ in zip(levels, scaled, actual,
                                                                    strict=True)}
    out["NIFTY ETF excess kurtosis by horizon (non-overlapping)"] = {
        f"{h}d": round(v, 1) for h, v in kurt_by_h.items()}
    return out


REPORTS = {"methods": report_methods}
