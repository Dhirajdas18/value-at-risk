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


BACKTEST_METHODS = {"historical, 250 days": ("historical", 250),
                    "historical, 1000 days": ("historical", 1000),
                    "normal, 250 days": ("normal", 250),
                    "Student-t, 250 days": ("student-t", 250)}


def backtest_table(p: pd.Series, methods: dict = BACKTEST_METHODS, alpha: float = 0.99):
    """Rolling forecasts for each method, scored on the same days (from the first day the
    longest window allows). Returns {method: dict of results} and the forecasts."""
    from var.backtest import christoffersen, kupiec, rolling_var

    fc = {k: rolling_var(p, m, w, alpha) for k, (m, w) in methods.items()}
    start = max(f.index[0] for f in fc.values())
    out = {}
    for k, f in fc.items():
        f = f[f.index >= start]
        hit = p.reindex(f.index) < -f
        n, x = len(hit), int(hit.sum())
        roll = hit.rolling(250).sum().dropna()
        out[k] = {"var": f, "hits": hit, "n": n, "x": x, "rate": x / n,
                  "kupiec": kupiec(x, n, 1 - alpha), "indep": christoffersen(hit.to_numpy()),
                  "after": float(hit[hit.shift(1, fill_value=False)].mean()),
                  "roll": roll, "red": float((roll >= 10).mean()),
                  "yellow": float(((roll >= 5) & (roll < 10)).mean())}
    return out


def report_backtest(out_dir: Path) -> dict:
    """Rolling 99% VaR on the NIFTY ETF and five stocks since 2013: count, clustering, Basel."""
    R = returns()
    R = R[R.index >= START]
    results = {name: backtest_table(portfolio_returns(R[list(h)].dropna(), h), BACKTEST_METHODS)
               for name, h in PORTFOLIOS.items() if name != "70% NIFTY / 30% gold"}
    nifty = results["NIFTY ETF"]
    p = R["NIFTYBEES"].dropna()

    fig, ax = plt.subplots(figsize=(9, 4.6))
    idx = nifty["historical, 250 days"]["var"].index
    ax.plot(idx, p.reindex(idx) * 100, color=plots.MUTED, lw=0.6, label="daily return")
    for k, c in (("historical, 250 days", plots.SERIES[0]),
                 ("historical, 1000 days", plots.SERIES[1])):
        ax.plot(idx, -nifty[k]["var"] * 100, color=c, lw=1.4, label=f"99% VaR, {k}")
    hits = nifty["historical, 1000 days"]["hits"]
    ax.plot(hits[hits].index, p.reindex(hits[hits].index) * 100, "o", ms=3,
            color=plots.SERIES[1], label="breach of the 1000-day VaR")
    ax.set_ylim(-14, 6)
    ax.set_ylabel("NIFTY ETF daily return, %")
    ax.legend(loc="lower left", fontsize=8)
    ax.set_title("A slow VaR gets the count right but its breaches arrive in bunches")
    plots.save(fig, out_dir, "backtest_breaches")

    names = list(BACKTEST_METHODS)
    fig, ax = plt.subplots(figsize=(9, 4.6))
    xs = np.arange(len(names))
    for j, (port, color) in enumerate((("NIFTY ETF", plots.SERIES[0]),
                                       ("5 large stocks", plots.SERIES[2]))):
        rates = [results[port][k]["rate"] * 100 for k in names]
        ax.bar(xs - 0.2 + 0.4 * j, rates, 0.38, color=color, label=port)
        for i, k in enumerate(names):
            pv = results[port][k]["kupiec"][1]
            ax.text(i - 0.2 + 0.4 * j, rates[i] + 0.03, f"p={pv:.2f}" if pv >= 0.01 else "p<0.01",
                    ha="center", fontsize=8, color=plots.TEXT_2)
    ax.axhline(1, color=plots.TEXT, ls="--", lw=1)
    ax.text(-0.45, 1.03, "promised: 1%", ha="left", fontsize=8)
    ax.set_xticks(xs, [n.replace(", ", "\n") for n in names], fontsize=9)
    ax.set_ylabel("share of days the 99% VaR was breached, %")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper right", fontsize=9)
    ax.set_title("Only the 1000-day historical VaR breached about as often as promised")
    plots.save(fig, out_dir, "backtest_rates")

    fig, ax = plt.subplots(figsize=(9, 4.6))
    roll_a = nifty["historical, 250 days"]["roll"]
    roll_b = nifty["historical, 1000 days"]["roll"]
    ax.axhspan(-0.5, 4.5, color=plots.SERIES[2], alpha=0.10)
    ax.axhspan(4.5, 9.5, color=plots.SERIES[3], alpha=0.15)
    ax.axhspan(9.5, max(roll_a.max(), roll_b.max()) + 1, color=plots.SERIES[7], alpha=0.10)
    for y, z in ((2, "green"), (7, "yellow"), (12, "red")):
        ax.text(roll_a.index[int(len(roll_a) * 0.62)], y, f"Basel {z}", fontsize=8,
                color=plots.TEXT_2, va="center")
    ax.plot(roll_a.index, roll_a, color=plots.SERIES[0], label="historical, 250 days")
    ax.plot(roll_b.index, roll_b, color=plots.SERIES[1], label="historical, 1000 days")
    ax.set_ylabel("breaches of 99% VaR in the last 250 days")
    ax.legend(loc="upper right", fontsize=9)
    ax.set_title(f"Both hit the red zone in 2020; the slow one went deeper "
                 f"({int(roll_b.max())} breaches vs {int(roll_a.max())})")
    plots.save(fig, out_dir, "backtest_traffic")

    out = {}
    for port, res in results.items():
        out[port] = {k: {"breaches": f"{v['x']}/{v['n']} = {v['rate']:.2%}",
                         "kupiec p": round(v["kupiec"][1], 3),
                         "independence p": round(v["indep"][1], 3),
                         "P(breach | breach yesterday)": f"{v['after']:.1%}",
                         "days red / yellow": (f"{v['red']:.0%}", f"{v['yellow']:.0%}"),
                         "max breaches in 250 days": int(v["roll"].max()),
                         "average VaR": f"{v['var'].mean():.2%}"}
                     for k, v in res.items()}
    first = nifty["historical, 250 days"]["var"].index[0]
    out["evaluation starts"] = f"{first:%Y-%m-%d}"
    red_b = roll_b[roll_b >= 10]
    out["1000-day red spells"] = (f"{red_b.index[0]:%Y-%m-%d} to {red_b.index[-1]:%Y-%m-%d}"
                                  if len(red_b) else "none")
    return out


def fhs_forecasts(p: pd.Series, alpha: float = 0.99) -> dict:
    """Daily VaR forecasts (past data only) for the topic 3 comparison."""
    from var.backtest import rolling_var
    from var.fhs import ewma_vol, fhs_var, garch_vol

    e, g = ewma_vol(p), garch_vol(p, 500, 21)
    return {"historical, 250 days": rolling_var(p, "historical", 250, alpha),
            "historical, 1000 days": rolling_var(p, "historical", 1000, alpha),
            "EWMA volatility, normal curve": -e * norm.ppf(1 - alpha),
            "filtered historical, EWMA": fhs_var(p, e, 500, alpha),
            "filtered historical, GARCH": fhs_var(p, g, 500, alpha)}


def score(p: pd.Series, forecasts: dict, alpha: float = 0.99) -> dict:
    from var.backtest import christoffersen, kupiec

    start = max(f.dropna().index[0] for f in forecasts.values())
    out = {}
    for k, f in forecasts.items():
        f = f[f.index >= start].dropna()
        hit = p.reindex(f.index) < -f
        n, x = len(hit), int(hit.sum())
        roll = hit.rolling(250).sum().dropna()
        out[k] = {"var": f, "hits": hit, "rate": x / n, "x": x, "n": n,
                  "kupiec": kupiec(x, n, 1 - alpha)[1], "indep": christoffersen(hit.to_numpy())[1],
                  "after": float(hit[hit.shift(1, fill_value=False)].mean()), "roll": roll,
                  "red": float((roll >= 10).mean())}
    return out


def report_fhs(out_dir: Path) -> dict:
    """Filtered historical simulation against topic 2's methods, NIFTY ETF and five stocks."""
    R = returns()
    R = R[R.index >= START]
    res = {}
    for name in ("NIFTY ETF", "5 large stocks"):
        h = PORTFOLIOS[name]
        p = portfolio_returns(R[list(h)].dropna(), h)
        res[name] = (p, score(p, fhs_forecasts(p)))
    p, sc = res["NIFTY ETF"]

    fig, ax = plt.subplots(figsize=(9, 4.6))
    lo, hi = "2019-10-01", "2021-06-30"
    seg = p.loc[lo:hi]
    ax.plot(seg.index, seg * 100, color=plots.MUTED, lw=0.8, label="daily return")
    for k, c in (("historical, 1000 days", plots.SERIES[1]),
                 ("filtered historical, EWMA", plots.SERIES[0])):
        v = sc[k]["var"].loc[lo:hi]
        ax.plot(v.index, -v * 100, color=c, lw=1.6, label=f"99% VaR, {k}")
        hits = sc[k]["hits"].loc[lo:hi]
        ax.plot(hits[hits].index, seg.reindex(hits[hits].index) * 100, "o", ms=4, color=c)
    ax.set_ylabel("NIFTY ETF daily return, %")
    ax.legend(loc="lower right", fontsize=8)
    ax.set_title("Filtered historical VaR widened within days of the 2020 crash")
    plots.save(fig, out_dir, "fhs_crash")

    names = list(sc)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 4.8), sharey=True)
    ys = np.arange(len(names))[::-1]
    rates = [sc[k]["rate"] * 100 for k in names]
    after = [sc[k]["after"] * 100 for k in names]
    a1.barh(ys, rates, 0.55, color=[plots.SERIES[0] if sc[k]["kupiec"] >= 0.05 else plots.MUTED
                                    for k in names])
    a1.axvline(1, color=plots.TEXT, ls="--", lw=1)
    a2.barh(ys, after, 0.55, color=[plots.SERIES[0] if sc[k]["indep"] >= 0.05 else plots.MUTED
                                    for k in names])
    a2.axvline(1, color=plots.TEXT, ls="--", lw=1)
    for i in range(len(names)):
        a1.text(rates[i] + 0.03, ys[i], f"{rates[i]:.2f}%", va="center", fontsize=8)
        a2.text(after[i] + 0.3, ys[i], f"{after[i]:.1f}%", va="center", fontsize=8)
    a1.set_yticks(ys, names, fontsize=9)
    a1.set_title("days breached (promise: 1%)", fontsize=10, fontweight="normal",
                 color=plots.TEXT_2)
    a2.set_title("breached again the next day (should be 1%)", fontsize=10, fontweight="normal",
                 color=plots.TEXT_2)
    for a in (a1, a2):
        a.grid(axis="y", visible=False)
    fig.suptitle("NIFTY ETF: only filtered historical VaR passed both tests (blue = passes)",
                 x=0.06, ha="left", fontsize=13, fontweight="semibold", color=plots.TEXT)
    plots.save(fig, out_dir, "fhs_scorecard")

    fig, ax = plt.subplots(figsize=(9, 4.6))
    ax.axhspan(-0.5, 4.5, color=plots.SERIES[2], alpha=0.10)
    ax.axhspan(4.5, 9.5, color=plots.SERIES[3], alpha=0.15)
    ax.axhspan(9.5, 17, color=plots.SERIES[7], alpha=0.10)
    for k, c in (("historical, 1000 days", plots.SERIES[1]),
                 ("filtered historical, EWMA", plots.SERIES[0])):
        roll = sc[k]["roll"]
        ax.plot(roll.index, roll, color=c, label=k)
    ax.set_ylabel("breaches of 99% VaR in the last 250 days")
    ax.legend(loc="upper left", fontsize=9)
    ax.set_title("Filtered historical VaR never reached Basel's red zone")
    plots.save(fig, out_dir, "fhs_traffic")

    out = {}
    for name, (_, s_) in res.items():
        out[name] = {k: (f"{v['x']}/{v['n']} = {v['rate']:.2%}", f"kupiec p {v['kupiec']:.3f}",
                         f"indep p {v['indep']:.3f}", f"after {v['after']:.1%}",
                         f"max250 {int(v['roll'].max())}", f"red {v['red']:.0%}",
                         f"avg VaR {v['var'].mean():.2%}") for k, v in s_.items()}
    crash = {k: f"{sc[k]['var'].loc['2020-03-23']:.2%}" for k in names}
    out["VaR on 23 Mar 2020"] = crash
    return out


REPORTS = {"methods": report_methods, "backtest": report_backtest, "fhs": report_fhs}
