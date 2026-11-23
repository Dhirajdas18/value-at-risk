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


ES_METHODS = {"historical, 250 days": ("historical", 250),
              "historical, 1000 days": ("historical", 1000),
              "normal, 250 days": ("normal", 250),
              "filtered historical, EWMA": ("filtered", 500)}


def report_es(out_dir: Path) -> dict:
    """97.5% expected shortfall (Basel's measure) vs 99% VaR, and the Acerbi-Szekely backtest."""
    from var.es import es_forecasts, z2_pvalue

    R = returns()
    R = R[R.index >= START]
    rng = np.random.default_rng(0)
    res = {}
    for name in ("NIFTY ETF", "5 large stocks"):
        h = PORTFOLIOS[name]
        p = portfolio_returns(R[list(h)].dropna(), h)
        es = {k: es_forecasts(p, m, w, 0.975) for k, (m, w) in ES_METHODS.items()}
        v99 = {k: es_forecasts(p, m, w, 0.99) for k, (m, w) in ES_METHODS.items()}
        start = max(f.index[0] for f in es.values())
        rows = {}
        for k, f in es.items():
            keep = f.index >= start
            ff = f[keep]
            x = p.reindex(ff.index).to_numpy()
            z, pv = z2_pvalue(x, ff["var"].to_numpy(), ff["es"].to_numpy(), 0.975,
                              lambda n, f=f, keep=keep: f.attrs["draw"](rng, n)[:, keep], 1000)
            hit = x < -ff["var"].to_numpy()
            rows[k] = {"es": ff["es"], "var975": ff["var"], "x": pd.Series(x, ff.index),
                       "es_mean": float(ff["es"].mean()),
                       "var99_mean": float(v99[k].loc[v99[k].index >= start, "var"].mean()),
                       "z2": z, "p": pv, "breach": float(hit.mean())}
        res[name] = rows
    names = list(ES_METHODS)
    nifty = res["NIFTY ETF"]

    fig, ax = plt.subplots(figsize=(9, 4.6))
    xs = np.arange(len(names))
    a = [nifty[k]["var99_mean"] * 100 for k in names]
    b = [nifty[k]["es_mean"] * 100 for k in names]
    ax.bar(xs - 0.2, a, 0.38, color=plots.MUTED, label="99% VaR (the old rule)")
    ax.bar(xs + 0.2, b, 0.38, color=plots.SERIES[0], label="97.5% expected shortfall (Basel now)")
    for i in range(len(names)):
        ax.text(i - 0.2, a[i] + 0.03, f"{a[i]:.2f}", ha="center", fontsize=8)
        ax.text(i + 0.2, b[i] + 0.03, f"{b[i]:.2f}", ha="center", fontsize=8)
    ax.set_xticks(xs, [n.replace(", ", "\n") for n in names], fontsize=9)
    ax.set_ylabel("average over 2013–2026, % of the NIFTY ETF")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper left", fontsize=9)
    ax.set_title("Switching to 97.5% shortfall barely changed the capital number")
    plots.save(fig, out_dir, "es_capital")

    fig, ax = plt.subplots(figsize=(9, 4.6))
    for j, (port, color) in enumerate((("NIFTY ETF", plots.SERIES[0]),
                                       ("5 large stocks", plots.SERIES[2]))):
        zs = [res[port][k]["z2"] for k in names]
        ax.bar(xs - 0.2 + 0.4 * j, zs, 0.38, color=color, label=port)
        for i, k in enumerate(names):
            pv = res[port][k]["p"]
            ax.text(i - 0.2 + 0.4 * j, zs[i] + (0.02 if zs[i] >= 0 else -0.05),
                    "p<0.01" if pv < 0.01 else f"p={pv:.2f}", ha="center", fontsize=8,
                    color=plots.TEXT_2)
    ax.axhline(0, color=plots.TEXT, lw=1)
    ax.set_xticks(xs, [n.replace(", ", "\n") for n in names], fontsize=9)
    ax.set_ylabel("Acerbi-Székely Z2 (0 = right; below = ES too small)")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="lower left", fontsize=9)
    ax.set_title("Shortfall backtest: normal fails on both; one-year history fails on NIFTY")
    plots.save(fig, out_dir, "es_backtest")

    fig, ax = plt.subplots(figsize=(9, 4.6))
    years = sorted(set(nifty[names[0]]["x"].index.year))
    xs = np.arange(len(years))
    for j, (k, color) in enumerate((("historical, 250 days", plots.SERIES[1]),
                                    ("filtered historical, EWMA", plots.SERIES[0]))):
        d = nifty[k]
        hit = d["x"] < -d["var975"]
        actual = (-d["x"][hit]).groupby(hit[hit].index.year).mean()
        forecast = d["es"][hit].groupby(hit[hit].index.year).mean()
        gap = ((actual - forecast) * 100).reindex(years)
        ax.bar(xs - 0.2 + 0.4 * j, gap, 0.38, color=color, label=k)
    ax.axhline(0, color=plots.TEXT, lw=1)
    ax.set_xticks(xs, years, fontsize=8)
    ax.set_ylabel("actual loss minus forecast ES on breach days, pts")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper left", fontsize=9)
    ax.set_title("When breached, losses beat the one-year forecast most in 2020")
    plots.save(fig, out_dir, "es_by_year")

    return {port: {k: (f"ES97.5 {v['es_mean']:.2%}", f"VaR99 {v['var99_mean']:.2%}",
                       f"ratio {v['es_mean'] / v['var99_mean']:.2f}",
                       f"breach97.5 {v['breach']:.2%}", f"Z2 {v['z2']:.3f}", f"p {v['p']:.3f}")
                   for k, v in rows.items()} for port, rows in res.items()}


def _t_var(p: pd.Series, q: float) -> float:
    """Student-t VaR with degrees of freedom matched to the excess kurtosis."""
    from scipy.stats import t as tdist

    k = kurtosis(p)
    df = 4 + 6 / k
    scale = p.std() * np.sqrt((df - 2) / df)
    return float(-(p.mean() + scale * tdist.ppf(1 - q, df))), df, scale


def report_evt(out_dir: Path) -> dict:
    """Peaks over threshold on the NIFTY ETF: the far tail, rare-loss frequencies, a split test."""
    from scipy.stats import poisson
    from scipy.stats import t as tdist

    from var.evt import evt_es, evt_var, exceed_probability, fit_tail

    R = returns()
    p = R.loc[R.index >= START, "NIFTYBEES"].dropna()
    tail = fit_tail(p, 0.95)
    mu, sd = float(p.mean()), float(p.std())
    _, df, scale = _t_var(p, 0.99)
    years = (p.index[-1] - p.index[0]).days / 365.25
    per_year = len(p) / years

    loss = np.sort(-p.to_numpy())[::-1]
    emp = np.arange(1, len(loss) + 1) / len(loss)
    big = loss > 0.01
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.plot(loss[big] * 100, emp[big], "o", ms=3, color=plots.MUTED, label="NIFTY ETF, actual")
    xs = np.linspace(0.01, 0.15, 300)
    ax.plot(xs * 100, norm.sf((xs + mu) / sd), color=plots.SERIES[0], label="normal curve")
    ax.plot(xs * 100, tdist.sf((xs + mu) / scale, df), color=plots.SERIES[2],
            label=f"Student-t ({df:.1f} degrees of freedom)")
    xe = xs[xs > tail["u"]]
    ax.plot(xe * 100, [exceed_probability(tail, x) for x in xe], color=plots.SERIES[1], lw=2,
            label=f"extreme value fit (ξ = {tail['xi']:.2f})")
    ax.set_yscale("log")
    ax.set_ylim(1e-6, 0.3)
    ax.set_xlabel("one-day loss, %")
    ax.set_ylabel("share of days with a bigger loss (log scale)")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_title("The normal curve's tail collapses; the extreme value fit follows the data")
    plots.save(fig, out_dir, "evt_tail")

    levels = [0.99, 0.995, 0.999]
    table = {q: {"historical": historical(p, q)[0], "normal": normal(mu, sd, q)[0],
                 "Student-t": _t_var(p, q)[0], "extreme value": evt_var(tail, q)}
             for q in levels}
    fig, ax = plt.subplots(figsize=(9, 4.6))
    meths = list(table[levels[0]])
    colors = [plots.TEXT, plots.SERIES[0], plots.SERIES[2], plots.SERIES[1]]
    xs_ = np.arange(len(levels))
    for j, (m, c) in enumerate(zip(meths, colors, strict=True)):
        vals = [table[q][m] * 100 for q in levels]
        ax.bar(xs_ - 0.3 + 0.2 * j, vals, 0.19, color=c, label=m)
        for i, v in enumerate(vals):
            ax.text(i - 0.3 + 0.2 * j, v + 0.05, f"{v:.1f}", ha="center", fontsize=7)
    ax.set_xticks(xs_, [f"{q:.1%}" for q in levels])
    ax.set_ylabel("one-day VaR, % of the NIFTY ETF")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper left", fontsize=8)
    ax.set_title("At 99.9% the normal curve puts the loss at half what history shows")
    plots.save(fig, out_dir, "evt_levels")

    rng = np.random.default_rng(0)
    thresholds = [0.90, 0.925, 0.95, 0.965, 0.975]
    xi_mid, xi_lo, xi_hi = [], [], []
    for th in thresholds:
        xi_mid.append(fit_tail(p, th)["xi"])
        boot = [fit_tail(rng.choice(p.to_numpy(), len(p)), th)["xi"] for _ in range(200)]
        xi_lo.append(np.quantile(boot, 0.05))
        xi_hi.append(np.quantile(boot, 0.95))
    fig, ax = plt.subplots(figsize=(9, 4.6))
    tx = [t_ * 100 for t_ in thresholds]
    ax.fill_between(tx, xi_lo, xi_hi, color=plots.SERIES[1], alpha=0.15, lw=0,
                    label="90% bootstrap range")
    ax.plot(tx, xi_mid, "o-", color=plots.SERIES[1], label="estimated tail shape ξ")
    ax.axhline(0, color=plots.TEXT_2, lw=1, ls="--")
    ax.text(tx[0], 0.01, "0 = thin, normal-like tail", fontsize=8, color=plots.TEXT_2)
    ax.set_xlabel("threshold: losses beyond this percentile are treated as the tail")
    ax.set_ylabel("ξ")
    ax.legend(loc="upper left", fontsize=9)
    ax.set_title("Whichever threshold, the tail shape stays clearly above zero")
    plots.save(fig, out_dir, "evt_threshold")

    a, b = p[p.index < "2020-01-01"], p[p.index >= "2020-01-01"]
    ta = fit_tail(a, 0.95)
    split = {"historical": historical(a, 0.999)[0], "normal": normal(a.mean(), a.std(), 0.999)[0],
             "Student-t": _t_var(a, 0.999)[0], "extreme value": evt_var(ta, 0.999)}
    expected = len(b) * 0.001
    split_out = {m: (f"{v:.2%}", int((b < -v).sum()),
                     round(float(poisson.sf(int((b < -v).sum()) - 1, expected)), 3))
                 for m, v in split.items()}
    freq = {}
    for x in (0.05, 0.08, 0.10, 0.13):
        freq[f"loss > {x:.0%}"] = {
            "actual days": int((p < -x).sum()),
            "EVT every (years)": round(1 / (exceed_probability(tail, x) * per_year), 1),
            "Student-t every (years)": round(1 / (tdist.sf((x + mu) / scale, df) * per_year), 1),
            "normal every (years)": f"{1 / (norm.sf((x + mu) / sd) * per_year):.2g}"}
    return {
        "tail fit": {k: round(v, 4) if isinstance(v, float) else v for k, v in tail.items()},
        "xi by threshold (low, mid, high)": {f"{t_:.1%}": (round(l_, 2), round(m_, 2), round(h_, 2))
                                             for t_, l_, m_, h_ in zip(thresholds, xi_lo, xi_mid,
                                                                       xi_hi, strict=True)},
        "VaR by level": {f"{q:.1%}": {m: f"{v:.2%}" for m, v in d.items()}
                         for q, d in table.items()},
        "ES 99.9%: historical / EVT": (f"{historical(p, 0.999)[1]:.2%}",
                                       f"{evt_es(tail, 0.999):.2%}"),
        "days beyond historical 99.9%": int((p < -historical(p, 0.999)[0]).sum()),
        "frequencies": freq,
        "split: fit 2009-19, test 2020-26 (VaR99.9, breaches, Poisson p)": split_out,
        "split expected breaches": round(expected, 2),
        "years": round(years, 1),
    }


REPORTS = {"methods": report_methods, "backtest": report_backtest, "fhs": report_fhs,
           "es": report_es, "evt": report_evt}


def report_decomp(out_dir: Path) -> dict:
    """Which holdings carry the risk, what a trade adds, and history's worst episodes replayed."""
    from var.decomp import SCENARIOS, component_es, component_var_normal, incremental_es, stress

    R = returns()
    Rs = R[R.index >= START]
    books = {n: {k: v * CAPITAL for k, v in w.items()} for n, w in PORTFOLIOS.items()}
    shares = {}
    for name in ("5 large stocks", "70% NIFTY / 30% gold"):
        h = books[name]
        c, cv = component_es(Rs, h), component_var_normal(Rs, h)["component"]
        shares[name] = pd.DataFrame({"money": pd.Series(h) / CAPITAL, "ES": c / c.sum(),
                                     "normal VaR": cv / cv.sum()})

    fig, ax = plt.subplots(figsize=(9, 4.8))
    rows = [(n, s) for n, d in shares.items() for s in d.index]
    y = np.arange(len(rows))[::-1].astype(float)
    y[: len(shares["5 large stocks"])] += 0.6  # gap between the two portfolios
    money = [shares[n].loc[s, "money"] * 100 for n, s in rows]
    risk = [shares[n].loc[s, "ES"] * 100 for n, s in rows]
    ax.barh(y + 0.2, money, 0.38, color=plots.MUTED, label="share of the money")
    ax.barh(y - 0.2, risk, 0.38, color=plots.SERIES[0], label="share of 97.5% expected shortfall")
    for yi, m, r_ in zip(y, money, risk, strict=True):
        ax.text(max(m, r_) + 1, yi, f"{r_:.0f}%", va="center", fontsize=8, color=plots.TEXT_2)
    ax.set_yticks(y, [f"{s}  ({'5 stocks' if n.startswith('5') else '70/30'})" for n, s in rows])
    ax.set_xlabel("% of the ₹10 lakh portfolio")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower right", fontsize=8)
    g = shares["70% NIFTY / 30% gold"].loc["GOLDBEES"]
    ax.set_title(f"Gold is {g['money']:.0%} of the money but {g['ES']:.0%} of the risk")
    plots.save(fig, out_dir, "decomp_shares")

    h = books["70% NIFTY / 30% gold"]
    c = component_es(Rs, h)
    marginal = c / pd.Series(h)
    sizes = np.linspace(0, 1_000_000, 21)
    inc = {s: [incremental_es(Rs, h, s, a) if a else 0.0 for a in sizes]
           for s in ("NIFTYBEES", "GOLDBEES")}
    fig, ax = plt.subplots(figsize=(9, 4.6))
    for s, color, label in (("NIFTYBEES", plots.SERIES[0], "NIFTY ETF"),
                            ("GOLDBEES", plots.SERIES[1], "gold ETF")):
        ax.plot(sizes / 1e5, np.array(inc[s]) / 1e3, color=color, lw=2,
                label=f"buy more {label}: recomputed")
        ax.plot(sizes / 1e5, marginal[s] * sizes / 1e3, color=color, lw=1.2, ls="--",
                label=f"buy more {label}: marginal × amount")
    ax.set_xlabel("₹ lakh added to the 70/30 portfolio")
    ax.set_ylabel("change in 97.5% expected shortfall, ₹ thousand")
    ax.legend(loc="upper left", fontsize=8)
    gold_gap = 1 - marginal["GOLDBEES"] * sizes[-1] / inc["GOLDBEES"][-1]
    ax.set_title(f"The marginal estimate holds for small trades; for ₹10 lakh of gold it is "
                 f"{gold_gap:.0%} short")
    plots.save(fig, out_dir, "decomp_incremental")

    var99 = {n: historical(portfolio_returns(Rs[list(b)].dropna(), b), 0.99)[0] * CAPITAL
             for n, b in books.items()}
    st = {n: {sc: stress(R, b, *SCENARIOS[sc]) for sc in SCENARIOS} for n, b in books.items()}
    fig, ax = plt.subplots(figsize=(9, 4.8))
    xs = np.arange(len(SCENARIOS))
    for j, (n, color) in enumerate(zip(books, (plots.SERIES[0], plots.SERIES[2],
                                                plots.SERIES[1]), strict=True)):
        loss = [-st[n][sc]["pnl"].sum() / 1e3 for sc in SCENARIOS]
        ax.bar(xs - 0.27 + 0.27 * j, loss, 0.25, color=color, label=n)
        for i, v in enumerate(loss):
            ax.text(i - 0.27 + 0.27 * j, v + 6, f"{v:.0f}", ha="center", fontsize=7)
    ax.axhline(var99["NIFTY ETF"] / 1e3, color=plots.TEXT_2, ls="--", lw=1,
               label=f"NIFTY ETF's 99% one-day VaR: ₹{var99['NIFTY ETF'] / 1e3:.0f}k")
    days = {sc: int(((R.index > a) & (R.index <= b)).sum()) for sc, (a, b) in SCENARIOS.items()}
    ax.set_xticks(xs, [f"{sc}\n{days[sc]} trading day{'s' if days[sc] > 1 else ''}"
                       + ("\nETFs proxied" if sc == "2008 crash" else "") for sc in SCENARIOS])
    ax.set_ylabel("loss on ₹10 lakh held today, ₹ thousand")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper right", fontsize=8)
    covid = -st["NIFTY ETF"]["COVID crash"]["pnl"].sum() / var99["NIFTY ETF"]
    ax.set_title(f"The COVID crash cost the NIFTY ETF {covid:.0f}× its 99% one-day VaR "
                 f"in {days['COVID crash']} trading days")
    plots.save(fig, out_dir, "stress_scenarios")

    return {
        "shares (money, ES97.5, normal VaR99)": {
            n: {s: tuple(f"{x:.1%}" for x in row) for s, row in d.iterrows()}
            for n, d in shares.items()},
        "70/30 ES97.5 ₹": round(c.sum()),
        "marginal ES per ₹ (70/30)": marginal.round(4).to_dict(),
        "incremental ES ₹ (recomputed, marginal) at 0.1/1/3/10 lakh": {
            s: [(round(inc[s][i]), round(marginal[s] * sizes[i])) for i in (0, 2, 6, 20)]
            for s in inc},
        "99% one-day VaR ₹": {n: round(v) for n, v in var99.items()},
        "stress ₹ (loss, × VaR99)": {
            n: {sc: (round(-d["pnl"].sum()), round(-d["pnl"].sum() / var99[n], 1))
                for sc, d in st[n].items()} for n in books},
        "stress returns by holding": {
            n: {sc: {s: (f"{r_['return']:.1%}", r_["source"]) for s, r_ in d.iterrows()}
                for sc, d in st[n].items()} for n in books},
        "scenario days": days,
    }


REPORTS["decomp"] = report_decomp


def option_scenarios(days: int) -> tuple[pd.Series, pd.Series]:
    """NIFTY's move and India VIX's ratio over every overlapping `days`-day window since START."""
    t = pd.read_parquet(DATA_PATH)["close"][["NIFTY", "INDIAVIX"]].loc[START:].dropna()
    ret = (t["NIFTY"] / t["NIFTY"].shift(days) - 1).dropna()
    return ret, (t["INDIAVIX"] / t["INDIAVIX"].shift(days)).dropna()


OPTION_METHODS = ("delta", "delta-gamma", "delta-gamma-vega", "full")


def report_options(out_dir: Path) -> dict:
    """VaR of three NIFTY option books by Greeks approximation and by full repricing."""
    from var.options import BOOKS, book_pnl

    t = pd.read_parquet(DATA_PATH)["close"][["NIFTY", "INDIAVIX"]].dropna()
    S, vol, asof = float(t["NIFTY"].iloc[-1]), float(t["INDIAVIX"].iloc[-1]) / 100, t.index[-1]
    horizons = {1: 1 / 365, 10: 14 / 365}  # 10 trading days ≈ 14 calendar days of decay
    var = {}
    pnl10 = {}
    for h, dt in horizons.items():
        ret, vr = option_scenarios(h)
        for name, make in BOOKS.items():
            for m in OPTION_METHODS:
                p = book_pnl(make(S), S, vol, ret.to_numpy(), vr.to_numpy(), dt, m)
                var[(h, name, m)] = -float(np.quantile(p, 0.01))
                if h == 10:
                    pnl10[(name, m)] = p
    ret10, vr10 = option_scenarios(10)

    colors = {"delta": plots.SERIES[2], "delta-gamma": plots.SERIES[0],
              "delta-gamma-vega": plots.SERIES[3], "full": plots.TEXT}
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6), sharey=True)
    xs = np.linspace(-0.31, 0.21, 300)
    for ax, name in zip(axes, ("short 5% OTM put", "short straddle"), strict=True):
        book = BOOKS[name](S)
        order = np.argsort(ret10.to_numpy())
        ax.plot(ret10.to_numpy()[order] * 100, pnl10[(name, "full")][order] / 1e3, "o", ms=2,
                color=plots.MUTED, label="full repricing, each 10-day window (with VIX)")
        for m, ls in (("delta", "--"), ("delta-gamma", "-"), ("full", "-")):
            ax.plot(xs * 100, book_pnl(book, S, vol, xs, np.ones_like(xs), 14 / 365, m) / 1e3,
                    color=colors[m], ls=ls, lw=2 if m == "full" else 1.5,
                    label=f"{m}{', VIX unchanged' if m == 'full' else ''}")
        ax.axhline(0, color=plots.TEXT_2, lw=0.8)
        ax.set_title(name, fontsize=10)
        ax.set_xlabel("NIFTY move over 10 trading days, %")
    axes[0].set_ylabel("P&L on ₹10 lakh of notional, ₹ thousand")
    axes[0].set_ylim(-260, 40)
    axes[0].legend(loc="lower right", fontsize=7)
    fig.suptitle("A straight line misses both books; the parabola misses on big moves",
                 fontweight="bold")
    plots.save(fig, out_dir, "options_payoff")

    fig, ax = plt.subplots(figsize=(9, 4.6))
    books = list(BOOKS)
    xb = np.arange(len(books))
    for j, m in enumerate(OPTION_METHODS):
        vals = [var[(10, b, m)] / 1e3 for b in books]
        ax.bar(xb - 0.3 + 0.2 * j, vals, 0.19, color=colors[m], label=m)
        for i, v in enumerate(vals):
            ax.text(i - 0.3 + 0.2 * j, v + 1.5, f"{v:.0f}", ha="center", fontsize=7)
    ax.set_xticks(xb, books)
    ax.set_ylabel("99% 10-day VaR, ₹ thousand")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper left", fontsize=8)
    lo = 1 - var[(10, "short 5% OTM put", "delta-gamma")] / var[(10, "short 5% OTM put", "full")]
    hi = var[(10, "short straddle", "delta-gamma")] / var[(10, "short straddle", "full")] - 1
    ax.set_title(f"Delta-gamma: the short put's 10-day VaR {lo:.0%} too low, "
                 f"the straddle's {hi:.0%} too high")
    plots.save(fig, out_dir, "options_var")

    fig, ax = plt.subplots(figsize=(9, 4.6))
    order = np.argsort(ret10.to_numpy())
    for name, c in (("short 5% OTM put", plots.SERIES[1]), ("short straddle", plots.SERIES[0])):
        err = (pnl10[(name, "delta-gamma")] - pnl10[(name, "full")])[order] / 1e3
        ax.plot(ret10.to_numpy()[order] * 100, err, "o", ms=2, color=c, label=name)
    ax.axhline(0, color=plots.TEXT_2, lw=0.8)
    ax.set_xlabel("NIFTY move over 10 trading days, %")
    ax.set_ylabel("delta-gamma minus repricing, ₹ thousand\n(below 0: overstates the loss)")
    ax.legend(loc="lower right", fontsize=8)
    ax.set_title("The approximation's error grows with the size of the move")
    plots.save(fig, out_dir, "options_error")

    return {
        "priced at": f"{asof:%d %b %Y}: NIFTY {S:,.0f}, India VIX {vol:.1%}, 30 days to expiry",
        "scenarios (1-day, 10-day)": (len(option_scenarios(1)[0]), len(ret10)),
        "99% VaR ₹ (horizon, book: method -> VaR, error vs full)": {
            f"{h}d {b}": {m: (round(var[(h, b, m)]),
                              f"{var[(h, b, m)] / var[(h, b, 'full')] - 1:+.0%}")
                          for m in OPTION_METHODS}
            for h in horizons for b in books},
        "worst 10-day NIFTY move": f"{ret10.min():.1%} ending {ret10.idxmin():%d %b %Y}, "
                                    f"VIX ×{vr10[ret10.idxmin()]:.2f}",
        "10-day loss on that window ₹ (book: method)": {
            b: {m: round(-pnl10[(b, m)][ret10.to_numpy().argmin()]) for m in OPTION_METHODS}
            for b in books},
    }


REPORTS["options"] = report_options
