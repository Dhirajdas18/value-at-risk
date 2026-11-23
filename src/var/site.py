"""Interactive dashboard: a live VaR calculator plus charts, as one static page."""

import json
import re
from pathlib import Path
from string import Template

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from var import plots

FONT = "system-ui, -apple-system, \"Segoe UI\", Roboto, sans-serif"

ICON = (
    "data:image/svg+xml,%3Csvg xmlns=%22http://www.w3.org/2000/svg%22 viewBox=%220 0 16 16%22%3E"
    "%3Cpath d=%22M1 12 5 7l3 3 7-8%22 fill=%22none%22 stroke=%22%232a78d6%22"
    " stroke-width=%222%22/%3E"
    "%3C/svg%3E"
)

def _layout(fig: go.Figure, ytitle: str) -> go.Figure:
    fig.update_layout(
        paper_bgcolor=plots.SURFACE,
        plot_bgcolor=plots.SURFACE,
        font=dict(family=FONT, color=plots.TEXT_2, size=13),
        hovermode="x unified",
        margin=dict(l=56, r=24, t=96, b=40),
        legend=dict(orientation="h", y=1.02, yanchor="bottom", x=0, xanchor="left"),
        yaxis=dict(title=ytitle, gridcolor=plots.GRID, zeroline=False, automargin=True),
        xaxis=dict(
            gridcolor=plots.GRID,
            rangeslider=dict(visible=False),
            rangeselector=dict(
                buttons=[dict(count=n, label=f"{n}y", step="year", stepmode="backward")
                         for n in (1, 3, 5)] + [dict(step="all", label="All")],
                x=1, xanchor="right", y=1.08, yanchor="bottom", bgcolor="white",
            ),
        ),
    )
    end = max(pd.Timestamp(max(t.x)) for t in fig.data)
    fig.update_xaxes(range=[end - pd.DateOffset(years=3), end])  # open on the last 3 years
    return fig


def _scatter_layout(fig: go.Figure, xtitle: str, ytitle: str) -> go.Figure:
    fig.update_layout(
        paper_bgcolor=plots.SURFACE, plot_bgcolor=plots.SURFACE,
        font=dict(family=FONT, color=plots.TEXT_2, size=13), hovermode="closest",
        margin=dict(l=56, r=24, t=56, b=48),
        legend=dict(orientation="h", y=1.02, yanchor="bottom", x=0, xanchor="left"),
        xaxis=dict(title=xtitle, gridcolor=plots.GRID, zeroline=False, rangemode="tozero"),
        yaxis=dict(title=ytitle, gridcolor=plots.GRID, zeroline=False, automargin=True),
    )
    return fig


CALCULATOR = Path(__file__).with_name("calculator.html").read_text(encoding="utf-8")
JS = Path(__file__).with_name("var.js").read_text(encoding="utf-8")
LABELS = {"NIFTYBEES": "NIFTY 50 ETF (NIFTYBEES)", "GOLDBEES": "Gold ETF (GOLDBEES)"}
CALC_DAYS = 1250  # five years of trading days embedded in the page
SCALE = 1e-5  # returns stored as integers in units of 0.001%


def calculator_data(r: pd.DataFrame) -> dict:
    """The last five years of daily returns for the calculator, as integers (null = no price)."""
    from var.data import ETFS, SYMBOLS

    symbols = ETFS + sorted(SYMBOLS)
    recent = r[symbols].iloc[-CALC_DAYS:]
    data = {s: [None if pd.isna(x) else int(round(x / SCALE)) for x in recent[s]]
            for s in symbols}
    return {"symbols": symbols, "labels": LABELS, "last": f"{recent.index[-1]:%d %b %Y}",
            "r": data, "stress": stress_data(r, symbols)}


def stress_data(r: pd.DataFrame, symbols: list[str]) -> list[dict]:
    """Each scenario's return for every symbol (proxied by NIFTY beta where it had no prices)."""
    from var.decomp import SCENARIOS, stress

    out = []
    for name, (start, end) in SCENARIOS.items():
        s = stress(r, dict.fromkeys(symbols, 1.0), start, end)
        out.append({"name": name, "start": start, "end": end,
                    "days": int(((r.index > start) & (r.index <= end)).sum()),
                    "r": {k: int(round(s.loc[k, "return"] / SCALE)) for k in symbols},
                    "proxied": [k for k in symbols if s.loc[k, "source"] != "history"]})
    return out


def calculator_section(r: pd.DataFrame) -> str:
    return (CALCULATOR.replace("$js", JS)
            .replace("$data", json.dumps(calculator_data(r), separators=(",", ":"))))


def methods_figure(r: pd.DataFrame) -> go.Figure:
    """99% one-day VaR and expected shortfall by method for three ₹10 lakh portfolios."""
    from var.reports import CAPITAL, PORTFOLIOS, START, method_table

    r = r[r.index >= START]
    names = ["historical", "normal", "Monte Carlo, Student-t"]
    colors = [plots.TEXT, plots.SERIES[0], plots.SERIES[1]]
    tables = {p: method_table(r, h) for p, h in PORTFOLIOS.items()}
    fig = go.Figure()
    for m, c in zip(names, colors, strict=True):
        fig.add_bar(x=list(PORTFOLIOS), y=[round(tables[p][m][0] * CAPITAL) for p in PORTFOLIOS],
                    name=f"VaR, {m}", marker_color=c,
                    hovertemplate=f"{m}: 99%% VaR ₹%{{y:,.0f}}<extra></extra>")
    fig.update_layout(barmode="group")
    return _scatter_layout(fig, "", "99% one-day VaR on ₹10 lakh, ₹")


def traffic_figure(r: pd.DataFrame) -> go.Figure:
    """Breaches of rolling 99% VaR in the last 250 days for the NIFTY ETF, with Basel zones."""
    from var.reports import START, backtest_table

    p = r.loc[r.index >= START, "NIFTYBEES"].dropna()
    res = backtest_table(p, {"historical, 250 days": ("historical", 250),
                             "historical, 1000 days": ("historical", 1000)})
    fig = go.Figure()
    for k, c in (("historical, 250 days", plots.SERIES[0]),
                 ("historical, 1000 days", plots.SERIES[1])):
        roll = res[k]["roll"]
        fig.add_scatter(x=list(roll.index.strftime("%Y-%m-%d")), y=roll.astype(int).tolist(),
                        name=k, mode="lines", line=dict(color=c, width=2, shape="hv"),
                        hovertemplate=f"{k}: %{{y}} breaches in 250 days<extra></extra>")
    for lo, hi, color in ((-0.5, 4.5, plots.SERIES[2]), (4.5, 9.5, plots.SERIES[3]),
                          (9.5, 20, plots.SERIES[7])):
        fig.add_hrect(y0=lo, y1=hi, fillcolor=color, opacity=0.08, line_width=0)
    fig = _layout(fig, "breaches of 99% VaR in the last 250 days")
    fig.update_xaxes(range=None, autorange=True)
    return fig


def options_figure(r: pd.DataFrame) -> go.Figure:
    """99% 10-day VaR of three NIFTY option books by method (priced at the last close)."""
    from var.data import DATA_PATH
    from var.options import BOOKS, book_pnl
    from var.reports import OPTION_METHODS, option_scenarios

    t = pd.read_parquet(DATA_PATH)["close"][["NIFTY", "INDIAVIX"]].dropna()
    S, vol = float(t["NIFTY"].iloc[-1]), float(t["INDIAVIX"].iloc[-1]) / 100
    ret, vr = option_scenarios(10)
    colors = [plots.SERIES[2], plots.SERIES[0], plots.SERIES[3], plots.TEXT]
    fig = go.Figure()
    for m, c in zip(OPTION_METHODS, colors, strict=True):
        y = [round(-float(np.quantile(book_pnl(make(S), S, vol, ret.to_numpy(), vr.to_numpy(),
                                                14 / 365, m), 0.01)))
             for make in BOOKS.values()]
        fig.add_bar(x=list(BOOKS), y=y, name=m, marker_color=c,
                    hovertemplate=f"{m}: 99%% 10-day VaR ₹%{{y:,.0f}}<extra></extra>")
    fig.update_layout(barmode="group")
    return _scatter_layout(fig, "", "99% 10-day VaR on ₹10 lakh of notional, ₹")


SECTIONS = [
    ("VaR calculator",
     "Enter rupee amounts in any of today's NIFTY 50 stocks or the NIFTY and gold ETFs. The page "
     "replays the chosen window of past daily returns on today's holdings: the historical loss "
     "exceeded on only 1% of days is the 99% VaR, and the average loss on those days the expected "
     "shortfall. Filtered historical rescales those days to today's volatility (the method that "
     "passed the backtests below); the normal method assumes a bell curve; Cornish-Fisher corrects "
     "it for skew and fat tails when they aren't too extreme. Below the totals: how much of the "
     "expected shortfall each holding carries, and what four of history's worst episodes would "
     "cost these holdings today.",
     calculator_section),
    ("Methods compared",
     "Three ₹10 lakh portfolios, daily returns since January 2009. A normal curve puts the 99% "
     "loss lower than history does; Monte Carlo with fat-tailed (Student-t) draws comes close to "
     "history.",
     methods_figure),
    ("Backtesting",
     "Each day's 99% VaR forecast from past data only, for the NIFTY ETF, counted against what "
     "happened. Basel's traffic light: up to 4 breaches in 250 days is green, 5 to 9 yellow, 10 or "
     "more red. The 1000-day VaR breached about as often as promised overall, but its breaches "
     "came in bunches and it went deepest into the red in 2020.",
     traffic_figure),
    ("Options book",
     "Three books of one-month NIFTY options on ₹10 lakh of notional, priced at the last close "
     "with India VIX as the volatility, replayed through every 10-day NIFTY and VIX move since "
     "2009. "
     "Greeks approximations (delta; delta-gamma; plus vega) against repricing every option: "
     "delta-gamma understates the short put's VaR and overstates the short straddle's.",
     options_figure),
]


REPO = "https://github.com/Dhirajdas18/value-at-risk"
PAGE = Template((Path(__file__).with_name("page.html")).read_text(encoding="utf-8"))


def _slug(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")


def build_html(data: pd.DataFrame) -> str:
    parts, nav, loaded = [], [], False
    for title, blurb, make in SECTIONS:
        out = make(data)
        if isinstance(out, str):
            body = out
        else:
            body = out.to_html(full_html=False, include_plotlyjs=False if loaded else "cdn",
                               config={"displaylogo": False, "responsive": True})
            loaded = True
        sid = _slug(title)
        nav.append(f'<li><a href="#{sid}">{title}</a></li>')
        parts.append(f'<section id="{sid}">\n<h2>{title}</h2>\n<p>{blurb}</p>\n{body}\n</section>')
    if not loaded:  # the page script expects Plotly
        parts.append('<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>')
    return PAGE.substitute(
        icon=ICON, font=FONT, repo=REPO, updated=f"{data.index[-1]:%d %b %Y}",
        surface=plots.SURFACE, text=plots.TEXT, text2=plots.TEXT_2, grid=plots.GRID,
        accent=plots.SERIES[0], d_surface=plots.DARK[plots.SURFACE], d_text=plots.DARK[plots.TEXT],
        d_text2=plots.DARK[plots.TEXT_2], d_grid=plots.DARK[plots.GRID],
        d_accent=plots.DARK[plots.SERIES[0]],
        dark=json.dumps({k.lower(): v for k, v in plots.DARK.items()}),
        dark_lines=json.dumps({k.lower(): v for k, v in plots.DARK_LINES.items()}),
        nav="\n".join(nav), sections="\n".join(parts),
    )
