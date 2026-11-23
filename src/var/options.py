"""VaR of a book of NIFTY options: approximate it with Greeks, or reprice every option.

A book is a list of legs (kind, strike, quantity): kind is "stock" (the index itself), "call" or
"put"; quantity is in index units, negative when sold. Each book holds ₹10 lakh of notional.
Scenario: the index moves by `ret`, implied volatility is multiplied by `vol_ratio` (India VIX's
move that day), and `dt` years pass.

Methods for the P&L of a scenario:
- delta: theta·dt + delta·dS (a straight line in the index move)
- delta-gamma: + ½·gamma·dS² (a parabola)
- delta-gamma-vega: + vega·dσ
- full: Black-Scholes reprice of every option after the move
"""

import numpy as np
from scipy.stats import norm

RATE = 0.065  # assumed risk-free rate, as in bsp
EXPIRY = 30 / 365  # options have a month left
NOTIONAL = 1_000_000.0


def _d1d2(S, K, T, r, vol):
    d1 = (np.log(S / K) + (r + vol**2 / 2) * T) / (vol * np.sqrt(T))
    return d1, d1 - vol * np.sqrt(T)


def bs_price(S, K, T, r, vol, kind):
    d1, d2 = _d1d2(S, K, T, r, vol)
    if kind == "call":
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def greeks(S, K, T, r, vol, kind) -> dict:
    """delta, gamma, vega (per 1.00 of vol) and theta (value change per year passing)."""
    d1, d2 = _d1d2(S, K, T, r, vol)
    gamma = norm.pdf(d1) / (S * vol * np.sqrt(T))
    vega = S * norm.pdf(d1) * np.sqrt(T)
    decay = -S * norm.pdf(d1) * vol / (2 * np.sqrt(T))
    if kind == "call":
        delta, theta = norm.cdf(d1), decay - r * K * np.exp(-r * T) * norm.cdf(d2)
    else:
        delta, theta = norm.cdf(d1) - 1, decay + r * K * np.exp(-r * T) * norm.cdf(-d2)
    return {"delta": delta, "gamma": gamma, "vega": vega, "theta": theta}


def _units(S):
    return NOTIONAL / S


BOOKS = {
    "protective put": lambda S: [("stock", S, _units(S)), ("put", 0.95 * S, _units(S))],
    "short 5% OTM put": lambda S: [("put", 0.95 * S, -_units(S))],
    "short straddle": lambda S: [("call", S, -_units(S)), ("put", S, -_units(S))],
}


def book_pnl(book, S, vol, ret, vol_ratio, dt, method="full", T=EXPIRY, r=RATE) -> np.ndarray:
    """₹ P&L of the book for each scenario (arrays `ret`, `vol_ratio`)."""
    ret, vol_ratio = np.asarray(ret, dtype=float), np.asarray(vol_ratio, dtype=float)
    dS, total = S * ret, np.zeros_like(ret)
    for kind, K, q in book:
        if kind == "stock":
            total += q * dS
            continue
        if method == "full":
            new = bs_price(S + dS, K, T - dt, r, vol * vol_ratio, kind)
            total += q * (new - bs_price(S, K, T, r, vol, kind))
            continue
        g = greeks(S, K, T, r, vol, kind)
        pnl = g["theta"] * dt + g["delta"] * dS
        if method in ("delta-gamma", "delta-gamma-vega"):
            pnl = pnl + 0.5 * g["gamma"] * dS**2
        if method == "delta-gamma-vega":
            pnl = pnl + g["vega"] * vol * (vol_ratio - 1)
        total += q * pnl
    return total
