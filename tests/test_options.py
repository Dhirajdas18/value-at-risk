import numpy as np
import pytest

from var.options import BOOKS, book_pnl, bs_price, greeks

S, T, R, VOL = 25_000.0, 30 / 365, 0.065, 0.16


def test_put_call_parity():
    c = bs_price(S, 24_000, T, R, VOL, "call")
    p = bs_price(S, 24_000, T, R, VOL, "put")
    assert c - p == pytest.approx(S - 24_000 * np.exp(-R * T), rel=1e-12)


@pytest.mark.parametrize("kind", ["call", "put"])
def test_greeks_match_finite_differences(kind):
    g = greeks(S, 25_500, T, R, VOL, kind)
    h = 1.0

    def f(s=S, t=T, v=VOL):
        return bs_price(s, 25_500, t, R, v, kind)

    assert g["delta"] == pytest.approx((f(S + h) - f(S - h)) / (2 * h), rel=1e-5)
    assert g["gamma"] == pytest.approx((f(S + h) - 2 * f() + f(S - h)) / h**2, rel=1e-3)
    assert g["vega"] == pytest.approx((f(v=VOL + 1e-4) - f(v=VOL - 1e-4)) / 2e-4, rel=1e-5)
    assert g["theta"] == pytest.approx((f(t=T - 1e-5) - f(t=T + 1e-5)) / 2e-5, rel=1e-4)


def test_stock_only_book_is_linear_in_every_method():
    book = [("stock", 1.0, 40.0)]
    for method in ("delta", "delta-gamma", "delta-gamma-vega", "full"):
        pnl = book_pnl(book, S, VOL, np.array([-0.1, 0.05]), np.array([1.3, 0.9]), 1 / 365, method)
        assert pnl == pytest.approx([-0.1 * S * 40, 0.05 * S * 40])


def test_approximations_converge_to_full_revaluation_for_small_moves():
    book = BOOKS["short straddle"](S)
    ret, vr, dt = np.array([0.002, -0.002]), np.array([1.0, 1.0]), 1 / 365
    full = book_pnl(book, S, VOL, ret, vr, dt, "full")
    dg = book_pnl(book, S, VOL, ret, vr, dt, "delta-gamma")
    assert dg == pytest.approx(full, abs=10)  # within ₹10 on ₹10 lakh of notional


def test_delta_gamma_errs_both_ways_on_a_crash_with_a_vol_spike():
    ret, vr = np.array([-0.13]), np.array([1.6])

    def pnl(book, method):
        return book_pnl(BOOKS[book](S), S, VOL, ret, vr, 1 / 365, method)[0]

    # short OTM put: gamma grows as the put nears the money, so the parabola understates the loss
    assert pnl("short 5% OTM put", "full") < pnl("short 5% OTM put", "delta-gamma") < 0
    # short straddle: deep in the money the payoff turns straight, so the parabola overstates it
    assert pnl("short straddle", "delta-gamma") < pnl("short straddle", "full") < 0
    assert pnl("short straddle", "delta") > 0  # delta alone even shows a gain


def test_books_hold_ten_lakh_of_notional():
    for make in BOOKS.values():
        legs = make(S)
        assert max(abs(q) for _, _, q in legs) * S == pytest.approx(1_000_000)
