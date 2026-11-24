"""Implied volatility inversion and a controlled smile demonstration.

The solver deliberately uses a bracketed method. It checks no-arbitrage bounds
before inversion, expands the volatility bracket when needed, and fails loudly
for impossible prices instead of returning an endpoint that looks like a
meaningful implied volatility.
"""
from math import exp

import numpy as np

from black_scholes import Option, price


def no_arbitrage_bounds(S, K, T, r, call=True, q=0.0):
    """European option price bounds under continuous carry/dividends."""
    spot_pv = S * exp(-q * T)
    strike_pv = K * exp(-r * T)
    if call:
        return max(0.0, spot_pv - strike_pv), spot_pv
    return max(0.0, strike_pv - spot_pv), strike_pv


def implied_vol(target: float, S, K, T, r, call=True, q=0.0,
                lo: float = 1e-8, hi: float = 1.0,
                tol: float = 1e-8, max_vol: float = 10.0) -> float:
    """Invert Black-Scholes by safeguarded bisection.

    Prices outside the European no-arbitrage interval raise ``ValueError``.
    The upper volatility bracket is expanded until it contains the target or
    ``max_vol`` is reached.
    """
    lower, upper = no_arbitrage_bounds(S, K, T, r, call=call, q=q)
    scale = max(1.0, abs(upper))
    eps = 1e-12 * scale
    if target < lower - eps or target > upper + eps:
        raise ValueError(
            f"target price {target:.8g} violates no-arbitrage bounds "
            f"[{lower:.8g}, {upper:.8g}]"
        )
    # Boundary prices correspond to limiting volatilities; returning an
    # arbitrary finite number would be misleading.
    if target <= lower + eps:
        return 0.0
    if target >= upper - eps:
        raise ValueError("target is at the infinite-volatility upper bound")

    def f(sig):
        return price(Option(S, K, T, r, sig, call=call, q=q)) - target

    hi_now = max(hi, lo * 2)
    while f(hi_now) < 0 and hi_now < max_vol:
        hi_now = min(max_vol, hi_now * 2)
    if f(hi_now) < 0:
        raise ValueError("could not bracket implied volatility below max_vol")

    lo_now = lo
    for _ in range(250):
        mid = 0.5 * (lo_now + hi_now)
        if f(mid) > 0:
            hi_now = mid
        else:
            lo_now = mid
        if hi_now - lo_now < tol:
            break
    return 0.5 * (lo_now + hi_now)


def mixture_price(S, K, T, r, sigmas=(0.15, 0.45), weights=(0.7, 0.3),
                  call=True, q=0.0) -> float:
    """Two-regime mixture used only to create a known non-BS smile."""
    if len(sigmas) != len(weights) or not np.isclose(sum(weights), 1.0):
        raise ValueError("sigmas and weights must align and weights must sum to 1")
    return float(sum(w * price(Option(S, K, T, r, s, call=call, q=q))
                     for s, w in zip(sigmas, weights)))


def smile(S=100.0, T=0.5, r=0.02, strikes=None, q=0.0):
    strikes = strikes if strikes is not None else np.arange(70, 131, 5)
    out = []
    for K in strikes:
        p = mixture_price(S, K, T, r, q=q)
        out.append((float(K), implied_vol(p, S, K, T, r, q=q)))
    return out
