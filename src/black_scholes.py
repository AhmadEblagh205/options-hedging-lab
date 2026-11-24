"""Black-Scholes closed forms: prices, Greeks, and put-call parity.

The closed forms are the ground truth used to validate the numerical layers of
this repository. Inputs are validated explicitly because silent propagation of
non-positive spot, strike, expiry or volatility produces meaningless Greeks and
is especially dangerous inside calibration routines.
"""
from dataclasses import dataclass
from math import exp, isfinite, log, sqrt

from scipy.stats import norm


@dataclass(frozen=True)
class Option:
    S: float          # spot
    K: float          # strike
    T: float          # years to expiry
    r: float          # continuously compounded rate
    sigma: float      # volatility
    call: bool = True
    q: float = 0.0    # continuous dividend yield

    def __post_init__(self):
        vals = (self.S, self.K, self.T, self.r, self.sigma, self.q)
        if not all(isfinite(float(v)) for v in vals):
            raise ValueError("option inputs must be finite")
        if self.S <= 0 or self.K <= 0:
            raise ValueError("spot and strike must be strictly positive")
        if self.T <= 0:
            raise ValueError("time to expiry must be strictly positive")
        if self.sigma <= 0:
            raise ValueError("volatility must be strictly positive")


def _d1(o: Option) -> float:
    return (log(o.S / o.K) + (o.r - o.q + 0.5 * o.sigma ** 2) * o.T) \
        / (o.sigma * sqrt(o.T))


def _d2(o: Option) -> float:
    return _d1(o) - o.sigma * sqrt(o.T)


def price(o: Option) -> float:
    d1, d2 = _d1(o), _d2(o)
    dq, dr = exp(-o.q * o.T), exp(-o.r * o.T)
    if o.call:
        return o.S * dq * norm.cdf(d1) - o.K * dr * norm.cdf(d2)
    return o.K * dr * norm.cdf(-d2) - o.S * dq * norm.cdf(-d1)


def delta(o: Option) -> float:
    d1, dq = _d1(o), exp(-o.q * o.T)
    return dq * norm.cdf(d1) if o.call else dq * (norm.cdf(d1) - 1.0)


def gamma(o: Option) -> float:
    return exp(-o.q * o.T) * norm.pdf(_d1(o)) / (o.S * o.sigma * sqrt(o.T))


def vega(o: Option) -> float:
    return o.S * exp(-o.q * o.T) * norm.pdf(_d1(o)) * sqrt(o.T)


def theta(o: Option) -> float:
    d1, d2 = _d1(o), _d2(o)
    dq, dr = exp(-o.q * o.T), exp(-o.r * o.T)
    decay = -o.S * dq * norm.pdf(d1) * o.sigma / (2 * sqrt(o.T))
    if o.call:
        return decay - o.r * o.K * dr * norm.cdf(d2) \
            + o.q * o.S * dq * norm.cdf(d1)
    return decay + o.r * o.K * dr * norm.cdf(-d2) \
        - o.q * o.S * dq * norm.cdf(-d1)


def rho(o: Option) -> float:
    d2 = _d2(o)
    sign = 1.0 if o.call else -1.0
    return sign * o.K * o.T * exp(-o.r * o.T) * norm.cdf(sign * d2)


def parity_gap(S, K, T, r, sigma, q: float = 0.0) -> float:
    """C - P - (S e^{-qT} - K e^{-rT}), zero in an arbitrage-free world."""
    c = price(Option(S, K, T, r, sigma, call=True, q=q))
    p = price(Option(S, K, T, r, sigma, call=False, q=q))
    return c - p - (S * exp(-q * T) - K * exp(-r * T))
