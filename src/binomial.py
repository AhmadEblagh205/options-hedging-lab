"""Cox-Ross-Rubinstein trees: European prices that must converge to
Black-Scholes, and the thing the closed form cannot do, American early
exercise, priced by backward induction with the exercise decision taken
at every node."""
from math import exp, sqrt

import numpy as np

from black_scholes import Option


def crr_price(o: Option, n_steps: int = 500,
              american: bool = False) -> float:
    if n_steps <= 0:
        raise ValueError("n_steps must be positive")
    dt = o.T / n_steps
    u = exp(o.sigma * sqrt(dt))
    d = 1 / u
    disc = exp(-o.r * dt)
    q = (exp((o.r - o.q) * dt) - d) / (u - d)
    if not 0.0 <= q <= 1.0:
        raise ValueError("CRR risk-neutral probability is outside [0, 1]; increase n_steps")

    j = np.arange(n_steps + 1)
    prices = o.S * u ** j * d ** (n_steps - j)
    payoff = (np.maximum(prices - o.K, 0.0) if o.call
              else np.maximum(o.K - prices, 0.0))
    values = payoff
    for step in range(n_steps - 1, -1, -1):
        values = disc * (q * values[1:] + (1 - q) * values[:-1])
        if american:
            j = np.arange(step + 1)
            prices = o.S * u ** j * d ** (step - j)
            exercise = (np.maximum(prices - o.K, 0.0) if o.call
                        else np.maximum(o.K - prices, 0.0))
            values = np.maximum(values, exercise)
    return float(values[0])


def early_exercise_premium(o: Option, n_steps: int = 500) -> float:
    return crr_price(o, n_steps, american=True) \
        - crr_price(o, n_steps, american=False)


def exercise_boundary(o, n_steps: int = 400):
    """The American put's critical spot at each time: the highest node
    where immediate exercise beats continuation, read straight off the
    tree. It rises toward the strike as expiry approaches, the shape
    every derivatives textbook draws and this one computes."""
    if o.call:
        raise ValueError("exercise_boundary is implemented for puts only")
    if n_steps <= 0:
        raise ValueError("n_steps must be positive")
    dt = o.T / n_steps
    u = exp(o.sigma * sqrt(dt))
    d = 1 / u
    disc = exp(-o.r * dt)
    p = (exp((o.r - o.q) * dt) - d) / (u - d)
    if not 0.0 <= p <= 1.0:
        raise ValueError("CRR risk-neutral probability is outside [0, 1]; increase n_steps")
    j = np.arange(n_steps + 1)
    prices = o.S * u ** j * d ** (n_steps - j)
    values = np.maximum(o.K - prices, 0.0)
    boundary = []
    for step in range(n_steps - 1, -1, -1):
        values = disc * (p * values[1:] + (1 - p) * values[:-1])
        j = np.arange(step + 1)
        prices = o.S * u ** j * d ** (step - j)
        exercise = np.maximum(o.K - prices, 0.0)
        exercised = exercise >= values + 1e-12
        values = np.maximum(values, exercise)
        if exercised.any():
            boundary.append((step * dt, float(prices[exercised].max())))
    return sorted(boundary)
