"""Path-dependent payoffs with explicit Monte Carlo bias/variance controls.

The arithmetic Asian option uses a geometric-Asian control variate. The barrier
example distinguishes two different objects: a discretely monitored barrier and
a continuously monitored barrier. A Brownian-bridge survival correction uses
the conditional crossing probability between simulated endpoints, which removes
most of the monitoring-grid bias without brute-force time stepping.
"""
import json
from math import exp, log, sqrt
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

from black_scholes import Option

ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "reports" / "figures"


def gbm_paths(o: Option, n_paths: int, n_steps: int, rng) -> np.ndarray:
    if n_paths <= 1 or n_steps <= 0:
        raise ValueError("n_paths must exceed one and n_steps must be positive")
    dt = o.T / n_steps
    z = rng.standard_normal((n_paths, n_steps))
    increments = (o.r - o.q - 0.5 * o.sigma ** 2) * dt \
        + o.sigma * sqrt(dt) * z
    return o.S * np.exp(np.cumsum(increments, axis=1))


def geometric_asian_price(o: Option, n_dates: int) -> float:
    """Closed form for the discrete geometric-average call."""
    if n_dates <= 0:
        raise ValueError("n_dates must be positive")
    dt = o.T / n_dates
    t_bar = dt * (n_dates + 1) / 2
    var = o.sigma ** 2 * dt * (n_dates + 1) * (2 * n_dates + 1) \
        / (6 * n_dates)
    mu = (o.r - o.q - 0.5 * o.sigma ** 2) * t_bar + 0.5 * var
    d1 = (log(o.S / o.K) + mu + 0.5 * var) / sqrt(var)
    d2 = d1 - sqrt(var)
    fwd = o.S * exp(mu)
    return exp(-o.r * o.T) * (fwd * norm.cdf(d1) - o.K * norm.cdf(d2))


def asian_call_mc(o: Option, n_dates: int = 50, n_paths: int = 40_000,
                  seed: int = 0, control_variate: bool = True):
    rng = np.random.default_rng(seed)
    paths = gbm_paths(o, n_paths, n_dates, rng)
    disc = exp(-o.r * o.T)
    arith = disc * np.maximum(paths.mean(axis=1) - o.K, 0.0)
    if not control_variate:
        return float(arith.mean()), float(arith.std(ddof=1) / sqrt(n_paths))
    geo = disc * np.maximum(np.exp(np.log(paths).mean(axis=1)) - o.K, 0.0)
    known = geometric_asian_price(o, n_dates)
    beta = np.cov(arith, geo, ddof=1)[0, 1] / np.var(geo, ddof=1)
    y = arith - beta * (geo - known)
    return float(y.mean()), float(y.std(ddof=1) / sqrt(n_paths))


def _validate_down_and_out(o: Option, barrier: float):
    if not o.call:
        raise ValueError("implemented closed form is for down-and-out calls")
    if not (0 < barrier < min(o.S, o.K)):
        raise ValueError("require 0 < barrier < min(spot, strike)")


def down_and_out_closed(o: Option, barrier: float) -> float:
    """Reiner-Rubinstein continuous down-and-out call, no rebate.

    This implementation is intentionally restricted to ``barrier < strike``
    and ``barrier < spot``; outside that region a different formula branch is
    required, so silently reusing this expression would be misleading.
    """
    _validate_down_and_out(o, barrier)
    lam = (o.r - o.q + 0.5 * o.sigma ** 2) / o.sigma ** 2
    x1 = log(o.S / o.K) / (o.sigma * sqrt(o.T)) + lam * o.sigma * sqrt(o.T)
    y1 = log(barrier ** 2 / (o.S * o.K)) / (o.sigma * sqrt(o.T)) \
        + lam * o.sigma * sqrt(o.T)
    dq, dr = exp(-o.q * o.T), exp(-o.r * o.T)
    vanilla_part = o.S * dq * norm.cdf(x1) \
        - o.K * dr * norm.cdf(x1 - o.sigma * sqrt(o.T))
    knock = o.S * dq * (barrier / o.S) ** (2 * lam) * norm.cdf(y1) \
        - o.K * dr * (barrier / o.S) ** (2 * lam - 2) \
        * norm.cdf(y1 - o.sigma * sqrt(o.T))
    return vanilla_part - knock


def down_and_out_mc(o: Option, barrier: float, n_steps: int,
                    n_paths: int = 60_000, seed: int = 1):
    """Price a *discretely monitored* down-and-out call."""
    _validate_down_and_out(o, barrier)
    rng = np.random.default_rng(seed)
    paths = gbm_paths(o, n_paths, n_steps, rng)
    alive = paths.min(axis=1) > barrier
    payoff = exp(-o.r * o.T) * np.maximum(paths[:, -1] - o.K, 0.0) * alive
    return float(payoff.mean()), float(payoff.std(ddof=1) / sqrt(n_paths))


def down_and_out_mc_bridge(o: Option, barrier: float, n_steps: int,
                           n_paths: int = 60_000, seed: int = 1):
    """Continuously monitored barrier estimate via Brownian-bridge weighting.

    Conditional on two log-price endpoints above the barrier, the probability
    of crossing a constant barrier between them is known in closed form. The
    product of interval survival probabilities is therefore a Rao-Blackwellised
    continuous-monitoring correction, not another fine-grid approximation.
    """
    _validate_down_and_out(o, barrier)
    if n_steps <= 0:
        raise ValueError("n_steps must be positive")
    rng = np.random.default_rng(seed)
    dt = o.T / n_steps
    log_barrier = log(barrier)
    x = np.full(n_paths, log(o.S))
    survival = np.ones(n_paths)

    for _ in range(n_steps):
        z = rng.standard_normal(n_paths)
        x_next = x + (o.r - o.q - 0.5 * o.sigma ** 2) * dt \
            + o.sigma * sqrt(dt) * z
        endpoints_above = (x > log_barrier) & (x_next > log_barrier)
        survival[~endpoints_above] = 0.0
        idx = np.flatnonzero(endpoints_above)
        if idx.size:
            hit_prob = np.exp(
                -2.0 * (x[idx] - log_barrier) * (x_next[idx] - log_barrier)
                / (o.sigma ** 2 * dt)
            )
            survival[idx] *= 1.0 - hit_prob
        x = x_next

    ST = np.exp(x)
    payoff = exp(-o.r * o.T) * np.maximum(ST - o.K, 0.0) * survival
    return float(payoff.mean()), float(payoff.std(ddof=1) / sqrt(n_paths))


def main():
    o = Option(S=100, K=100, T=1.0, r=0.03, sigma=0.25)
    plain = asian_call_mc(o, control_variate=False)
    cv = asian_call_mc(o, control_variate=True)

    barrier = 85.0
    closed = down_and_out_closed(o, barrier)
    monitor = [4, 12, 52, 252]
    discrete = [down_and_out_mc(o, barrier, n) for n in monitor]
    bridge = [down_and_out_mc_bridge(o, barrier, n) for n in monitor]

    FIGURES.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    ax.semilogx(monitor, [x[0] for x in discrete], "o-", lw=1.8,
                color="#0b3d91", label="discrete monitoring")
    ax.semilogx(monitor, [x[0] for x in bridge], "s-", lw=1.8,
                color="#2f7d32", label="Brownian-bridge continuous correction")
    ax.axhline(closed, color="#c1440e", ls="--", lw=1.4,
               label=f"continuous closed form {closed:.3f}")
    ax.set(xlabel="simulated time intervals per year",
           ylabel="down-and-out call price",
           title="Brownian bridge removes most barrier monitoring bias")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(FIGURES / "barrier_bridge.png", dpi=150)
    # Keep legacy filename because the portfolio currently references it.
    fig.savefig(FIGURES / "barrier_bias.png", dpi=150)
    plt.close(fig)

    summary = {
        "asian_plain": {"price": round(plain[0], 4), "se": round(plain[1], 5)},
        "asian_control_variate": {"price": round(cv[0], 4),
                                  "se": round(cv[1], 6)},
        "se_reduction_factor": round(plain[1] / cv[1], 1),
        "barrier": {
            "continuous_closed_form": round(closed, 4),
            "discrete_mc_by_intervals": {
                str(n): {"price": round(v[0], 4), "se": round(v[1], 4)}
                for n, v in zip(monitor, discrete)
            },
            "brownian_bridge_by_intervals": {
                str(n): {"price": round(v[0], 4), "se": round(v[1], 4)}
                for n, v in zip(monitor, bridge)
            },
        },
    }
    (ROOT / "reports" / "exotics_summary.json").write_text(
        json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
