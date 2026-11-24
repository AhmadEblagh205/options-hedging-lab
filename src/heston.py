"""Heston stochastic volatility: the model that prices a smile instead
of explaining one. Variance follows a mean reverting square root
process correlated with the spot, the call price comes from the
characteristic function through two probability integrals (the standard
semi closed form, written in the numerically stable branch), and two
checks keep it honest: as vol of vol goes to zero the price collapses
to Black-Scholes, and a full truncation Euler Monte Carlo agrees with
the integral to within its standard error.

Usage:
    python src/heston.py
"""
import json
from math import exp, log, pi, sqrt
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

from black_scholes import Option, price as bs_price
from implied_vol import implied_vol

ROOT = Path(__file__).resolve().parents[1]


def _validate_heston(v0, kappa, theta, xi, rho):
    if min(v0, kappa, theta, xi) <= 0:
        raise ValueError("v0, kappa, theta and xi must be positive")
    if not -1.0 <= rho <= 1.0:
        raise ValueError("rho must lie in [-1, 1]")


def _cf(u, S, T, r, q, v0, kappa, theta, xi, rho, j):
    """Characteristic function of log spot, Albrecher et al stable form."""
    a = kappa * theta
    b = kappa - rho * xi if j == 1 else kappa
    u_j = 0.5 if j == 1 else -0.5
    d = np.sqrt((rho * xi * 1j * u - b) ** 2
                - xi ** 2 * (2 * u_j * 1j * u - u ** 2))
    g = (b - rho * xi * 1j * u - d) / (b - rho * xi * 1j * u + d)
    exp_dT = np.exp(-d * T)
    C = (r - q) * 1j * u * T + a / xi ** 2 * (
        (b - rho * xi * 1j * u - d) * T
        - 2 * np.log((1 - g * exp_dT) / (1 - g)))
    D = (b - rho * xi * 1j * u - d) / xi ** 2 \
        * (1 - exp_dT) / (1 - g * exp_dT)
    return np.exp(C + D * v0 + 1j * u * log(S))


def heston_call(S, K, T, r, v0, kappa, theta, xi, rho, q=0.0) -> float:
    _validate_heston(v0, kappa, theta, xi, rho)
    if min(S, K, T) <= 0:
        raise ValueError("S, K and T must be positive")
    def integrand(u, j):
        val = np.exp(-1j * u * log(K)) \
            * _cf(u, S, T, r, q, v0, kappa, theta, xi, rho, j) / (1j * u)
        return val.real

    P = []
    for j in (1, 2):
        integral, _ = quad(integrand, 1e-8, 200, args=(j,), limit=400)
        P.append(0.5 + integral / pi)
    return float(S * exp(-q * T) * P[0] - K * exp(-r * T) * P[1])


def heston_mc(S, K, T, r, v0, kappa, theta, xi, rho, n_paths=60_000,
              n_steps=252, seed=0, q=0.0):
    """Full truncation Euler: variance floored at zero inside the
    drift and diffusion, the scheme's standard bias fix."""
    _validate_heston(v0, kappa, theta, xi, rho)
    if min(S, K, T) <= 0 or n_paths <= 1 or n_steps <= 0:
        raise ValueError("positive S/K/T/steps and n_paths > 1 are required")
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    logS = np.full(n_paths, log(S))
    v = np.full(n_paths, v0)
    for _ in range(n_steps):
        z1 = rng.standard_normal(n_paths)
        z2 = rho * z1 + sqrt(1 - rho ** 2) * rng.standard_normal(n_paths)
        v_plus = np.maximum(v, 0.0)
        logS += (r - q - 0.5 * v_plus) * dt + np.sqrt(v_plus * dt) * z1
        v += kappa * (theta - v_plus) * dt + xi * np.sqrt(v_plus * dt) * z2
    payoff = exp(-r * T) * np.maximum(np.exp(logS) - K, 0.0)
    return float(payoff.mean()), float(payoff.std() / sqrt(n_paths))


def smile_figure(path):
    S, T, r = 100.0, 0.5, 0.02
    base = dict(v0=0.04, kappa=2.0, theta=0.04, xi=0.6)
    strikes = np.arange(75, 126, 5)
    fig, ax = plt.subplots(figsize=(7.8, 4.4))
    for rho in (-0.7, 0.0, 0.7):
        ivs = []
        for K in strikes:
            p = heston_call(S, K, T, r, rho=rho, **base)
            ivs.append(100 * implied_vol(p, S, K, T, r))
        ax.plot(strikes, ivs, "o-", lw=1.8, label=f"rho = {rho:+.1f}")
    ax.set(xlabel="strike", ylabel="implied vol (%)",
           title="Heston prices the smile; correlation sets the skew")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    S, K, T, r = 100.0, 100.0, 1.0, 0.02
    params = dict(v0=0.04, kappa=2.0, theta=0.04, xi=0.6, rho=-0.7)
    semi = heston_call(S, K, T, r, **params)
    mc, se = heston_mc(S, K, T, r, **params)
    bs_limit = heston_call(S, K, T, r, v0=0.04, kappa=2.0, theta=0.04,
                           xi=1e-4, rho=0.0)
    bs = bs_price(Option(S, K, T, r, sqrt(0.04)))
    smile_figure(ROOT / "reports" / "figures" / "heston_smile.png")
    summary = {
        "semi_closed_form": round(semi, 4),
        "monte_carlo": {"price": round(mc, 4), "se": round(se, 4)},
        "bs_limit_check": {"heston_xi_to_0": round(bs_limit, 4),
                           "black_scholes": round(bs, 4)},
        "feller_ratio_2kappa_theta_over_xi2": round(
            2 * params["kappa"] * params["theta"] / params["xi"] ** 2, 3),
        "feller_condition_satisfied": bool(
            2 * params["kappa"] * params["theta"] >= params["xi"] ** 2),
    }
    (ROOT / "reports" / "heston_summary.json").write_text(
        json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
