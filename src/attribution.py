"""Executable Black-Scholes identities and hedged-P&L attribution.

Two levels of attribution are shown. A raw regression against realised variance
is intuitive but coarse because option convexity (gamma) changes through time.
The stronger experiment uses the Black-Scholes PDE directly: for a short option
hedged at implied volatility, terminal P&L is approximated by the accumulated
``0.5 * Gamma * S^2 * (sigma_implied^2 dt - realised quadratic variation)``.
As the hedge grid is refined, the discrete self-financing P&L converges to this
gamma-weighted variance attribution path by path.
"""
import json
from math import exp, log, sqrt
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

from black_scholes import Option, delta, gamma, price, theta

ROOT = Path(__file__).resolve().parents[1]


def pde_residual(o: Option) -> float:
    """theta + 0.5 sigma^2 S^2 gamma + (r - q) S delta - r V."""
    return (theta(o) + 0.5 * o.sigma ** 2 * o.S ** 2 * gamma(o)
            + (o.r - o.q) * o.S * delta(o) - o.r * price(o))


def breakeven_daily_move(o: Option) -> float:
    """The one-standard-deviation Black-Scholes move over one trading day."""
    return o.S * o.sigma * sqrt(1 / 252)


def pnl_and_realized_var(n_paths=4000, n_steps=128, S0=100.0, K=100.0,
                         T=1.0, r=0.02, sigma=0.2, seed=9):
    """Fair-vol hedged short-call P&L and annualised path realised variance."""
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    S = np.full(n_paths, S0)
    d0 = delta(Option(S0, K, T, r, sigma))
    cash = price(Option(S0, K, T, r, sigma)) - d0 * S
    held = np.full(n_paths, d0)
    rv = np.zeros(n_paths)
    for step in range(1, n_steps + 1):
        z = rng.standard_normal(n_paths)
        dlog = (r - 0.5 * sigma ** 2) * dt + sigma * sqrt(dt) * z
        S = S * np.exp(dlog)
        rv += dlog ** 2
        cash *= exp(r * dt)
        ttm = T - step * dt
        if ttm > 1e-12:
            d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * ttm) \
                / (sigma * sqrt(ttm))
            new_delta = norm.cdf(d1)
            cash -= (new_delta - held) * S
            held = new_delta
    pnl = cash + held * S - np.maximum(S - K, 0.0)
    return pnl, rv / T


def gamma_weighted_attribution(n_paths=5000, n_steps=128, S0=100.0,
                               K=100.0, T=1.0, r=0.02, mu=0.02,
                               sigma_real=0.20, sigma_implied=0.25, seed=3):
    """Compare discrete hedge P&L with the PDE variance attribution.

    The option is sold and hedged using ``sigma_implied`` while spot is
    generated with ``sigma_real``. Each local variance contribution is grown at
    the risk-free rate to expiry so it is on the same terminal-value basis as
    the self-financing hedge P&L.
    """
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    S = np.full(n_paths, S0)
    initial = Option(S0, K, T, r, sigma_implied)
    d0 = delta(initial)
    cash = price(initial) - d0 * S
    held = np.full(n_paths, d0)
    predicted = np.zeros(n_paths)

    for step in range(1, n_steps + 1):
        t_prev = (step - 1) * dt
        ttm_prev = T - t_prev
        d1 = (np.log(S / K) + (r + 0.5 * sigma_implied ** 2) * ttm_prev) \
            / (sigma_implied * sqrt(ttm_prev))
        gam = norm.pdf(d1) / (S * sigma_implied * sqrt(ttm_prev))

        z = rng.standard_normal(n_paths)
        dlog = (mu - 0.5 * sigma_real ** 2) * dt \
            + sigma_real * sqrt(dt) * z
        predicted += (0.5 * gam * S ** 2
                      * (sigma_implied ** 2 * dt - dlog ** 2)
                      * exp(r * (T - t_prev)))

        S = S * np.exp(dlog)
        cash *= exp(r * dt)
        ttm = T - step * dt
        if ttm > 1e-12:
            d1_next = (np.log(S / K) + (r + 0.5 * sigma_implied ** 2) * ttm) \
                / (sigma_implied * sqrt(ttm))
            new_delta = norm.cdf(d1_next)
            cash -= (new_delta - held) * S
            held = new_delta

    pnl = cash + held * S - np.maximum(S - K, 0.0)
    return pnl, predicted


def attribution_convergence(steps=(16, 32, 64, 128, 256, 512),
                            n_paths=4000, seed=3):
    rows = []
    for n in steps:
        pnl, pred = gamma_weighted_attribution(
            n_paths=n_paths, n_steps=n, seed=seed)
        rows.append({
            "steps": int(n),
            "corr": float(np.corrcoef(pnl, pred)[0, 1]),
            "rmse": float(np.sqrt(np.mean((pnl - pred) ** 2))),
            "mean_pnl": float(pnl.mean()),
            "mean_attribution": float(pred.mean()),
        })
    return rows


def main():
    figures = ROOT / "reports" / "figures"
    figures.mkdir(parents=True, exist_ok=True)

    residuals = [abs(pde_residual(Option(S, K, T, 0.04, sig, call, q=q)))
                 for S in (80, 100, 130) for K in (90, 110)
                 for T in (0.1, 1.0) for sig in (0.15, 0.4)
                 for call in (True, False) for q in (0.0, 0.03)]

    pnl, rv = pnl_and_realized_var()
    slope, intercept = np.polyfit(rv, pnl, 1)
    corr = float(np.corrcoef(rv, pnl)[0, 1])
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    ax.plot(rv, pnl, ".", ms=2.5, alpha=0.35, color="#0b3d91")
    grid = np.linspace(rv.min(), rv.max(), 50)
    ax.plot(grid, slope * grid + intercept, lw=2, color="#c1440e",
            label=f"fit, corr {corr:.2f}")
    ax.axvline(0.2 ** 2, color="grey", ls="--", lw=1.2,
               label="implied variance sold")
    ax.set(xlabel="realised variance on the path",
           ylabel="hedged short call P&L",
           title="Raw realised variance explains much, but ignores gamma weighting")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(figures / "pnl_vs_realized_var.png", dpi=150)
    plt.close(fig)

    actual, predicted = gamma_weighted_attribution()
    attr_corr = float(np.corrcoef(actual, predicted)[0, 1])
    attr_rmse = float(np.sqrt(np.mean((actual - predicted) ** 2)))
    fig2, ax2 = plt.subplots(figsize=(6.2, 5.0))
    ax2.scatter(predicted, actual, s=7, alpha=0.25, color="#0b3d91")
    lo = min(predicted.min(), actual.min())
    hi = max(predicted.max(), actual.max())
    ax2.plot([lo, hi], [lo, hi], "--", color="#c1440e", lw=1.4,
             label="perfect attribution")
    ax2.set(xlabel="gamma-weighted variance attribution",
            ylabel="actual terminal hedge P&L",
            title=f"PDE attribution tracks pathwise P&L (corr {attr_corr:.3f})")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.25)
    fig2.tight_layout()
    fig2.savefig(figures / "gamma_variance_attribution.png", dpi=150)
    plt.close(fig2)

    convergence = attribution_convergence()
    fig3, ax3 = plt.subplots(figsize=(7.2, 4.2))
    ax3.loglog([r["steps"] for r in convergence],
               [r["rmse"] for r in convergence], "o-", color="#0b3d91")
    ax3.set(xlabel="hedge intervals", ylabel="attribution RMSE",
            title="Discrete hedge P&L converges to the gamma-weighted variance identity")
    ax3.grid(alpha=0.3, which="both")
    fig3.tight_layout()
    fig3.savefig(figures / "gamma_attribution_convergence.png", dpi=150)
    plt.close(fig3)

    o = Option(100, 100, 0.25, 0.0, 0.2, q=0.0)
    daily_theta = theta(o) / 252
    gamma_gain = 0.5 * gamma(o) * breakeven_daily_move(o) ** 2
    summary = {
        "max_abs_pde_residual": float(np.max(residuals)),
        "raw_pnl_vs_realized_var": {
            "corr": round(corr, 3), "slope": round(float(slope), 2)},
        "gamma_weighted_attribution_128": {
            "corr": round(attr_corr, 4),
            "rmse": round(attr_rmse, 4),
            "mean_actual_pnl": round(float(actual.mean()), 4),
            "mean_attributed_pnl": round(float(predicted.mean()), 4),
        },
        "gamma_attribution_convergence": [
            {k: round(v, 4) if isinstance(v, float) else v for k, v in row.items()}
            for row in convergence
        ],
        "breakeven": {"daily_move": round(breakeven_daily_move(o), 3),
                      "daily_theta": round(daily_theta, 5),
                      "gamma_gain_at_breakeven": round(gamma_gain, 5)},
    }
    (ROOT / "reports" / "attribution_summary.json").write_text(
        json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
