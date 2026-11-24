"""Discrete delta hedging experiments with explicit robustness checks.

The simulator sells a European call, finances a delta hedge, and allows the
physical drift, realised volatility, hedge volatility and proportional trading
cost to be varied independently. The headline ``N^-1/2`` discretisation law is
not taken from one lucky seed: a multi-seed study records the distribution of
fitted log-log slopes.
"""
import json
from math import exp, sqrt
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

from black_scholes import Option, delta, price

ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "reports" / "figures"


def hedge_pnl(n_paths: int, n_steps: int, S0=100.0, K=100.0, T=1.0,
              r=0.02, mu=0.02, sigma_real=0.2, sigma_sell=0.2,
              sigma_hedge=None, seed=0, cost_bps: float = 0.0) -> np.ndarray:
    """Terminal P&L of a short call with discrete delta hedging.

    ``cost_bps`` is a proportional one-way trading cost charged on the initial
    hedge, each rebalance, and final liquidation. The centrepiece uses zero
    dividend yield; dividend effects are tested separately in the pricing/tree
    stack where the cash-flow accounting is explicit.
    """
    if n_paths <= 1 or n_steps <= 0:
        raise ValueError("n_paths must exceed one and n_steps must be positive")
    if min(sigma_real, sigma_sell) <= 0:
        raise ValueError("volatilities must be positive")
    sigma_hedge = sigma_sell if sigma_hedge is None else sigma_hedge
    if sigma_hedge <= 0:
        raise ValueError("hedge volatility must be positive")
    if cost_bps < 0:
        raise ValueError("trading cost cannot be negative")

    rng = np.random.default_rng(seed)
    dt = T / n_steps
    S = np.full(n_paths, S0)
    fee = cost_bps / 1e4
    d0 = delta(Option(S0, K, T, r, sigma_hedge))
    cash = price(Option(S0, K, T, r, sigma_sell)) - d0 * S \
        - fee * abs(d0) * S
    held = np.full(n_paths, d0)

    for step in range(1, n_steps + 1):
        z = rng.standard_normal(n_paths)
        S = S * np.exp((mu - 0.5 * sigma_real ** 2) * dt
                       + sigma_real * sqrt(dt) * z)
        cash = cash * exp(r * dt)
        ttm = T - step * dt
        if ttm > 1e-12:
            d1 = (np.log(S / K) + (r + 0.5 * sigma_hedge ** 2) * ttm) \
                / (sigma_hedge * sqrt(ttm))
            new_delta = norm.cdf(d1)
            trade = new_delta - held
            cash -= trade * S + fee * np.abs(trade) * S
            held = new_delta

    payoff = np.maximum(S - K, 0.0)
    # Liquidate the long stock hedge and pay the final one-way cost.
    return cash + held * S * (1 - fee) - payoff


def error_scaling(n_paths=3000, steps=(4, 8, 16, 32, 64, 128, 256, 512),
                  seed=7):
    stds = [float(hedge_pnl(n_paths, n, seed=seed).std(ddof=1)) for n in steps]
    slope = float(np.polyfit(np.log(steps), np.log(stds), 1)[0])
    return list(steps), stds, slope


def scaling_robustness(seeds=range(20), n_paths=2500,
                       steps=(8, 16, 32, 64, 128, 256)):
    slopes = []
    curves = []
    for seed in seeds:
        s, stds, slope = error_scaling(n_paths=n_paths, steps=steps, seed=seed)
        slopes.append(slope)
        curves.append(stds)
    curves = np.asarray(curves)
    return {
        "steps": list(steps),
        "slopes": slopes,
        "slope_mean": float(np.mean(slopes)),
        "slope_sd": float(np.std(slopes, ddof=1)),
        "slope_q025": float(np.quantile(slopes, 0.025)),
        "slope_q975": float(np.quantile(slopes, 0.975)),
        "std_mean": curves.mean(axis=0).tolist(),
        "std_q10": np.quantile(curves, 0.10, axis=0).tolist(),
        "std_q90": np.quantile(curves, 0.90, axis=0).tolist(),
    }


def cost_frontier(n_paths=3000, cost_bps=(0.0, 10.0, 25.0),
                  steps=(4, 8, 16, 32, 64, 128, 256, 512), seed=7):
    """RMSE frontier when discretisation risk trades off against fees."""
    out = {}
    for bps in cost_bps:
        rmse, mean_cost = [], []
        for n in steps:
            gross = hedge_pnl(n_paths, n, seed=seed)
            net = hedge_pnl(n_paths, n, seed=seed, cost_bps=bps)
            rmse.append(float(np.sqrt(np.mean(net ** 2))))
            mean_cost.append(float((gross - net).mean()))
        out[bps] = {"steps": list(steps), "rmse": rmse,
                    "mean_cost": mean_cost,
                    "best_n": int(steps[int(np.argmin(rmse))])}
    return out


def cost_optimum_robustness(seeds=range(12), n_paths=2500,
                            cost_bps=(10.0, 25.0),
                            steps=(4, 8, 16, 32, 64, 128, 256, 512)):
    out = {}
    for bps in cost_bps:
        best = []
        for seed in seeds:
            row = cost_frontier(n_paths=n_paths, cost_bps=(bps,),
                                steps=steps, seed=seed)[bps]
            best.append(row["best_n"])
        counts = {str(n): int(best.count(n)) for n in sorted(set(best))}
        out[str(int(bps))] = {"best_n_by_seed": best, "counts": counts}
    return out


def rebalancing_figure(frontier, path):
    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    for bps, row in frontier.items():
        label = f"{bps:.0f} bps (best N = {row['best_n']})"
        ax.loglog(row["steps"], row["rmse"], "o-", lw=1.8, label=label)
    ax.set(xlabel="rebalances over the option's life",
           ylabel="root mean square terminal P&L",
           title="Trading costs turn the discretisation curve into a U")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    FIGURES.mkdir(parents=True, exist_ok=True)
    steps, stds, slope = error_scaling()
    robust = scaling_robustness()

    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    s = np.asarray(robust["steps"])
    ax.fill_between(s, robust["std_q10"], robust["std_q90"], alpha=0.18,
                    color="#0b3d91", label="10–90% across seeds")
    ax.loglog(s, robust["std_mean"], "o-", lw=1.8, color="#0b3d91",
              label=f"mean slope {robust['slope_mean']:.3f}")
    ref = robust["std_mean"][0] * (s / s[0]) ** -0.5
    ax.loglog(s, ref, "--", lw=1.2, color="#c1440e",
              label="reference slope -1/2")
    ax.set(xlabel="rebalances over the option's life",
           ylabel="std of hedged P&L",
           title="The N^-1/2 hedging-error law survives across random seeds")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(FIGURES / "hedge_error_scaling.png", dpi=150)
    plt.close(fig)

    drifts = [-0.10, 0.02, 0.20]
    rows = []
    for mu in drifts:
        pnl = hedge_pnl(4000, 64, mu=mu, seed=11)
        rows.append((mu, float(pnl.mean()),
                     float(1.96 * pnl.std(ddof=1) / sqrt(len(pnl)))))
    fig2, ax2 = plt.subplots(figsize=(7.0, 2.8))
    for i, (mu, m, ci) in enumerate(rows):
        ax2.hlines(i, m - ci, m + ci, lw=2, color="#0b3d91")
        ax2.plot(m, i, "o", color="#0b3d91")
    ax2.axvline(0, color="crimson", ls="--", lw=1.2)
    ax2.set_yticks(range(len(rows)), [f"drift {mu:+.0%}" for mu, _, _ in rows])
    ax2.set(xlabel="mean hedged P&L (95% normal-approximation interval)",
            title="With correct volatility, drift has little effect on mean hedge P&L")
    ax2.grid(alpha=0.3, axis="x")
    fig2.tight_layout()
    fig2.savefig(FIGURES / "pnl_drift.png", dpi=150)
    plt.close(fig2)

    sale, real = 0.25, 0.20
    gap = price(Option(100, 100, 1.0, 0.02, sale)) \
        - price(Option(100, 100, 1.0, 0.02, real))
    at_real = hedge_pnl(4000, 128, sigma_sell=sale, sigma_real=real,
                        sigma_hedge=real, seed=3)
    at_impl = hedge_pnl(4000, 128, sigma_sell=sale, sigma_real=real,
                        sigma_hedge=sale, seed=3)
    fig3, ax3 = plt.subplots(figsize=(7.6, 4.2))
    bins = np.linspace(min(at_impl.min(), at_real.min()),
                       max(at_impl.max(), at_real.max()), 60)
    ax3.hist(at_impl, bins=bins, alpha=0.55,
             label=f"hedge at implied (mean {at_impl.mean():.2f})",
             color="#0b3d91")
    ax3.hist(at_real, bins=bins, alpha=0.55,
             label=f"hedge at realised (mean {at_real.mean():.2f})",
             color="#c1440e")
    ax3.axvline(gap, color="black", ls="--", lw=1.4,
                label=f"BS price gap {gap:.2f}")
    ax3.set(xlabel="hedged P&L from selling 25% vol in a 20% vol world",
            title="Same volatility spread, different pathwise hedge noise")
    ax3.legend(fontsize=8)
    ax3.grid(alpha=0.3)
    fig3.tight_layout()
    fig3.savefig(FIGURES / "vol_capture.png", dpi=150)
    plt.close(fig3)

    frontier = cost_frontier()
    rebalancing_figure(frontier, FIGURES / "optimal_rebalancing.png")
    cost_robust = cost_optimum_robustness()

    summary = {
        "single_seed_error_scaling_slope": round(slope, 3),
        "multi_seed_error_scaling": {
            "n_seeds": len(robust["slopes"]),
            "mean_slope": round(robust["slope_mean"], 4),
            "sd_slope": round(robust["slope_sd"], 4),
            "empirical_95pct_range": [round(robust["slope_q025"], 4),
                                      round(robust["slope_q975"], 4)],
        },
        "optimal_rebalancing_best_n_single_seed": {
            str(int(b)): row["best_n"] for b, row in frontier.items()},
        "optimal_rebalancing_robustness": cost_robust,
        "drift_independence": [
            {"mu": mu, "mean_pnl": round(m, 4), "ci95_halfwidth": round(ci, 4)}
            for mu, m, ci in rows],
        "vol_spread": {
            "bs_price_gap": round(gap, 4),
            "mean_hedge_at_realised": round(float(at_real.mean()), 4),
            "sd_hedge_at_realised": round(float(at_real.std(ddof=1)), 4),
            "mean_hedge_at_implied": round(float(at_impl.mean()), 4),
            "sd_hedge_at_implied": round(float(at_impl.std(ddof=1)), 4),
        },
    }
    (ROOT / "reports" / "hedging_summary.json").write_text(
        json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
