"""Monte Carlo pricing with variance reduction measured correctly.

Antithetic paths are *paired estimators*: the two payoffs generated from z and
-z are averaged before the standard error is computed. Treating those two
payoffs as independent understates/overstates the sampling uncertainty because
it discards their covariance. The control variate is the terminal stock price,
whose expectation is known exactly under the pricing measure.
"""
import json
from math import exp, sqrt
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from black_scholes import Option

ROOT = Path(__file__).resolve().parents[1]


def terminal_prices(o: Option, n_paths: int, rng,
                    antithetic: bool = False) -> np.ndarray:
    if n_paths <= 1:
        raise ValueError("n_paths must exceed one")
    if antithetic:
        if n_paths % 2:
            raise ValueError("antithetic sampling requires an even n_paths")
        z = rng.standard_normal(n_paths // 2)
        z = np.concatenate([z, -z])
    else:
        z = rng.standard_normal(n_paths)
    drift = (o.r - o.q - 0.5 * o.sigma ** 2) * o.T
    return o.S * np.exp(drift + o.sigma * sqrt(o.T) * z)


def _payoff(o: Option, ST: np.ndarray) -> np.ndarray:
    return np.maximum(ST - o.K, 0.0) if o.call else np.maximum(o.K - ST, 0.0)


def _estimator_units(o: Option, n_paths: int, rng, antithetic: bool):
    """Return independent estimator units (discounted payoff, terminal S).

    For ordinary MC, each path is one unit. For antithetic MC, each z/-z pair
    is averaged first, leaving n_paths/2 independent pair-level observations.
    """
    disc = exp(-o.r * o.T)
    if not antithetic:
        ST = terminal_prices(o, n_paths, rng, antithetic=False)
        return disc * _payoff(o, ST), ST

    if n_paths % 2:
        raise ValueError("antithetic sampling requires an even n_paths")
    m = n_paths // 2
    z = rng.standard_normal(m)
    drift = (o.r - o.q - 0.5 * o.sigma ** 2) * o.T
    scale = o.sigma * sqrt(o.T)
    ST_plus = o.S * np.exp(drift + scale * z)
    ST_minus = o.S * np.exp(drift - scale * z)
    y = 0.5 * disc * (_payoff(o, ST_plus) + _payoff(o, ST_minus))
    x = 0.5 * (ST_plus + ST_minus)
    return y, x


def mc_price(o: Option, n_paths: int = 100_000, seed: int = 0,
             antithetic: bool = False,
             control_variate: bool = False):
    """Return ``(estimate, standard_error)``.

    ``n_paths`` counts total simulated trajectories. With antithetics, those
    trajectories form ``n_paths/2`` independent pairs and the uncertainty is
    calculated at pair level.
    """
    rng = np.random.default_rng(seed)
    y, x = _estimator_units(o, n_paths, rng, antithetic)
    if control_variate:
        if len(y) < 3:
            raise ValueError("too few estimator units for a control variate")
        beta = np.cov(y, x, ddof=1)[0, 1] / np.var(x, ddof=1)
        known_x = o.S * exp((o.r - o.q) * o.T)
        y = y - beta * (x - known_x)
    return float(y.mean()), float(y.std(ddof=1) / sqrt(len(y)))


def pathwise_delta(o: Option, n_paths: int = 100_000,
                   seed: int = 0) -> float:
    """Pathwise derivative estimator for European delta."""
    if n_paths % 2:
        raise ValueError("pathwise antithetic estimator requires even n_paths")
    rng = np.random.default_rng(seed)
    ST = terminal_prices(o, n_paths, rng, antithetic=True)
    if not o.call:
        vals = exp(-o.r * o.T) * -(ST < o.K) * ST / o.S
    else:
        vals = exp(-o.r * o.T) * (ST > o.K) * ST / o.S
    return float(np.mean(vals))


def variance_reduction_benchmark(o=None, n_paths=100_000, seed=1):
    """Compare equal-trajectory Monte Carlo estimators by standard error."""
    o = o or Option(100, 100, 1.0, 0.05, 0.2)
    configs = {
        "plain": (False, False),
        "antithetic": (True, False),
        "control_variate": (False, True),
        "antithetic_plus_control": (True, True),
    }
    out = {}
    for name, (anti, cv) in configs.items():
        est, se = mc_price(o, n_paths=n_paths, seed=seed,
                           antithetic=anti, control_variate=cv)
        out[name] = {"price": est, "se": se}
    base = out["plain"]["se"]
    for row in out.values():
        row["se_reduction_factor"] = base / row["se"]
    return out


def variance_reduction_robustness(seeds=range(10), n_paths=50_000):
    names = ["antithetic", "control_variate", "antithetic_plus_control"]
    rows = {name: [] for name in names}
    for seed in seeds:
        out = variance_reduction_benchmark(n_paths=n_paths, seed=seed)
        for name in names:
            rows[name].append(out[name]["se_reduction_factor"])
    return {
        name: {
            "mean_factor": float(np.mean(vals)),
            "min_factor": float(np.min(vals)),
            "max_factor": float(np.max(vals)),
        }
        for name, vals in rows.items()
    }


def main():
    out = variance_reduction_benchmark()
    robust = variance_reduction_robustness()
    rounded = {
        k: {"price": round(v["price"], 5),
            "se": round(v["se"], 6),
            "se_reduction_factor": round(v["se_reduction_factor"], 2)}
        for k, v in out.items()
    }
    reports = ROOT / "reports"
    figures = reports / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    payload = {"representative_seed": rounded,
               "ten_seed_robustness": {
                   k: {kk: round(vv, 2) for kk, vv in row.items()}
                   for k, row in robust.items()
               }}
    (reports / "variance_reduction_summary.json").write_text(
        json.dumps(payload, indent=2))

    labels = ["Plain", "Antithetic", "Control\nvariate", "Both"]
    ses = [out[k]["se"] for k in out]
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    bars = ax.bar(labels, ses, color="#0b3d91", alpha=0.85)
    ax.set(ylabel="Monte Carlo standard error",
           title="Variance reduction: uncertainty must be measured at pair level")
    ax.grid(alpha=0.25, axis="y")
    for bar, key in zip(bars, out):
        fac = out[key]["se_reduction_factor"]
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                f"{fac:.2f}×", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(figures / "variance_reduction.png", dpi=150)
    plt.close(fig)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
