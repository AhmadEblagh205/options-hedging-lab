"""Render the American put exercise boundary.

Usage:
    python src/boundary_figure.py
"""
from pathlib import Path

import matplotlib.pyplot as plt

from binomial import exercise_boundary
from black_scholes import Option

ROOT = Path(__file__).resolve().parents[1]


def main():
    o = Option(S=100, K=100, T=1.0, r=0.05, sigma=0.25, call=False)
    pts = exercise_boundary(o, 600)
    t, s = zip(*pts)
    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    ax.plot(t, s, lw=1.8, color="#0b3d91")
    ax.axhline(o.K, color="grey", ls="--", lw=1, label="strike")
    ax.set(xlabel="time (years)", ylabel="critical spot",
           title="American put: exercise below the line")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    out = ROOT / "reports" / "figures" / "exercise_boundary.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"Saved -> {out}")


if __name__ == "__main__":
    main()
