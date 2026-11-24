import numpy as np
import pytest

from attribution import gamma_weighted_attribution
from binomial import crr_price
from black_scholes import Option, price
from exotics import down_and_out_closed, down_and_out_mc, down_and_out_mc_bridge
from hedging import error_scaling
from heston import heston_call
from implied_vol import implied_vol, no_arbitrage_bounds
from monte_carlo import variance_reduction_benchmark


def test_antithetic_uncertainty_is_computed_at_pair_level():
    out = variance_reduction_benchmark(n_paths=60_000, seed=1)
    assert out["antithetic"]["se_reduction_factor"] > 1.2
    assert out["control_variate"]["se_reduction_factor"] > 2.0
    assert out["antithetic_plus_control"]["se_reduction_factor"] > 4.0
    true = price(Option(100, 100, 1.0, 0.05, 0.2))
    best = out["antithetic_plus_control"]
    assert abs(best["price"] - true) < 4 * best["se"]


def test_implied_vol_checks_bounds_and_roundtrips_put_with_dividend():
    o = Option(100, 110, 0.7, 0.03, 0.31, call=False, q=0.02)
    target = price(o)
    recovered = implied_vol(target, o.S, o.K, o.T, o.r,
                            call=False, q=o.q)
    assert abs(recovered - o.sigma) < 1e-6
    low, high = no_arbitrage_bounds(o.S, o.K, o.T, o.r,
                                    call=o.call, q=o.q)
    with pytest.raises(ValueError):
        implied_vol(high + 1.0, o.S, o.K, o.T, o.r,
                    call=o.call, q=o.q)
    assert low <= target <= high


def test_brownian_bridge_removes_most_monitoring_bias():
    o = Option(100, 100, 1.0, 0.03, 0.25)
    barrier = 85.0
    closed = down_and_out_closed(o, barrier)
    discrete, _ = down_and_out_mc(o, barrier, 12, n_paths=80_000, seed=7)
    bridge, se = down_and_out_mc_bridge(o, barrier, 12,
                                        n_paths=80_000, seed=7)
    assert abs(bridge - closed) < abs(discrete - closed)
    assert abs(bridge - closed) < 4 * se


def test_gamma_weighted_variance_attribution_tracks_hedged_pnl():
    pnl, attr = gamma_weighted_attribution(n_paths=2500, n_steps=128, seed=3)
    corr = np.corrcoef(pnl, attr)[0, 1]
    rmse = np.sqrt(np.mean((pnl - attr) ** 2))
    assert corr > 0.99
    assert rmse < 0.14
    assert abs(pnl.mean() - attr.mean()) < 0.03


def test_root_n_hedging_law_is_not_a_single_seed_accident():
    slopes = []
    for seed in range(6):
        _, _, slope = error_scaling(n_paths=1400,
                                    steps=(8, 16, 32, 64, 128, 256),
                                    seed=seed)
        slopes.append(slope)
    assert abs(np.mean(slopes) + 0.5) < 0.03
    assert max(slopes) < -0.44
    assert min(slopes) > -0.55


def test_invalid_model_inputs_fail_loudly():
    with pytest.raises(ValueError):
        Option(0, 100, 1, 0.02, 0.2)
    with pytest.raises(ValueError):
        crr_price(Option(100, 100, 1, 2.0, 0.01), n_steps=1)
    with pytest.raises(ValueError):
        heston_call(100, 100, 1, 0.02, v0=0.04, kappa=2.0,
                    theta=0.04, xi=0.6, rho=1.2)
