
from binomial import crr_price, early_exercise_premium
from black_scholes import Option, delta, gamma, parity_gap, price, vega
from hedging import error_scaling, hedge_pnl
from implied_vol import implied_vol, smile
from monte_carlo import mc_price, pathwise_delta

ATM = Option(S=100, K=100, T=1.0, r=0.05, sigma=0.2)


def test_put_call_parity_holds_exactly():
    for K in (80, 100, 125):
        assert abs(parity_gap(100, K, 0.7, 0.03, 0.3)) < 1e-10


def test_greeks_match_finite_differences():
    h = 1e-4
    fd_delta = (price(Option(100 + h, 100, 1, 0.05, 0.2))
                - price(Option(100 - h, 100, 1, 0.05, 0.2))) / (2 * h)
    assert abs(fd_delta - delta(ATM)) < 1e-6
    fd_gamma = (price(Option(100 + h, 100, 1, 0.05, 0.2))
                - 2 * price(ATM)
                + price(Option(100 - h, 100, 1, 0.05, 0.2))) / h ** 2
    assert abs(fd_gamma - gamma(ATM)) < 1e-4
    fd_vega = (price(Option(100, 100, 1, 0.05, 0.2 + h))
               - price(Option(100, 100, 1, 0.05, 0.2 - h))) / (2 * h)
    assert abs(fd_vega - vega(ATM)) < 1e-4


def test_binomial_converges_and_american_costs_more():
    bs = price(ATM)
    assert abs(crr_price(ATM, 200) - bs) < 0.02
    assert abs(crr_price(ATM, 1000) - bs) < 0.005
    put = Option(100, 110, 1.0, 0.05, 0.25, call=False)
    prem = early_exercise_premium(put)
    assert prem > 0.5                       # deep-ish ITM put with rates
    call_prem = early_exercise_premium(ATM)
    assert abs(call_prem) < 1e-6            # no dividends, never exercise


def test_monte_carlo_unbiased_and_control_variate_earns_its_name():
    bs = price(ATM)
    est, se = mc_price(ATM, 100_000, seed=1)
    assert abs(est - bs) < 4 * se
    _, se_cv = mc_price(ATM, 100_000, seed=1, antithetic=True,
                        control_variate=True)
    assert se / se_cv > 2                   # measured, not asserted
    assert abs(pathwise_delta(ATM, 200_000, seed=2) - delta(ATM)) < 0.01


def test_implied_vol_roundtrips_and_the_mixture_smiles():
    p = price(Option(100, 90, 0.5, 0.02, 0.33))
    assert abs(implied_vol(p, 100, 90, 0.5, 0.02) - 0.33) < 1e-5
    sm = dict(smile())
    assert sm[70] > sm[100] and sm[130] > sm[100]      # a genuine smile


def test_hedging_error_scales_like_root_n():
    _, _, slope = error_scaling(n_paths=1200,
                                steps=(8, 16, 32, 64, 128, 256))
    assert abs(slope + 0.5) < 0.12


def test_drift_is_hedged_away_and_vol_spread_is_captured():
    premium = price(Option(100, 100, 1.0, 0.02, 0.2))
    for mu in (-0.10, 0.20):
        pnl = hedge_pnl(2500, 64, mu=mu, seed=5)
        assert abs(pnl.mean()) < 0.01 * premium   # under 1% of premium
    gap = price(Option(100, 100, 1.0, 0.02, 0.25)) \
        - price(Option(100, 100, 1.0, 0.02, 0.20))
    at_real = hedge_pnl(2500, 128, sigma_sell=0.25, sigma_real=0.20,
                        sigma_hedge=0.20, seed=6)
    at_impl = hedge_pnl(2500, 128, sigma_sell=0.25, sigma_real=0.20,
                        sigma_hedge=0.25, seed=6)
    assert abs(at_real.mean() - gap) < 0.15
    assert abs(at_impl.mean() - gap) < 0.15
    assert at_impl.std() > at_real.std()     # same mean, noisier path
    flipped = hedge_pnl(1500, 128, sigma_sell=0.20, sigma_real=0.25,
                        sigma_hedge=0.25, seed=6)
    assert flipped.mean() < -0.5             # selling cheap vol loses


def test_costs_create_an_interior_optimal_frequency():
    from hedging import cost_frontier
    steps = (4, 8, 16, 32, 64, 128, 256, 512)
    frontier = cost_frontier(n_paths=900, cost_bps=(0.0, 10.0, 25.0),
                             steps=steps)
    free = frontier[0.0]
    assert free["rmse"][-1] < free["rmse"][0]     # frictionless: hedge more
    assert free["best_n"] == 512
    for bps in (10.0, 25.0):
        costly = frontier[bps]
        assert costly["best_n"] < 512             # the U has an interior min
        assert costly["rmse"][-1] > min(costly["rmse"])
        assert costly["mean_cost"][-1] > costly["mean_cost"][0]
    assert frontier[25.0]["best_n"] <= frontier[10.0]["best_n"]


def test_dividends_flow_through_the_stack():
    from binomial import crr_price, early_exercise_premium
    from black_scholes import Option, parity_gap, price
    from monte_carlo import mc_price
    o = Option(100, 100, 1.0, 0.03, 0.2, q=0.04)
    assert abs(parity_gap(100, 100, 1.0, 0.03, 0.2, q=0.04)) < 1e-10
    assert abs(crr_price(o, 800) - price(o)) < 0.01
    est, se = mc_price(o, 120_000, seed=3, antithetic=True,
                       control_variate=True)
    assert abs(est - price(o)) < 4 * se
    # with a fat dividend yield, early exercise of the CALL becomes real
    rich_div = Option(100, 95, 1.0, 0.02, 0.2, call=True, q=0.08)
    assert early_exercise_premium(rich_div, 600) > 0.1


def test_put_boundary_rises_toward_the_strike():
    from binomial import exercise_boundary
    from black_scholes import Option
    o = Option(100, 100, 1.0, 0.05, 0.25, call=False)
    pts = exercise_boundary(o, 400)
    t, s = zip(*pts)
    # the tree's boundary wobbles node to node, so the shape claims are
    # made on a smoothed curve: early average well below late average,
    # never above the strike, and hugging it close to expiry
    import numpy as np
    early = np.mean([v for x, v in pts if x < 0.2])
    late = np.mean([v for x, v in pts if x > 0.95])
    assert late > early + 2.0
    assert max(s) <= o.K + 1e-9
    assert late > 0.92 * o.K


def test_kemna_vorst_control_variate_is_dramatic():
    from black_scholes import Option
    from exotics import asian_call_mc, geometric_asian_price, gbm_paths
    import numpy as np
    o = Option(100, 100, 1.0, 0.03, 0.25)
    # the geometric closed form is verified against its own Monte Carlo
    rng = np.random.default_rng(0)
    paths = gbm_paths(o, 60_000, 50, rng)
    geo_mc = float((np.exp(-o.r * o.T)
                    * np.maximum(np.exp(np.log(paths).mean(axis=1)) - o.K,
                                 0.0)).mean())
    assert abs(geo_mc - geometric_asian_price(o, 50)) < 0.05
    plain = asian_call_mc(o, seed=2, control_variate=False)
    cv = asian_call_mc(o, seed=2, control_variate=True)
    assert plain[1] / cv[1] > 15                 # the classic, measured
    assert abs(plain[0] - cv[0]) < 4 * plain[1]  # same answer, less noise


def test_barrier_bias_shrinks_with_monitoring():
    from black_scholes import Option
    from exotics import down_and_out_closed, down_and_out_mc
    o = Option(100, 100, 1.0, 0.03, 0.25)
    closed = down_and_out_closed(o, 85.0)
    vanilla = __import__("black_scholes").price(o)
    assert closed < vanilla                      # knocking out cannot add value
    coarse, _ = down_and_out_mc(o, 85.0, 12, n_paths=40_000)
    fine, _ = down_and_out_mc(o, 85.0, 504, n_paths=40_000)
    assert coarse > fine > closed - 0.05         # converges from above
    assert (coarse - closed) > (fine - closed)


def test_heston_collapses_to_black_scholes_and_mc_agrees():
    from math import sqrt as _sqrt
    from black_scholes import Option, price
    from heston import heston_call, heston_mc
    S, K, T, r = 100.0, 100.0, 1.0, 0.02
    limit = heston_call(S, K, T, r, v0=0.04, kappa=2.0, theta=0.04,
                        xi=1e-4, rho=0.0)
    assert abs(limit - price(Option(S, K, T, r, _sqrt(0.04)))) < 0.02
    params = dict(v0=0.04, kappa=2.0, theta=0.04, xi=0.6, rho=-0.7)
    semi = heston_call(S, K, T, r, **params)
    mc, se = heston_mc(S, K, T, r, n_paths=40_000, n_steps=200, seed=4,
                       **params)
    assert abs(mc - semi) < 4 * se


def test_negative_rho_makes_a_put_skew():
    from heston import heston_call
    from implied_vol import implied_vol
    S, T, r = 100.0, 0.5, 0.02
    base = dict(v0=0.04, kappa=2.0, theta=0.04, xi=0.6, rho=-0.7)
    iv_low = implied_vol(heston_call(S, 80, T, r, **base), S, 80, T, r)
    iv_atm = implied_vol(heston_call(S, 100, T, r, **base), S, 100, T, r)
    iv_high = implied_vol(heston_call(S, 120, T, r, **base), S, 120, T, r)
    assert iv_low > iv_atm                      # crash correlation: put skew
    assert iv_low - iv_atm > iv_high - iv_atm   # steeper on the downside


def test_the_pde_is_an_identity_across_the_grid():
    from attribution import pde_residual
    from black_scholes import Option
    worst = max(abs(pde_residual(Option(S, K, T, 0.04, sig, call, q=q)))
                for S in (80, 100, 130) for K in (90, 110)
                for T in (0.1, 1.0) for sig in (0.15, 0.4)
                for call in (True, False) for q in (0.0, 0.03))
    assert worst < 1e-8


def test_theta_equals_gamma_at_breakeven_when_rates_vanish():
    from attribution import breakeven_daily_move
    from black_scholes import Option, gamma, theta
    o = Option(100, 100, 0.25, 0.0, 0.2, q=0.0)
    lhs = -theta(o) / 252
    rhs = 0.5 * gamma(o) * breakeven_daily_move(o) ** 2
    assert abs(lhs - rhs) / rhs < 0.01


def test_hedged_pnl_is_short_realized_variance():
    import numpy as np
    from attribution import pnl_and_realized_var
    pnl, rv = pnl_and_realized_var(n_paths=2500, n_steps=96, seed=3)
    corr = float(np.corrcoef(rv, pnl)[0, 1])
    slope = float(np.polyfit(rv, pnl, 1)[0])
    assert corr < -0.75
    assert slope < 0
    near = pnl[np.abs(rv - 0.04) < 0.003]
    assert abs(near.mean()) < 0.15         # fair vol paths pay about zero
