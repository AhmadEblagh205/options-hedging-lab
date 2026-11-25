# Methodology and claim boundaries

This repository is an independent analytical/simulation project. It does **not** use live option-market data and it does not report a trading track record. Closed-form identities and controlled simulations are used precisely because the ground truth is available.

## 1. Black-Scholes benchmark

For a continuously dividend-paying underlying, the European call is

\[
C = S e^{-qT}N(d_1)-K e^{-rT}N(d_2),
\]

with

\[
d_1=\frac{\ln(S/K)+(r-q+\tfrac12\sigma^2)T}{\sigma\sqrt{T}},
\qquad d_2=d_1-\sigma\sqrt{T}.
\]

Prices, delta, gamma, vega, theta and rho are implemented analytically. The numerical stack is checked against these formulas, finite differences, put-call parity and the Black-Scholes PDE.

The PDE residual used as a consistency check is

\[
\Theta + \tfrac12\sigma^2 S^2\Gamma +(r-q)S\Delta-rV.
\]

Across the repository's test grid its maximum absolute residual is approximately `4.3e-15`.

## 2. CRR trees and American exercise

A Cox-Ross-Rubinstein tree uses

\[
u=e^{\sigma\sqrt{\Delta t}},\quad d=u^{-1},\quad
p=\frac{e^{(r-q)\Delta t}-d}{u-d}.
\]

The implementation validates that `p` lies in `[0,1]`. European tree prices are checked for convergence to Black-Scholes. American values are computed by comparing continuation and immediate exercise at every node. The exercise-boundary helper is intentionally restricted to puts.

## 3. Monte Carlo and variance reduction

Risk-neutral GBM terminal prices are simulated as

\[
S_T=S_0\exp\left[(r-q-\tfrac12\sigma^2)T+\sigma\sqrt{T}Z\right].
\]

### Antithetic estimator

For each independent `Z`, a paired path is generated with `-Z`. The two payoffs are averaged **before** the standard error is calculated. The independent sampling unit is therefore the pair, not the individual path. The original version of this repository failed to respect that covariance structure when reporting the antithetic standard error; v2 corrects it.

### Control variate

The terminal stock price is used because its pricing-measure expectation is known:

\[
E[S_T]=S_0e^{(r-q)T}.
\]

With estimated coefficient `beta`, the adjusted payoff is

\[
Y^*=Y-\beta(X-E[X]).
\]

Across ten fixed-seed equal-path experiments, the mean standard-error reduction factors are approximately:

- antithetic only: **1.42x**;
- terminal-stock control variate: **2.63x**;
- antithetic + control variate: **5.38x**.

These are simulation results for the stated benchmark, not universal constants.

## 4. Implied volatility

The implied-volatility solver uses a bracketed bisection method. Before inversion it checks the European no-arbitrage interval.

For calls:

\[
\max(0,Se^{-qT}-Ke^{-rT}) \le C < Se^{-qT}.
\]

For puts:

\[
\max(0,Ke^{-rT}-Se^{-qT}) \le P < Ke^{-rT}.
\]

Impossible target prices raise an exception instead of silently returning a volatility endpoint. The mixture-world smile is generated from a controlled mixture of Black-Scholes regimes and is not fitted to observed option prices.

## 5. Delta hedging and the root-N law

A short call is financed and rebalanced to the Black-Scholes delta on a discrete grid. With correct volatility and zero transaction costs, the standard deviation of discretisation error should scale approximately as

\[
\operatorname{sd}(\Pi_N)\propto N^{-1/2}.
\]

The original single-seed fit was about `-0.48`. V2 repeats the experiment across 20 random seeds: the mean fitted slope is **-0.4858**, with empirical 2.5%-97.5% range **[-0.4931, -0.4778]**. This is the stronger evidence for the public `-0.48 vs -0.50` claim.

## 6. Transaction costs

A one-way proportional fee is charged on the initial hedge, every rebalance and final liquidation. The objective reported is terminal-P&L RMSE around zero, which combines discretisation risk and trading-cost drag.

In the default experiment:

- zero cost continues to favour the finest grid;
- 10 bps gives a best grid of 256 hedges in 10/12 random seeds and 128 in 2/12;
- 25 bps gives 64 hedges in 12/12 seeds.

These optima depend on the model, fee definition and option parameters; they are not market recommendations.

## 7. Gamma-weighted realised-variance attribution

A raw regression of hedged P&L on unweighted realised variance is intuitive but incomplete because gamma changes through time. For a short option sold/hedged at implied volatility, the continuous-time PDE gives the local variance contribution

\[
\frac12\Gamma_tS_t^2\left(\sigma_{imp}^2dt-d\langle\log S\rangle_t\right).
\]

The implementation approximates the quadratic variation by squared log-return increments and grows each local contribution to expiry at the risk-free rate.

With 128 hedge intervals, the pathwise correlation between actual terminal hedge P&L and this gamma-weighted attribution is **0.9941**, with RMSE about **0.101**. As the hedge grid is refined, the correlation rises and the residual falls, reaching correlation about **0.9987** at 512 intervals in the controlled experiment.

This is a more defensible statement than saying a delta-hedged option is simply a linear position in raw realised variance.

## 8. Asian control variate

The arithmetic-average Asian call has no simple Black-Scholes-style closed form. Its discretely monitored geometric-average sibling is lognormal with a closed-form price, making it a strong control variate. In the default simulation the standard error falls by about **29.8x**.

## 9. Barrier monitoring and Brownian bridges

A grid-monitored down-and-out simulation overprices a continuously monitored knock-out because it misses barrier crossings between dates.

Conditional on two log-price endpoints `x0` and `x1` above the log barrier `b`, the GBM Brownian-bridge crossing probability over interval `dt` is

\[
P(\text{hit}|x_0,x_1)=
\exp\left[-\frac{2(x_0-b)(x_1-b)}{\sigma^2\Delta t}\right].
\]

The corrected Monte Carlo multiplies the conditional interval survival probabilities. This removes most of the monitoring-grid bias and is checked against the continuous Reiner-Rubinstein closed form. It is exact for the GBM bridge assumptions used here; it should not be transplanted unchanged to arbitrary stochastic-volatility/jump models.

## 10. Heston

The Heston example is a controlled stochastic-volatility model, not a market calibration. A semi-closed characteristic-function price is checked against full-truncation Euler Monte Carlo and against the Black-Scholes limit as vol-of-vol tends to zero.

The demonstration parameters deliberately have

\[
\frac{2\kappa\theta}{\xi^2}=0.444<1,
\]

so the Feller sufficient condition is not satisfied. That makes variance-boundary handling in the simulation material and is one reason full truncation is used. Failure of the Feller condition does not by itself invalidate the Heston price.

## 11. Claim boundaries

The repository supports claims about:

- analytical/numerical consistency;
- controlled Monte Carlo efficiency;
- hedging-error scaling under the stated model;
- option-P&L attribution under controlled volatility mismatch;
- transaction-cost sensitivity under an illustrative fee model;
- American exercise and path-dependent option mechanics;
- stochastic-volatility smile/skew mechanics.

It does **not** support claims about:

- realised investment returns;
- live-market calibration quality;
- profitable option trading;
- production execution costs;
- empirical option-market forecasting.


## Notes moved from the README

- Hedging slope: the original one-seed fit was about -0.48. Over 20 seeds the mean is -0.4858 with a 2.5% to 97.5% range of [-0.4931, -0.4778].
- Gamma attribution at 128 intervals: correlation 0.9941, RMSE 0.101, mean actual P&L 1.983, mean attributed P&L 1.981. A plain regression of hedged P&L on realised variance gives a correlation of about -0.81.
- Antithetic bug: the standard error treated the z and -z payoffs as independent. The sampling unit is the pair average. The earlier claim that antithetics do essentially nothing came from that error.
- Barrier at 52 intervals: continuous closed form 10.1239, naive discrete Monte Carlo 10.5713, Brownian-bridge estimate 10.1258.
- Volatility mismatch: selling at 25% implied volatility in a world realising 20% gives a price gap of about 1.9545 and a mean hedged P&L of about 1.99 over 128 hedges, whether delta uses realised or implied volatility (the implied-volatility hedge has more pathwise dispersion). With sale, realised and hedge volatility equal, changing drift from -10% to +20% has little effect on mean hedged P&L.
- Validation: put-call parity holds below 1e-10 and the PDE residual is below about 4.3e-15 on the test grid. Heston demonstration parameters have a Feller ratio of 0.444.
