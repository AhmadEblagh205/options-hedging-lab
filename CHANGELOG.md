# Changelog

## 0.7.0 — defensibility and attribution audit

- Corrected the antithetic Monte Carlo uncertainty calculation: `z/-z` paths are now averaged into independent pair-level estimator units before standard errors are computed. The old “antithetics do essentially nothing” interpretation is retired. Across ten seeds, mean standard-error reduction is now approximately 1.42x for antithetics, 2.63x for the terminal-stock control variate, and 5.38x when combined.
- Added European no-arbitrage checks and safeguarded implied-volatility bracketing, including dividend-aware put/call round trips and explicit failure on impossible target prices.
- Added Brownian-bridge continuous-barrier correction and validation against the Reiner-Rubinstein down-and-out closed form; retained the naive discrete monitor as a deliberate bias demonstration.
- Added pathwise gamma-weighted realised-variance attribution from the Black-Scholes PDE. At 128 hedge intervals the controlled experiment gives correlation 0.9941 between attributed and actual terminal P&L, rising further as the grid is refined.
- Repeated the `N^-1/2` hedge-error experiment across 20 random seeds: mean slope -0.4858, empirical 95% range [-0.4931, -0.4778].
- Repeated transaction-cost hedge-frequency selection across seeds; 25 bps selects 64 hedges in all 12 checked seeds, while 10 bps selects 256 in 10/12 and 128 in 2/12.
- Added input/domain guards to Black-Scholes, CRR, barrier, Heston and simulation helpers so unsupported parameter regions fail loudly.
- Exposed the Heston Feller ratio in the generated report and documented why full truncation matters for the chosen demonstration parameters.
- Added `docs/METHODOLOGY.md`, new figures/reports, and six defensibility tests. Test suite: 23 passing tests.

# Changelog

## 0.6.0, update round 5
P&L attribution and the executable PDE. The Black-Scholes equation is now a test: theta plus half sigma squared S squared gamma plus (r minus q) S delta minus rV evaluates to 4e-15 at worst across a grid of calls, puts, tenors and dividend yields, the strongest possible consistency check on the Greeks. At zero rates it reduces to the trader's daily rule, theta decay equals half gamma times the breakeven move squared, verified to one percent. And a per path regression shows what the hedged book is: hedged short call P&L against realised variance is a clean negative line (correlation -0.81, fair vol paths paying about zero), a delta hedged option is a realised variance position, drawn.

## 0.5.0, update round 4
Heston stochastic volatility: the semi closed form through the characteristic function in its numerically stable branch, checked two ways, the vol of vol to zero limit reproduces Black-Scholes to the fourth decimal (8.916 both), and a full truncation Euler Monte Carlo agrees with the integral inside its standard error (8.147 against 8.137, se 0.04). The smile figure shows correlation setting the skew: rho at -0.7 steepens the put wing, at +0.7 the call wing, at zero a symmetric smile.

## 0.4.0, update round 3
Exotics with the classic tools. Arithmetic Asian call by Monte Carlo with the Kemna-Vorst geometric control variate, the discrete geometric closed form derived and verified against its own simulation, and the reduction measured: the control variate cuts the standard error 29.8 times. Down and out barrier priced against the Reiner-Rubinstein continuous closed form, with the discrete monitoring bias charted, converging to the closed form from above as monitoring dates grow.

## 0.3.0, update round 2
Dividends through the whole stack (a continuous yield q in the closed forms, every Greek, parity, the tree and the Monte Carlo drift, defaulting to zero so nothing upstream moved), plus the two theorems it unlocks: the American call premium becomes real under a fat yield, and the American put exercise boundary is now computed from the tree and rendered, rising toward the strike as expiry approaches.

## 0.2.0, update round 1
Transaction costs in the hedging simulator (a per side fee on every rebalance trade, the initial delta and the final unwind) and the study they enable: the frictionless minus one half law becomes a U, with the cost optimal rebalancing frequency falling from 512 at zero cost to 256 at 10 basis points to 64 at 25, over hedging measurably worse than the interior optimum.

## 0.1.0
Initial release: Black-Scholes closed forms with all Greeks and parity, CRR trees with American exercise, Monte Carlo with measured variance reduction and a pathwise delta, an implied vol solver with a mixture-world smile demonstration, and a delta hedging simulator that measures the root N error law, drift independence, and vol spread capture under both hedge vols.
