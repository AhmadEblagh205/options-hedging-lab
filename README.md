# Options Pricing, Hedging and Monte Carlo Laboratory

[![CI](https://github.com/AhmadEblagh205/options-hedging-lab/actions/workflows/ci.yml/badge.svg)](https://github.com/AhmadEblagh205/options-hedging-lab/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

I built this as an independent set of derivatives pricing and hedging experiments. Wherever a closed-form answer exists, I check the numerical code against it. The code links Black-Scholes, CRR trees, Monte Carlo, implied volatility, American exercise, exotic options, Heston stochastic volatility and discrete delta hedging.

It is a simulation project. It uses no live option-market data and reports no investment track record.

## What it does

- Prices European options and Greeks in closed form, and checks them against finite differences, put-call parity and the Black-Scholes PDE.
- Prices with CRR trees (including American exercise) and Monte Carlo, with antithetic and control-variate variance reduction.
- Simulates discrete delta hedging, with and without proportional trading costs.
- Prices Asian and barrier options, using a Brownian-bridge correction for discrete barrier monitoring.
- Prices under Heston stochastic volatility with a characteristic function and full-truncation Euler Monte Carlo.

## Results

All results are from simulated data.

- The log-log slope of hedging P&L standard deviation against the number of rebalances is -0.4858 on average over 20 seeds (2.5% to 97.5% range -0.4931 to -0.4778), against a theoretical -0.5.
- With 128 hedge intervals, gamma-weighted variance attribution matches actual hedged P&L with correlation 0.9941 and RMSE 0.101. At 512 intervals the correlation is about 0.9987.
- I corrected an error in the antithetic standard error, which had treated correlated paths as independent. Across ten seeds, the standard error falls by about 1.42 times (antithetic), 2.63 times (control variate) and 5.38 times (both).
- For a down-and-out call monitored at 52 dates, the continuous closed form is 10.1239, naive discrete Monte Carlo gives 10.5713 and the Brownian-bridge estimate is 10.1258.
- With proportional costs, 512 rebalances are best at 0 bps. At 10 bps, 256 is best in 10 of 12 seeds and 128 in 2. At 25 bps, 64 is best in all 12.
- The geometric Asian control variate cuts the arithmetic Asian standard error by about 29.8 times.
- Selling at 25% implied volatility into a world realising 20% gives a price gap of about 1.9545 and a mean hedged P&L of about 1.99 over 128 hedges.

Figures are in `reports/figures/` and the numbers are in `reports/*.json`.

## How to run

```bash
pip install -r requirements.txt
pytest -q
python src/monte_carlo.py
python src/hedging.py
python src/exotics.py
python src/heston.py
python src/attribution.py
python src/boundary_figure.py
```

The suite has 23 tests.

## Repository layout

- `src/black_scholes.py`: European prices and Greeks.
- `src/binomial.py`: CRR pricing and American exercise.
- `src/monte_carlo.py`: Monte Carlo pricing, antithetics and control variates.
- `src/implied_vol.py`: bounded implied volatility inversion and a mixture smile.
- `src/hedging.py`: discrete hedge simulator and trading costs.
- `src/exotics.py`: Asian options and the barrier bridge experiment.
- `src/heston.py`: Heston pricing and Monte Carlo.
- `src/attribution.py`: gamma-weighted P&L attribution.
- `docs/METHODOLOGY.md`: equations and assumptions.

## Notes

The Heston demonstration parameters have a Feller ratio of 0.444, so the Feller condition is not met and the variance boundary handling matters. The implied volatility solver rejects prices outside the no-arbitrage bounds. The trading-cost results come from the model and are not execution advice.

## Licence

MIT, see [LICENSE](LICENSE).
