# Generated tables (python run.py)

## Strategy performance, net of costs

|  | CAGR | Vol | Sharpe | Sortino | Max DD | Calmar | Turnover/yr | $1 grows to |
|---|---|---|---|---|---|---|---|---|
| Equal weight | 6.00% | 11.74% | 0.43 | 0.62 | -34.3% | 0.18 | 30% | $2.99 |
| 60/40 | 7.92% | 11.87% | 0.58 | 0.82 | -35.2% | 0.22 | 20% | $4.19 |
| Min variance | 3.89% | 5.58% | 0.46 | 0.63 | -17.4% | 0.22 | 35% | $2.05 |
| Max Sharpe | 7.76% | 10.58% | 0.63 | 0.88 | -25.5% | 0.30 | 238% | $4.08 |
| Risk parity | 5.41% | 7.84% | 0.53 | 0.75 | -23.7% | 0.23 | 30% | $2.69 |

## Sharpe difference tests

|  | Sharpe diff | 95% bootstrap CI | bootstrap p | JK-Memmel p |
|---|---|---|---|---|
| 60/40 vs Equal weight | +0.15 | [-0.06, +0.36] | 0.16 | 0.18 |
| Min variance vs Equal weight | +0.02 | [-0.28, +0.34] | 0.89 | 0.91 |
| Max Sharpe vs Equal weight | +0.19 | [-0.17, +0.54] | 0.28 | 0.36 |
| Risk parity vs Equal weight | +0.10 | [-0.10, +0.30] | 0.35 | 0.43 |
| Equal weight vs 60/40 | -0.15 | [-0.36, +0.06] | 0.16 | 0.18 |
| Min variance vs 60/40 | -0.13 | [-0.54, +0.29] | 0.55 | 0.59 |
| Max Sharpe vs 60/40 | +0.04 | [-0.40, +0.46] | 0.84 | 0.86 |
| Risk parity vs 60/40 | -0.05 | [-0.40, +0.28] | 0.76 | 0.77 |

## VaR backtest, 99%

|  | Exceptions | Rate | Kupiec p | Christoffersen ind. p | Cond. coverage p | Avg VaR | Avg ES | Loss / ES on exception days | Basel: share of 250d windows green/yellow/red | Worst 250d count | Last 250d |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Historical sim | 92 / 53 | 1.74% | 0.000 | 0.000 | 0.000 | 1.95% | 2.56% | 1.06 | 62% / 24% / 14% | 23 | 1 (green) |
| Normal | 144 / 53 | 2.72% | 0.000 | 0.000 | 0.000 | 1.54% | 1.77% | 1.34 | 46% / 23% / 31% | 31 | 2 (green) |
| EWMA (0.94) | 118 / 53 | 2.23% | 0.000 | 0.067 | 0.000 | 1.43% | 1.64% | 1.20 | 27% / 67% / 6% | 12 | 5 (yellow) |
| Filtered HS | 61 / 53 | 1.15% | 0.273 | 0.001 | 0.002 | 1.86% | 2.39% | 0.97 | 84% / 16% / 0% | 7 | 3 (green) |

## VaR backtest, 95%

|  | Exceptions | Rate | Kupiec p | Christoffersen ind. p | Cond. coverage p | Avg VaR | Avg ES | Loss / ES on exception days |
|---|---|---|---|---|---|---|---|---|
| Historical sim | 295 / 264 | 5.58% | 0.057 | 0.000 | 0.000 | 1.06% | 1.62% | 1.07 |
| Normal | 286 / 264 | 5.41% | 0.177 | 0.000 | 0.000 | 1.08% | 1.36% | 1.33 |
| EWMA (0.94) | 285 / 264 | 5.39% | 0.198 | 0.230 | 0.213 | 1.01% | 1.27% | 1.17 |
| Filtered HS | 275 / 264 | 5.20% | 0.504 | 0.132 | 0.257 | 1.07% | 1.58% | 0.96 |

## 99% VaR exceptions in selected calendar years (Basel zone)

|  | 2008 | 2020 | 2022 |
|---|---|---|---|
| EWMA (0.94) | 6 (yellow) | 10 (red) | 7 (yellow) |
| Filtered HS | 3 (green) | 6 (yellow) | 1 (green) |
| Historical sim | 22 (red) | 11 (red) | 8 (yellow) |
| Normal | 30 (red) | 13 (red) | 12 (red) |

## 99% VaR exceptions in crisis windows

|  | 2008-09 (Lehman to Mar 2009 low) | 2020 (COVID crash) |
|---|---|---|
| EWMA (0.94) | 4 (exp. 1.5) | 5 (exp. 0.8) |
| Filtered HS | 2 (exp. 1.5) | 5 (exp. 0.8) |
| Historical sim | 18 (exp. 1.5) | 10 (exp. 0.8) |
| Normal | 23 (exp. 1.5) | 12 (exp. 0.8) |

## Calendar-year net returns

|  | 2008 | 2020 | 2022 |
|---|---|---|---|
| Equal weight | -16.89% | 13.28% | -17.77% |
| 60/40 | -21.15% | 14.62% | -15.85% |
| Min variance | 5.96% | 9.85% | -12.46% |
| Max Sharpe | -1.09% | 11.27% | -17.08% |
| Risk parity | -0.50% | 11.74% | -17.34% |
