"""Strict walk-forward monthly-rebalanced backtest.

Timing: on the last trading day of each month t, weights are estimated from
the trailing `window` daily returns ending on day t (all known at the close of
t) and the portfolio is traded at that close. New weights earn returns from
day t+1. Between rebalances weights drift with prices. Trading cost is
`cost_bps` per unit of one-way traded notional, charged on the rebalance day.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import optimize as opt

STRATEGIES = ["Equal weight", "60/40", "Min variance", "Max Sharpe", "Risk parity"]


@dataclass
class BacktestResult:
    net: pd.DataFrame            # daily net returns per strategy
    gross: pd.DataFrame          # daily returns before costs
    turnover: pd.DataFrame       # one-way turnover on each rebalance date
    weights: dict = field(default_factory=dict)   # strategy -> DataFrame of target weights
    shrinkage: pd.Series = None  # Ledoit-Wolf intensity at each rebalance
    max_sharpe_fallbacks: int = 0


def target_weights(name, window_rets: np.ndarray, rf_window: np.ndarray, tickers):
    n = window_rets.shape[1]
    if name == "Equal weight":
        return np.full(n, 1.0 / n), None, False
    if name == "60/40":
        w = np.zeros(n)
        w[tickers.index("SPY")] = 0.6
        w[tickers.index("AGG")] = 0.4
        return w, None, False
    cov, delta = opt.ledoit_wolf(window_rets)
    if name == "Min variance":
        return opt.min_variance(cov), delta, False
    if name == "Risk parity":
        return opt.risk_parity(cov), delta, False
    if name == "Max Sharpe":
        mu_ex = window_rets.mean(axis=0) - rf_window.mean()
        w, fb = opt.max_sharpe(mu_ex, cov)
        return w, delta, fb
    raise ValueError(name)


def run_backtest(rets: pd.DataFrame, rf: pd.Series, window: int = 756,
                 cost_bps: float = 10.0) -> BacktestResult:
    tickers = list(rets.columns)
    R = rets.to_numpy()
    rfv = rf.to_numpy()
    dates = rets.index
    month_end = dates.to_series().groupby(dates.to_period("M")).transform("max") == dates
    rebal_idx = [i for i in np.flatnonzero(month_end.to_numpy()) if i >= window - 1 and i < len(dates) - 1]
    start = rebal_idx[0]

    out_gross, out_net, out_to, all_w = {}, {}, {}, {}
    shrink, fallbacks = {}, 0
    for name in STRATEGIES:
        w_cur = np.zeros(len(tickers))  # start in cash: first trade is full turnover
        g = np.zeros(len(dates))
        c = np.zeros(len(dates))
        to = {}
        wts = {}
        rset = set(rebal_idx)
        for i in range(start, len(dates)):
            if i > start:
                # earn day-i returns with weights held from the previous close
                port = w_cur @ R[i]
                g[i] = port
                w_cur = w_cur * (1 + R[i]) / (1 + port)
            if i in rset:
                win = R[i - window + 1: i + 1]
                w_new, delta, fb = target_weights(name, win, rfv[i - window + 1: i + 1], tickers)
                if delta is not None and name == "Min variance":
                    shrink[dates[i]] = delta
                fallbacks += int(fb)
                trade = np.abs(w_new - w_cur).sum()
                to[dates[i]] = trade
                # cost is paid at the rebalance close, i.e. deducted from day-i return
                c[i] = cost_bps / 1e4 * trade
                w_cur = w_new
                wts[dates[i]] = w_new
        sl = slice(start + 1, len(dates))
        out_gross[name] = pd.Series(g[sl], index=dates[sl])
        # cost at the close of day i is charged against the next day's return
        # for the first rebalance (no return on day `start` in the sample)
        cc = c.copy()
        cc[start + 1] += cc[start]
        out_net[name] = pd.Series((g - cc)[sl], index=dates[sl])
        out_to[name] = pd.Series(to)
        all_w[name] = pd.DataFrame(wts, index=tickers).T

    return BacktestResult(
        net=pd.DataFrame(out_net),
        gross=pd.DataFrame(out_gross),
        turnover=pd.DataFrame(out_to),
        weights=all_w,
        shrinkage=pd.Series(shrink),
        max_sharpe_fallbacks=fallbacks,
    )
