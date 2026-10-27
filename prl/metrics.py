"""Performance metrics and Sharpe-ratio difference tests."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

ANN = 252


def drawdown(r: pd.Series) -> pd.Series:
    wealth = (1 + r).cumprod()
    return wealth / wealth.cummax() - 1


def perf_table(rets: pd.DataFrame, rf: pd.Series, turnover: pd.DataFrame | None = None) -> pd.DataFrame:
    rows = {}
    years = len(rets) / ANN
    for c in rets.columns:
        r = rets[c]
        ex = r - rf
        wealth = (1 + r).prod()
        cagr = wealth ** (1 / years) - 1
        vol = r.std(ddof=1) * np.sqrt(ANN)
        sharpe = ex.mean() / ex.std(ddof=1) * np.sqrt(ANN)
        downside = np.sqrt(np.mean(np.minimum(ex, 0) ** 2)) * np.sqrt(ANN)
        sortino = ex.mean() * ANN / downside
        mdd = drawdown(r).min()
        row = {
            "CAGR": cagr,
            "Volatility": vol,
            "Sharpe": sharpe,
            "Sortino": sortino,
            "Max drawdown": mdd,
            "Calmar": cagr / abs(mdd),
            "Growth of $1": wealth,
        }
        if turnover is not None:
            to = turnover[c].dropna().iloc[1:]  # exclude the initial build from cash
            row["Turnover (one-way, ann.)"] = to.sum() / years
        rows[c] = row
    return pd.DataFrame(rows).T


# --------------------------------------------------------------------------- #
# Sharpe difference tests                                                      #
# --------------------------------------------------------------------------- #
def jobson_korkie_memmel(ex_a: np.ndarray, ex_b: np.ndarray) -> tuple[float, float]:
    """Jobson-Korkie (1981) test with Memmel (2003) correction. Assumes iid
    normal returns. Returns (z, two-sided p)."""
    T = len(ex_a)
    sa, sb = ex_a.std(ddof=1), ex_b.std(ddof=1)
    SRa, SRb = ex_a.mean() / sa, ex_b.mean() / sb
    rho = np.corrcoef(ex_a, ex_b)[0, 1]
    theta = (2 - 2 * rho + 0.5 * (SRa**2 + SRb**2 - 2 * SRa * SRb * rho**2)) / T
    z = (SRa - SRb) / np.sqrt(theta)
    return float(z), float(2 * stats.norm.sf(abs(z)))


def block_bootstrap_sharpe_diff(ex_a: np.ndarray, ex_b: np.ndarray, n_boot: int = 5000,
                                block: int = 21, seed: int = 0) -> dict:
    """Circular block bootstrap of the annualised Sharpe difference SR_a - SR_b.

    Resamples blocks of `block` consecutive days jointly for both series, which
    keeps their cross-correlation, volatility clustering and fat tails.
    p-value: share of centred bootstrap differences at least as extreme as the
    observed one (two-sided).
    """
    rng = np.random.default_rng(seed)
    T = len(ex_a)
    nb = int(np.ceil(T / block))
    X = np.column_stack([ex_a, ex_b])

    def sr_diff(M):
        m = M.mean(axis=-2)
        s = M.std(axis=-2, ddof=1)
        sr = m / s * np.sqrt(ANN)
        return sr[..., 0] - sr[..., 1]

    obs = float(sr_diff(X))
    diffs = np.empty(n_boot)
    chunk = 500
    for k in range(0, n_boot, chunk):
        m = min(chunk, n_boot - k)
        starts = rng.integers(0, T, size=(m, nb))
        idx = (starts[:, :, None] + np.arange(block)[None, None, :]).reshape(m, -1)[:, :T] % T
        diffs[k:k + m] = sr_diff(X[idx])
    p = float(np.mean(np.abs(diffs - obs) >= abs(obs)))
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return {"diff": obs, "ci_low": float(lo), "ci_high": float(hi), "p_value": p}
