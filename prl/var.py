"""One-day Value-at-Risk / Expected Shortfall models and backtests.

Convention: VaR and ES are reported as positive loss fractions. An exception
on day t is r_t < -VaR_t, where VaR_t is forecast with data up to day t-1.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from scipy import stats

MODELS = ["Historical sim", "Normal", "EWMA (0.94)", "Filtered HS"]


# --------------------------------------------------------------------------- #
# Single-sample building blocks (used by the rolling engine and the tests)     #
# --------------------------------------------------------------------------- #
def hs_var_es(sample: np.ndarray, alpha: float) -> tuple[float, float]:
    """Historical-simulation VaR/ES at tail probability alpha (e.g. 0.01).
    VaR is the empirical alpha-quantile of returns (linear interpolation),
    ES is the mean of returns at or below that quantile."""
    q = np.quantile(sample, alpha)
    return -q, -sample[sample <= q].mean()


def normal_var_es(mu: float, sigma: float, alpha: float) -> tuple[float, float]:
    z = stats.norm.ppf(alpha)
    var = -(mu + sigma * z)
    es = -(mu - sigma * stats.norm.pdf(z) / alpha)
    return var, es


def ewma_variance(r: np.ndarray, lam: float = 0.94, init_window: int = 30) -> np.ndarray:
    """sigma2[t] = forecast variance for day t using returns up to t-1
    (RiskMetrics, zero mean). Seeded with the mean square of the first
    `init_window` returns; sigma2[t] for t < init_window is NaN."""
    s2 = np.full(len(r), np.nan)
    s2[init_window] = np.mean(r[:init_window] ** 2)
    for t in range(init_window + 1, len(r)):
        s2[t] = lam * s2[t - 1] + (1 - lam) * r[t - 1] ** 2
    return s2


# --------------------------------------------------------------------------- #
# Rolling out-of-sample forecasts                                              #
# --------------------------------------------------------------------------- #
def rolling_forecasts(r: pd.Series, alpha: float, window: int = 500,
                      lam: float = 0.94) -> pd.DataFrame:
    """Return a frame with columns '<model> VaR', '<model> ES' and 'return',
    indexed by forecast day, starting once `window` past returns exist."""
    x = r.to_numpy()
    T = len(x)
    s2 = ewma_variance(x, lam)
    sig = np.sqrt(s2)
    z = x / sig  # standardised residuals (NaN in burn-in)

    rows = []
    # windows ending at t-1 for forecast day t, t = window .. T-1
    W = sliding_window_view(x, window)[:-1]           # W[k] = x[k : k+window], forecasts day k+window
    Z = sliding_window_view(z, window)[:-1]
    days = np.arange(window, T)

    out = {"return": x[days]}
    # Historical simulation
    q = np.quantile(W, alpha, axis=1)
    es = np.array([-w[w <= qq].mean() for w, qq in zip(W, q)])
    out["Historical sim VaR"], out["Historical sim ES"] = -q, es
    # Normal (rolling mean and sd)
    mu, sd = W.mean(axis=1), W.std(axis=1, ddof=1)
    zq = stats.norm.ppf(alpha)
    out["Normal VaR"] = -(mu + sd * zq)
    out["Normal ES"] = -(mu - sd * stats.norm.pdf(zq) / alpha)
    # EWMA RiskMetrics, zero mean
    sd_e = sig[days]
    out["EWMA (0.94) VaR"] = -sd_e * zq
    out["EWMA (0.94) ES"] = sd_e * stats.norm.pdf(zq) / alpha
    # Filtered HS: empirical quantile of past standardised residuals, rescaled
    # by today's EWMA volatility forecast
    zq_emp = np.full(len(days), np.nan)
    zes = np.full(len(days), np.nan)
    for k, zw in enumerate(Z):
        zw = zw[~np.isnan(zw)]
        qz = np.quantile(zw, alpha)
        zq_emp[k] = qz
        zes[k] = zw[zw <= qz].mean()
    out["Filtered HS VaR"] = -sd_e * zq_emp
    out["Filtered HS ES"] = -sd_e * zes
    return pd.DataFrame(out, index=r.index[days])


# --------------------------------------------------------------------------- #
# Backtests                                                                    #
# --------------------------------------------------------------------------- #
def _xlogy(x, y):
    return 0.0 if x == 0 else x * np.log(y)


def kupiec_pof(n_exc: int, T: int, p: float) -> tuple[float, float]:
    """Kupiec (1995) proportion-of-failures LR test. Returns (LR, p-value), LR ~ chi2(1)."""
    x = n_exc
    phat = x / T
    ll0 = _xlogy(T - x, 1 - p) + _xlogy(x, p)
    ll1 = _xlogy(T - x, 1 - phat) + _xlogy(x, phat)
    lr = -2 * (ll0 - ll1)
    return float(lr), float(stats.chi2.sf(lr, 1))


def christoffersen_ind(hits: np.ndarray) -> tuple[float, float]:
    """Christoffersen (1998) independence LR test on a 0/1 exception series.
    Returns (LR, p-value), LR ~ chi2(1)."""
    h = np.asarray(hits).astype(int)
    a, b = h[:-1], h[1:]
    n00 = int(np.sum((a == 0) & (b == 0)))
    n01 = int(np.sum((a == 0) & (b == 1)))
    n10 = int(np.sum((a == 1) & (b == 0)))
    n11 = int(np.sum((a == 1) & (b == 1)))
    pi01 = n01 / (n00 + n01) if (n00 + n01) else 0.0
    pi11 = n11 / (n10 + n11) if (n10 + n11) else 0.0
    pi = (n01 + n11) / (n00 + n01 + n10 + n11)
    ll0 = _xlogy(n00 + n10, 1 - pi) + _xlogy(n01 + n11, pi)
    ll1 = _xlogy(n00, 1 - pi01) + _xlogy(n01, pi01) + _xlogy(n10, 1 - pi11) + _xlogy(n11, pi11)
    lr = -2 * (ll0 - ll1)
    return float(lr), float(stats.chi2.sf(lr, 1))


def basel_zone(n_exc_250: int) -> str:
    """Basel Committee (1996) traffic light for 99% VaR over 250 days."""
    if n_exc_250 <= 4:
        return "green"
    if n_exc_250 <= 9:
        return "yellow"
    return "red"
