"""Covariance estimation and long-only, fully-invested portfolio construction."""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize


# --------------------------------------------------------------------------- #
# Ledoit-Wolf (2004) shrinkage toward a scaled identity                        #
# --------------------------------------------------------------------------- #
def ledoit_wolf(returns: np.ndarray) -> tuple[np.ndarray, float]:
    """Return (shrunk covariance, shrinkage intensity delta in [0, 1]).

    Implements Ledoit & Wolf (2004), "A well-conditioned estimator for
    large-dimensional covariance matrices": Sigma = delta*mu*I + (1-delta)*S,
    with S the (1/T) sample covariance of demeaned returns.
    """
    X = np.asarray(returns, dtype=float)
    T, n = X.shape
    X = X - X.mean(axis=0)
    S = X.T @ X / T
    mu = np.trace(S) / n
    F = mu * np.eye(n)
    d2 = np.sum((S - F) ** 2)  # ||S - mu I||_F^2
    # b2_bar = (1/T^2) sum_t ||x_t x_t' - S||_F^2, expanded so it is O(T n):
    # sum_t ||x_t x_t' - S||^2 = sum_t ||x_t||^4 - T ||S||^2
    b2_bar = (np.sum(np.sum(X**2, axis=1) ** 2) - T * np.sum(S**2)) / T**2
    b2 = min(b2_bar, d2)
    delta = b2 / d2 if d2 > 0 else 1.0
    return delta * F + (1 - delta) * S, float(delta)


# --------------------------------------------------------------------------- #
# Risk decomposition                                                           #
# --------------------------------------------------------------------------- #
def portfolio_vol(w: np.ndarray, cov: np.ndarray) -> float:
    return float(np.sqrt(w @ cov @ w))


def risk_contributions(w: np.ndarray, cov: np.ndarray) -> np.ndarray:
    """Euler risk contributions RC_i = w_i (Sigma w)_i / sigma_p; they sum to sigma_p."""
    sig = portfolio_vol(w, cov)
    return w * (cov @ w) / sig


# --------------------------------------------------------------------------- #
# Optimisers (all long-only, weights sum to one)                               #
# --------------------------------------------------------------------------- #
def _simplex_constraints(n):
    return [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0, "jac": lambda w: np.ones(n)}]


def min_variance(cov: np.ndarray) -> np.ndarray:
    n = cov.shape[0]
    res = minimize(
        lambda w: w @ cov @ w,
        np.full(n, 1.0 / n),
        jac=lambda w: 2 * cov @ w,
        bounds=[(0.0, 1.0)] * n,
        constraints=_simplex_constraints(n),
        method="SLSQP",
        options={"ftol": 1e-14, "maxiter": 1000},
    )
    w = np.clip(res.x, 0, None)
    return w / w.sum()


def max_sharpe(mu_excess: np.ndarray, cov: np.ndarray) -> tuple[np.ndarray, bool]:
    """Tangency portfolio, long-only.

    Uses the convex reformulation  min y'Sigma y  s.t. mu'y = 1, y >= 0,
    w = y / sum(y). If no asset has a positive expected excess return the
    problem has no positive-Sharpe solution; we then fall back to minimum
    variance and return fallback=True.
    """
    n = cov.shape[0]
    if np.max(mu_excess) <= 0:
        return min_variance(cov), True
    y0 = np.where(mu_excess > 0, 1.0, 0.0)
    y0 = y0 / (mu_excess @ y0)
    res = minimize(
        lambda y: y @ cov @ y,
        y0,
        jac=lambda y: 2 * cov @ y,
        bounds=[(0.0, None)] * n,
        constraints=[{"type": "eq", "fun": lambda y: mu_excess @ y - 1.0, "jac": lambda y: mu_excess}],
        method="SLSQP",
        options={"ftol": 1e-16, "maxiter": 1000},
    )
    y = np.clip(res.x, 0, None)
    return y / y.sum(), False


def risk_parity(cov: np.ndarray) -> np.ndarray:
    """Equal-risk-contribution weights via the convex log-barrier problem
    min 0.5 y'Sigma y - (1/n) sum log y  (Spinu 2013); w = y / sum(y)."""
    n = cov.shape[0]
    b = np.full(n, 1.0 / n)
    C = cov / np.mean(np.diag(cov))  # ERC weights are scale-invariant; rescale for conditioning
    y0 = 1.0 / np.sqrt(np.diag(C)) / n
    res = minimize(
        lambda y: 0.5 * y @ C @ y - b @ np.log(y),
        y0,
        jac=lambda y: C @ y - b / y,
        bounds=[(1e-12, None)] * n,
        method="L-BFGS-B",
        options={"ftol": 1e-15, "gtol": 1e-12, "maxiter": 5000},
    )
    y = res.x
    return y / y.sum()
