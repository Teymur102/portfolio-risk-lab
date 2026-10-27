import numpy as np
import pytest
from scipy import stats

from prl import optimize as opt
from prl import var as vr
from prl.metrics import block_bootstrap_sharpe_diff, jobson_korkie_memmel


@pytest.fixture
def cov():
    rng = np.random.default_rng(1)
    # full-rank correlation matrix with a mix of positive and negative entries
    X = rng.normal(size=(500, 6)) @ rng.normal(size=(6, 6))
    corr = np.corrcoef(X, rowvar=False)
    vols = np.array([0.15, 0.05, 0.20, 0.10, 0.18, 0.07]) / np.sqrt(252)
    return np.outer(vols, vols) * corr


# ----------------------------------------------------------------- risk
def test_risk_contributions_sum_to_portfolio_vol(cov):
    w = np.array([0.3, 0.1, 0.2, 0.15, 0.15, 0.1])
    rc = opt.risk_contributions(w, cov)
    assert rc.sum() == pytest.approx(opt.portfolio_vol(w, cov), rel=1e-12)


def test_risk_parity_equalises_contributions(cov):
    w = opt.risk_parity(cov)
    rc = opt.risk_contributions(w, cov)
    assert w.sum() == pytest.approx(1.0)
    assert np.all(w > 0)
    assert np.max(rc) / np.min(rc) == pytest.approx(1.0, abs=1e-5)


def test_min_variance_is_long_only_and_beats_equal_weight(cov):
    w = opt.min_variance(cov)
    assert w.sum() == pytest.approx(1.0)
    assert np.all(w >= -1e-12)
    ew = np.full(6, 1 / 6)
    assert w @ cov @ w <= ew @ cov @ ew


def test_min_variance_two_asset_closed_form():
    s1, s2, rho = 0.2, 0.1, 0.3
    c = np.array([[s1**2, rho * s1 * s2], [rho * s1 * s2, s2**2]])
    w1 = (s2**2 - rho * s1 * s2) / (s1**2 + s2**2 - 2 * rho * s1 * s2)
    assert opt.min_variance(c)[0] == pytest.approx(w1, abs=1e-6)


def test_max_sharpe_matches_unconstrained_tangency_when_interior(cov):
    mu = np.array([0.06, 0.03, 0.08, 0.04, 0.07, 0.035]) / 252
    w, fb = opt.max_sharpe(mu, cov)
    t = np.linalg.solve(cov, mu)
    t = t / t.sum()
    if np.all(t > 0):  # only comparable when the long-only bound is not binding
        np.testing.assert_allclose(w, t, atol=1e-5)
    assert not fb and w.sum() == pytest.approx(1.0)


def test_ledoit_wolf_properties():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(60, 10))
    S_lw, d = opt.ledoit_wolf(X)
    assert 0 <= d <= 1
    assert np.all(np.linalg.eigvalsh(S_lw) > 0)
    Xc = X - X.mean(0)
    S = Xc.T @ Xc / len(X)
    np.testing.assert_allclose(np.trace(S_lw), np.trace(S))  # shrinkage preserves the trace
    # few observations relative to dimension -> substantial shrinkage
    assert d > 0.2


# ------------------------------------------------------------------ VaR
def test_hs_var_is_empirical_quantile():
    sample = np.arange(-50, 50) / 1000.0  # -0.050 ... 0.049, 100 obs
    var, es = vr.hs_var_es(sample, 0.05)
    assert var == pytest.approx(-np.quantile(sample, 0.05))
    assert var == pytest.approx(0.04505)  # numpy linear interpolation between -0.046 and -0.045
    assert es == pytest.approx(-np.mean(sample[:5]))  # the five returns below the quantile


def test_normal_var_es_known_values():
    var, es = vr.normal_var_es(0.0, 0.01, 0.01)
    assert var == pytest.approx(0.01 * 2.326347874, rel=1e-8)
    assert es == pytest.approx(0.01 * 2.665214220, rel=1e-8)


def test_hs_var_converges_to_normal_quantile():
    rng = np.random.default_rng(3)
    x = rng.normal(0, 0.01, 400_000)
    assert vr.hs_var_es(x, 0.01)[0] == pytest.approx(0.02326, rel=0.01)


def test_ewma_recursion():
    r = np.array([0.01, -0.02, 0.0, 0.03, -0.01])
    s2 = vr.ewma_variance(r, 0.94, init_window=2)
    assert s2[2] == pytest.approx(np.mean(r[:2] ** 2))
    assert s2[3] == pytest.approx(0.94 * s2[2] + 0.06 * r[2] ** 2)
    assert s2[4] == pytest.approx(0.94 * s2[3] + 0.06 * r[3] ** 2)


# ------------------------------------------------------------- backtests
def test_kupiec_zero_when_rate_matches():
    lr, p = vr.kupiec_pof(10, 1000, 0.01)
    assert lr == pytest.approx(0.0, abs=1e-10)
    assert p == pytest.approx(1.0)


def test_kupiec_known_case():
    # T=1000, p=1%, 20 exceptions. Hand computation:
    # LR = -2[980 ln .99 + 20 ln .01] + 2[980 ln .98 + 20 ln .02]
    #    = -2[-9.84933 - 92.10340] + 2[-19.79865 - 78.24046] = 7.8272
    lr, p = vr.kupiec_pof(20, 1000, 0.01)
    assert lr == pytest.approx(7.8272, abs=1e-3)
    assert p == pytest.approx(stats.chi2.sf(7.8272, 1), rel=1e-3)
    assert p < 0.01  # twice the expected exceptions is rejected at 1%


def test_kupiec_zero_exceptions():
    lr, _ = vr.kupiec_pof(0, 250, 0.01)
    assert lr == pytest.approx(-2 * 250 * np.log(0.99))


def test_christoffersen_detects_clustering():
    clustered = np.zeros(1000, int)
    clustered[100:110] = 1
    spread = np.zeros(1000, int)
    spread[::100] = 1
    assert vr.christoffersen_ind(clustered)[1] < 0.001
    assert vr.christoffersen_ind(spread)[1] > 0.05


def test_basel_zones():
    assert [vr.basel_zone(n) for n in (0, 4, 5, 9, 10)] == ["green", "green", "yellow", "yellow", "red"]


# ---------------------------------------------------------- Sharpe tests
def test_sharpe_tests_basic_properties():
    rng = np.random.default_rng(5)
    x = rng.normal(0.0003, 0.01, 2000)
    y = 0.5 * x + rng.normal(0.0, 0.01, 2000)
    z_xy, p_xy = jobson_korkie_memmel(x, y)
    z_yx, p_yx = jobson_korkie_memmel(y, x)
    assert z_xy == pytest.approx(-z_yx) and p_xy == pytest.approx(p_yx)  # antisymmetric
    bb = block_bootstrap_sharpe_diff(x, x, n_boot=200)
    assert bb["diff"] == pytest.approx(0.0) and bb["ci_low"] == pytest.approx(0.0)


def test_sharpe_tests_detect_a_large_true_difference():
    rng = np.random.default_rng(6)
    common = rng.normal(0, 0.01, 5000)
    a = common + 0.002 + rng.normal(0, 0.003, 5000)   # very high Sharpe
    b = common + rng.normal(0, 0.003, 5000)            # ~zero Sharpe
    assert jobson_korkie_memmel(a, b)[1] < 0.001
    assert block_bootstrap_sharpe_diff(a, b, n_boot=500)["p_value"] < 0.01
