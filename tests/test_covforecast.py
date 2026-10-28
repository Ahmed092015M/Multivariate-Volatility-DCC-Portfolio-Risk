"""
Unit tests for the covforecast package (Project 4).

Each test checks one function against a known answer: a closed form, scipy,
pandas, a hand-worked example from notes/theory.md, or a simulation with known
parameters and a fixed seed. Run from the repo root with:  pytest -v
"""
import numpy as np
import pandas as pd
import pytest
from scipy import stats

from covforecast.covariance import cov_to_corr, ewma_cov, pair_series, portfolio_variance, rolling_cov
from covforecast.data import eur_returns, portfolio_returns, return_index, total_return
from covforecast.evaluation import diebold_mariano, qlike_matrix, qlike_portfolio
from covforecast.mgarch import (ccc_cov, dcc_cov, dcc_correlations, dcc_loglik, fit_dcc, qbar,
                                rolling_ccc_dcc)
from covforecast.portfolio import min_variance_weights, turnover
from covforecast.risk import (acerbi_szekely_z2, christoffersen_test, kupiec_test, var_es_fhs,
                              var_es_normal)
from covforecast.univariate import fit_gjr_t, gjr_t_loglik, gjr_variance

H2 = np.array([[1.44, -0.18], [-0.18, 0.25]])          # theory example: stocks 1.2%, bonds 0.5%, rho -0.3


def _returns(T=800, seed=0):
    """Correlated fat-tailed returns for 3 assets."""
    rng = np.random.default_rng(seed)
    A = np.array([[1.0, 0.5, -0.2], [0.0, 0.8, 0.1], [0.0, 0.0, 0.5]])
    return pd.DataFrame(rng.standard_t(6, (T, 3)) @ A, columns=["a", "b", "c"],
                        index=pd.bdate_range("2010-01-01", periods=T))


def _simulate_dcc(T, a, b, Qb, seed=1):
    """Standardised residuals from a DCC process with known parameters."""
    g = np.random.default_rng(seed)
    E, Q = np.zeros((T, len(Qb))), Qb.copy()
    for t in range(T):
        if t > 0:
            e = E[t - 1][:, None]
            Q = (1 - a - b) * Qb + a * (e @ e.T) + b * Q
        E[t] = np.linalg.cholesky(cov_to_corr(Q)) @ g.standard_normal(len(Qb))
    return E


# --------------------------------------------------------------------------- #
# Step 1 - data                                                               #
# --------------------------------------------------------------------------- #
def test_total_return_adds_back_payout():
    close = pd.Series([100.0, 101.0, 98.0, 99.0], index=pd.bdate_range("2020-01-01", periods=4))
    payouts = pd.Series([3.0], index=[close.index[2]])                 # 3 EUR paid on day 3
    tr = total_return(close, payouts)
    assert tr.iloc[1] == pytest.approx(100 * ((98 + 3) / 101 - 1))
    assert tr.iloc[0] == pytest.approx(1.0) and tr.iloc[2] == pytest.approx(100 * (99 / 98 - 1))


def test_return_index_reproduces_returns():
    r = pd.Series([1.0, -2.0, 0.5], index=pd.bdate_range("2020-01-02", periods=3))
    idx = return_index(r, first_date="2020-01-01")
    assert idx.iloc[0] == 100 and np.allclose(100 * idx.pct_change().dropna(), r)


def test_eur_returns_constant_fx_and_log_identity():
    rng = np.random.default_rng(0)
    i = pd.bdate_range("2020-01-01", periods=200)
    prices = pd.DataFrame({"EQ": 100 * np.exp(np.cumsum(0.01 * rng.standard_normal(200))),
                           "US": 50 * np.exp(np.cumsum(0.01 * rng.standard_normal(200)))}, index=i)
    fx = pd.DataFrame({"EURUSD": 1.1 * np.exp(np.cumsum(0.005 * rng.standard_normal(200)))}, index=i)
    Y = eur_returns(prices, fx, usd_assets=["US"])
    log_diff = 100 * (np.log(prices["US"]).diff() - np.log(fx["EURUSD"]).diff()).dropna()
    assert np.allclose(100 * np.log(1 + Y["US"] / 100), log_diff)
    fx["EURUSD"] = 1.1
    assert np.allclose(eur_returns(prices, fx, ["US"])["US"], 100 * prices["US"].pct_change().dropna())


def test_portfolio_returns_and_weight_check():
    Y = _returns(100)
    Rp = portfolio_returns(Y, {"a": 0.5, "b": 0.3, "c": 0.2})
    assert np.allclose(Rp, 0.5 * Y["a"] + 0.3 * Y["b"] + 0.2 * Y["c"])
    with pytest.raises(ValueError):
        portfolio_returns(Y, {"a": 0.5, "b": 0.5, "c": 0.5})


# --------------------------------------------------------------------------- #
# Steps 2-3 - covariance basics                                               #
# --------------------------------------------------------------------------- #
def test_portfolio_variance_closed_form_and_stack():
    w = np.array([0.6, 0.4])
    assert portfolio_variance(H2, w) == pytest.approx(0.472)
    assert np.allclose(portfolio_variance(np.stack([H2, 2 * H2]), w), [0.472, 0.944])


def test_rolling_cov_matches_direct_product_and_identity():
    Y = _returns(400)
    H = rolling_cov(Y, window=250)
    X = Y.to_numpy()[50:300]
    assert np.allclose(H[300], X.T @ X / 250) and np.isnan(H[249]).all()
    w = np.array([0.5, 0.3, 0.2])
    Rp = Y.to_numpy() @ w
    direct = pd.Series(Rp ** 2).rolling(250).mean().shift(1).to_numpy()
    assert np.allclose(portfolio_variance(H, w)[250:], direct[250:])


def test_ewma_cov_matches_pandas_and_is_positive_definite():
    Y = _returns(900)
    H = ewma_cov(Y, lam=0.94)
    pandas_cov = (Y["a"] * Y["c"]).ewm(alpha=0.06, adjust=False).mean().shift(1)
    assert np.allclose(H[600:, 0, 2], pandas_cov.iloc[600:])
    assert np.linalg.eigvalsh(H).min() > 0


def test_cov_to_corr_and_pair_series():
    R = cov_to_corr(H2)
    assert np.allclose(np.diag(R), 1) and R[0, 1] == pytest.approx(-0.3)
    s = pair_series(np.stack([H2, H2]), ["eq", "bd"], "eq", "bd", pd.bdate_range("2020", periods=2))
    assert list(s) == [-0.18, -0.18] and s.name == "eq-bd"


# --------------------------------------------------------------------------- #
# Step 4 - univariate GJR and CCC                                             #
# --------------------------------------------------------------------------- #
def test_gjr_variance_one_step_and_leverage():
    y = pd.Series([-2.0, 2.0, 0.0])
    s2 = gjr_variance(y, 0.02, 0.03, 0.10, 0.90, sigma2_start=1.0)
    assert s2.iloc[1] == pytest.approx(0.02 + 0.13 * 4 + 0.90)        # after a fall: alpha + gamma
    assert s2.iloc[2] == pytest.approx(0.02 + 0.03 * 4 + 0.90 * s2.iloc[1])   # after a rise: alpha only


def test_gjr_t_loglik_equals_scipy_and_fit_recovers_parameters():
    g = np.random.default_rng(4)
    T, (om, al, ga, be, nu) = 6000, (0.02, 0.02, 0.12, 0.90, 7.0)
    y, s2 = np.zeros(T), np.full(T, om / (1 - al - ga / 2 - be))
    for t in range(1, T):
        s2[t] = om + (al + ga * (y[t - 1] < 0)) * y[t - 1] ** 2 + be * s2[t - 1]
        y[t] = np.sqrt(s2[t]) * np.sqrt((nu - 2) / nu) * g.standard_t(nu)
    y = pd.Series(y)
    p = (om, al, ga, be, nu)
    sig2 = gjr_variance(y, *p[:4]).to_numpy()[1:]
    ref = stats.t.logpdf(y.to_numpy()[1:], df=nu, scale=np.sqrt(sig2 * (nu - 2) / nu)).sum()
    assert gjr_t_loglik(p, y) == pytest.approx(ref, rel=1e-10)
    fit = fit_gjr_t(y)
    assert fit["converged"] and abs(fit["gamma"] - ga) < 0.05 and abs(fit["beta"] - be) < 0.04


def test_ccc_cov_diagonal_and_correlations():
    sig2 = np.array([[1.0, 4.0], [2.0, 0.5]])
    H = ccc_cov(sig2, cov_to_corr(H2))
    assert np.allclose(np.diagonal(H, axis1=1, axis2=2), sig2)
    assert np.allclose(cov_to_corr(H), cov_to_corr(H2))


# --------------------------------------------------------------------------- #
# Step 5 - DCC                                                                #
# --------------------------------------------------------------------------- #
def test_dcc_theory_example():
    E = np.array([[-3.0, -2.5], [0.0, 0.0]])
    R = dcc_correlations(E, 0.05, 0.93, Q_bar=np.array([[1.0, 0.5], [0.5, 1.0]]))
    assert R[1, 0, 1] == pytest.approx(0.6394, abs=1e-4)


def test_dcc_zero_parameters_equals_ccc():
    E = _returns(300).to_numpy()
    assert np.allclose(dcc_correlations(E, 0.0, 0.0), cov_to_corr(qbar(E)))


def test_dcc_cov_equals_ccc_with_constant_correlation():
    sig2 = np.array([[1.0, 4.0], [2.0, 0.5]])
    R = np.stack([cov_to_corr(H2)] * 2)
    assert np.allclose(dcc_cov(sig2, R), ccc_cov(sig2, cov_to_corr(H2)))


def test_dcc_loglik_equals_scipy_plus_constant():
    E = _returns(150).to_numpy() / 1.5
    Qb = qbar(E)
    R = dcc_correlations(E, 0.03, 0.95, Q_bar=Qb)
    ref = sum(stats.multivariate_normal(np.zeros(3), R[t]).logpdf(E[t]) for t in range(150))
    const = 0.5 * np.sum(3 * np.log(2 * np.pi) + np.sum(E ** 2, axis=1))
    assert dcc_loglik((0.03, 0.95), E, Q_bar=Qb) == pytest.approx(ref + const, rel=1e-10)
    assert dcc_loglik((0.5, 0.6), E) == -1e10                                   # a + b >= 1 refused


def test_fit_dcc_recovers_simulated_parameters():
    Qb = np.array([[1.0, 0.6, -0.3], [0.6, 1.0, -0.2], [-0.3, -0.2, 1.0]])
    fit = fit_dcc(_simulate_dcc(4000, 0.05, 0.93, Qb))
    assert fit["converged"] and abs(fit["a"] - 0.05) < 0.015 and abs(fit["b"] - 0.93) < 0.03


def test_rolling_ccc_dcc_shape_and_no_look_ahead():
    Y = _returns(560)
    Hc, Hd, ps = rolling_ccc_dcc(Y, window=500, refit_every=20)
    assert Hc.shape == (60, 3, 3) and len(ps) == 3
    assert np.allclose(np.diagonal(Hc, axis1=1, axis2=2), np.diagonal(Hd, axis1=1, axis2=2))
    fake = Y.copy()
    fake.iloc[530, 0] = -20.0                                                   # out-of-sample day 30
    _, Hd_f, _ = rolling_ccc_dcc(fake, window=500, refit_every=20)
    assert np.array_equal(Hd[:31], Hd_f[:31]) and not np.allclose(Hd[31], Hd_f[31])


# --------------------------------------------------------------------------- #
# Step 8 - evaluation                                                         #
# --------------------------------------------------------------------------- #
def test_qlike_matrix_one_asset_and_scipy():
    Y = _returns(50).to_numpy()
    H = np.stack([np.cov(Y.T)] * 50)
    ref = np.array([stats.multivariate_normal(np.zeros(3), H[t]).logpdf(Y[t]) for t in range(50)])
    assert np.allclose(-0.5 * (qlike_matrix(H, Y) + 3 * np.log(2 * np.pi)), ref)
    h1 = H[:, :1, :1]
    assert np.allclose(qlike_matrix(h1, Y[:, :1]), qlike_portfolio(h1[:, 0, 0], Y[:, 0]))


def test_diebold_mariano_sign_and_identical_losses():
    g = np.random.default_rng(5)
    base = g.chisquare(1, 3000)
    dm, p = diebold_mariano(base, base + 0.5 + 0.1 * g.standard_normal(3000))
    assert dm < -3 and p < 0.01                                                 # model A clearly better
    assert np.isnan(diebold_mariano(base, base)[0])


# --------------------------------------------------------------------------- #
# Step 9 - risk                                                               #
# --------------------------------------------------------------------------- #
def test_var_es_normal_multipliers():
    v, e = var_es_normal(1.0)
    assert float(v) == pytest.approx(2.3263, abs=1e-4) and float(e) == pytest.approx(2.3378, abs=1e-4)


def test_var_es_fhs_converges_and_no_look_ahead():
    z = np.random.default_rng(3).standard_normal(20000)
    v, e = var_es_fhs(z, np.ones(20000))
    assert abs(v[-1] - 2.326) < 0.05 and abs(e[-1] - 2.338) < 0.05 and np.isnan(v[:250]).all()
    z2 = z.copy()
    z2[5000] = -50.0
    v2, e2 = var_es_fhs(z2, np.ones(20000))
    assert v2[5000] == v[5000] and e2[5001] != e[5001]


def test_kupiec_and_christoffersen_hand_examples():
    h = np.zeros(250, dtype=bool)
    h[:6] = True
    x, p = kupiec_test(h, 0.01)
    assert x == 6 and p == pytest.approx(0.0594, abs=1e-3)
    h = np.zeros(250, dtype=bool)
    h[[10, 11, 100, 101, 150, 200]] = True
    n11, p_cc = christoffersen_test(h, 0.01)
    assert n11 == 2 and p_cc == pytest.approx(0.0029, abs=5e-4)


def test_acerbi_szekely_theory_example():
    r = np.zeros(1000)
    r[:30] = -3.6
    assert acerbi_szekely_z2(r, np.full(1000, 2.5), np.full(1000, 3.0)) == pytest.approx(-0.44)


# --------------------------------------------------------------------------- #
# Step 10 - portfolios                                                        #
# --------------------------------------------------------------------------- #
def test_min_variance_closed_form_and_optimality():
    w = min_variance_weights(H2)
    s1, s2, s12 = H2[0, 0], H2[1, 1], H2[0, 1]
    assert w[0] == pytest.approx((s2 - s12) / (s1 + s2 - 2 * s12)) and w.sum() == pytest.approx(1)
    H = np.cov(_returns(500).to_numpy().T)
    wm = min_variance_weights(H)
    for other in [np.ones(3) / 3, np.array([0.5, 0.2, 0.3])]:
        assert portfolio_variance(H, wm) <= portfolio_variance(H, other)


def test_turnover():
    W = np.array([[0.5, 0.5], [0.6, 0.4], [0.6, 0.4]])
    t = turnover(W)
    assert np.isnan(t[0]) and t[1] == pytest.approx(0.2) and t[2] == 0.0


def test_plots_return_figures():
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib.figure import Figure
    from covforecast.plots import plot_correlations, plot_mv_tradeoff, plot_scores, plot_var_backtest

    m = ["Rolling 250", "EWMA", "CCC", "DCC"]
    s = pd.Series([0.1, -0.2], index=pd.bdate_range("2020", periods=2))
    scores = pd.DataFrame({"matrix QLIKE": [1.0, 0.7, 0.4, 0.3], "portfolio QLIKE": [0.0, -0.2, -0.25, -0.26],
                           "DM vs DCC (matrix)": [3.8, 6.8, 7.0, np.nan]}, index=m)
    bt = pd.DataFrame({(mm, t): {"breaches 99%": 40, "avg VaR 99% (EUR k)": 150} for mm in m
                       for t in ["normal", "FHS"]}).T
    mv = pd.DataFrame({"realised vol % (annual)": [4.4] * 4, "avg turnover % / day": [1, 8, 5, 6]}, index=m)
    costs = pd.DataFrame({(bp, mm): {"annual return % (net)": 2.0 - bp / 10} for bp in [0, 5] for mm in m}).T
    figs = [plot_correlations({"x": s}, "t"), plot_scores(scores), plot_var_backtest(bt, 33.5),
            plot_mv_tradeoff(mv, costs)]
    assert all(isinstance(f, Figure) for f in figs)
