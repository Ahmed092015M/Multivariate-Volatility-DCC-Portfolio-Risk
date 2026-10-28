"""
Steps 4-7 - Multivariate GARCH: CCC and DCC, and rolling out-of-sample forecasts.

Both models use one univariate GJR-GARCH per asset for the volatilities D_t and
model the correlations of the standardised residuals eps_t = D_t^{-1} y_t:

    CCC: R_t = R_bar (constant)
    DCC: R_t from the Q_t recursion
    and H_t = D_t R_t D_t.

Sections
--------
1. Long-run target and CCC     qbar, ccc_cov                       (theory B4, Equation 9)
2. DCC                         dcc_correlations, dcc_cov,          (theory B5, Equations 10-13)
                               dcc_loglik, fit_dcc
3. Out-of-sample forecasts     rolling_ccc_dcc                     (Step 7)
"""
import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .covariance import cov_to_corr
from .univariate import fit_gjr_t, gjr_variance

__all__ = ["qbar", "ccc_cov", "dcc_correlations", "dcc_cov", "dcc_loglik", "fit_dcc",
           "rolling_ccc_dcc"]


# =========================================================================== #
# 1. Long-run target and CCC                                                  #
# =========================================================================== #
def qbar(eps):
    """Unconditional covariance of the standardised residuals (zero mean).

        Q_bar = (1/T) * sum_t eps_t eps_t'

    Its correlation version is the CCC matrix R_bar, and Q_bar itself is the
    long-run target of the DCC recursion (correlation targeting).
    """
    E = np.asarray(eps, dtype=float)
    return E.T @ E / len(E)


def ccc_cov(sigma2, R_bar):
    """CCC covariance forecast:  H_t = D_t R_bar D_t      (Equation 9).

    Parameters
    ----------
    sigma2 : pd.DataFrame or array (T, N)
        Univariate variance forecasts, one column per asset.
    R_bar : array (N, N)
        Constant correlation matrix.

    Returns
    -------
    array (T, N, N)
    """
    sd = np.sqrt(np.asarray(sigma2, dtype=float))                        # volatilities, (T, N)
    return sd[:, :, None] * np.asarray(R_bar)[None, :, :] * sd[:, None, :]   # h_ij = s_i rho_ij s_j


# =========================================================================== #
# 2. DCC                                                                      #
# =========================================================================== #
def dcc_correlations(eps, a, b, Q_bar=None):
    """DCC conditional correlation matrices      (Equations 10-11).

        Q_t = (1 - a - b) * Q_bar + a * eps_{t-1} eps_{t-1}' + b * Q_{t-1}
        R_t = diag(Q_t)^{-1/2} Q_t diag(Q_t)^{-1/2}

    A GARCH(1,1) for the whole correlation structure: a reaction a to
    yesterday's joint shock, a persistence b, and mean reversion to Q_bar.
    With a = b = 0, DCC reduces exactly to CCC.

    Parameters
    ----------
    eps : pd.DataFrame or array (T, N)
        Standardised residuals.
    a, b : float
        Reaction and persistence (a, b >= 0, a + b < 1).
    Q_bar : array (N, N), optional
        Long-run target; default = qbar(eps).

    Returns
    -------
    array (T, N, N)
        Correlation forecast for each day (built with residuals up to t-1).
    """
    E = np.asarray(eps, dtype=float)
    T, N = E.shape
    Qb = qbar(E) if Q_bar is None else np.asarray(Q_bar, dtype=float)
    Q = np.empty((T, N, N))
    Q[0] = Qb                                                  # start at the long-run level
    for t in range(1, T):
        e = E[t - 1][:, None]                                  # yesterday's shocks, column vector
        Q[t] = (1 - a - b) * Qb + a * (e @ e.T) + b * Q[t - 1] # Equation 10
    return cov_to_corr(Q)                                      # Equation 11: rescale to correlations


def dcc_cov(sigma2, R):
    """Covariance from daily volatilities and daily correlations:  H_t = D_t R_t D_t  (Equation 11).

    Parameters
    ----------
    sigma2 : pd.DataFrame or array (T, N)
        Univariate variance forecasts.
    R : array (T, N, N)
        Correlation matrices, one per day (e.g. from dcc_correlations).

    Returns
    -------
    array (T, N, N)
    """
    sd = np.sqrt(np.asarray(sigma2, dtype=float))
    return sd[:, :, None] * np.asarray(R) * sd[:, None, :]


def dcc_loglik(params, eps, Q_bar=None):
    """Correlation part of the Gaussian DCC log-likelihood      (Equation 12).

        l_C(a, b) = -1/2 * sum_t [ log|R_t| + eps_t' R_t^{-1} eps_t - eps_t' eps_t ]

    The full Gaussian log-likelihood of y_t splits into a volatility part
    (estimated per asset in step 1) and this correlation part. The eps'eps term
    does not depend on (a, b); it makes l_C = 0 when R_t = I.

    Returns -1e10 for impossible parameters (a < 0, b < 0 or a + b >= 1).
    """
    a, b = params
    if a < 0 or b < 0 or a + b >= 1:
        return -1e10
    E = np.asarray(eps, dtype=float)
    R = dcc_correlations(E, a, b, Q_bar)
    _, logdet = np.linalg.slogdet(R)                                              # log|R_t| for every day
    quad = np.einsum("ti,ti->t", E, np.linalg.solve(R, E[:, :, None])[:, :, 0])   # eps' R^{-1} eps
    return -0.5 * np.sum(logdet + quad - np.sum(E ** 2, axis=1))


def fit_dcc(eps, start=(0.02, 0.95), Q_bar=None):
    """Second-step DCC estimate of (a, b) by maximum likelihood (SLSQP).

    Parameters
    ----------
    eps : pd.DataFrame or array (T, N)
        Standardised residuals from the univariate step.
    start : tuple
        Starting values (a, b).
    Q_bar : array (N, N), optional
        Long-run target; default = qbar(eps).

    Returns
    -------
    dict
        a, b, loglik, converged.
    """
    res = minimize(lambda p: -dcc_loglik(p, eps, Q_bar), x0=np.array(start), method="SLSQP",
                   bounds=[(0.0, 1.0), (0.0, 1.0)],
                   constraints=[{"type": "ineq", "fun": lambda p: 0.9999 - p[0] - p[1]}])   # a + b < 1
    a, b = res.x
    return {"a": a, "b": b, "loglik": -res.fun, "converged": res.success}


# =========================================================================== #
# 3. Out-of-sample forecasts                                                  #
# =========================================================================== #
def rolling_ccc_dcc(Y, window=1000, refit_every=20):
    """Out-of-sample CCC and DCC covariance forecasts with a rolling estimation window.

    For each block of `refit_every` days starting at day s:
      1. fit GJR-GARCH-t per asset on days s-window ... s-1;
      2. run each variance through the window and the block (start = mean y^2 of the window);
      3. residuals eps = y / sigma; Q_bar and R_bar from the window only;
      4. fit DCC (a, b) on the window residuals;
      5. run the DCC recursion through the window and the block;
      6. keep H_t = D_t R_t D_t on the block days (CCC: R_t = R_bar).

    No look-ahead: parameters only see days before the block, and every
    forecast for day t uses returns up to t-1.

    Parameters
    ----------
    Y : pd.DataFrame
        Daily returns in %, one column per asset.
    window : int
        Estimation window (days).
    refit_every : int
        Re-estimation frequency (days).

    Returns
    -------
    H_ccc, H_dcc : arrays (T - window, N, N)
        Forecasts for days window ... T-1 (dates: Y.index[window:]).
    params : pd.DataFrame
        One row per re-estimation: DCC a, b, convergence, each asset's GJR persistence.
    """
    H_ccc, H_dcc, params = [], [], []
    for s in range(window, len(Y), refit_every):
        train = Y.iloc[s - window:s]                                  # estimation window, before the block
        block = Y.iloc[s - window:s + refit_every]                    # window + next block

        # --- 1-2. univariate GJR per asset, variances through window and block
        sig2_b, row = {}, {"date": Y.index[s]}
        for a in Y.columns:
            p = fit_gjr_t(train[a])
            sig2_b[a] = gjr_variance(block[a], p["omega"], p["alpha"], p["gamma"], p["beta"],
                                     sigma2_start=np.mean(train[a] ** 2))
            row[f"pers_{a}"] = p["alpha"] + p["gamma"] / 2 + p["beta"]
        sig2_b = pd.DataFrame(sig2_b)

        # --- 3. residuals; long-run targets from the window only
        eps_b = block / np.sqrt(sig2_b)
        Qb = qbar(eps_b.iloc[:window])
        Rb = cov_to_corr(Qb)

        # --- 4-5. DCC on the window, recursion through window and block
        d = fit_dcc(eps_b.iloc[:window], Q_bar=Qb)
        R_b = dcc_correlations(eps_b, d["a"], d["b"], Q_bar=Qb)

        # --- 6. keep the block days only
        s2_out = sig2_b.iloc[window:]
        H_ccc.append(ccc_cov(s2_out, Rb))
        H_dcc.append(dcc_cov(s2_out, R_b[window:]))
        row.update({"a": d["a"], "b": d["b"], "dcc_converged": d["converged"]})
        params.append(row)

    return np.concatenate(H_ccc), np.concatenate(H_dcc), pd.DataFrame(params).set_index("date")
