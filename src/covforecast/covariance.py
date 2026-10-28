"""
Steps 2-3 - Covariance basics: portfolio variance, rolling and EWMA covariance, correlations.

Arrays of daily matrices have shape (T, N, N): one N x N matrix per day.
Convention: the matrix stored at day t is the FORECAST for day t, built with
returns up to t-1 only (no look-ahead).

Sections
--------
1. Portfolio variance          portfolio_variance               (theory B1, Equation 4)
2. Benchmark forecasts         rolling_cov, ewma_cov            (theory B2-B3, Equations 7-8)
3. Correlations                cov_to_corr, pair_series         (theory B1, Equation 3)
"""
import numpy as np
import pandas as pd

__all__ = ["portfolio_variance", "rolling_cov", "ewma_cov", "cov_to_corr", "pair_series"]


# =========================================================================== #
# 1. Portfolio variance                                                       #
# =========================================================================== #
def portfolio_variance(H, w):
    """Portfolio variance w' H w      (Equation 4).

    Parameters
    ----------
    H : array (N, N) or (T, N, N)
        One covariance matrix, or one per day.
    w : array (N,)
        Portfolio weights.

    Returns
    -------
    float, or array (T,)
    """
    w = np.asarray(w, dtype=float)
    return np.einsum("i,...ij,j->...", w, np.asarray(H, dtype=float), w)   # sum_i sum_j w_i h_ij w_j


# =========================================================================== #
# 2. Benchmark covariance forecasts                                           #
# =========================================================================== #
def rolling_cov(Y, window=250):
    """Rolling-window covariance forecast, zero mean      (Equation 7).

        H_t = (1/W) * sum_{k=1..W} y_{t-k} y_{t-k}'

    Every day of the window has the same weight: slow to react, and an extreme
    day leaves the window abruptly after W days (ghost effect).

    Parameters
    ----------
    Y : pd.DataFrame or array (T, N)
        Daily returns in %.
    window : int
        Number of past days (W).

    Returns
    -------
    array (T, N, N)
        Forecast for each day; NaN for the first `window` days.
    """
    X = np.asarray(Y, dtype=float)
    T, N = X.shape
    outer = X[:, :, None] * X[:, None, :]          # y_t y_t' for every day, shape (T, N, N)
    H = np.full((T, N, N), np.nan)
    for t in range(window, T):
        H[t] = outer[t - window:t].mean(axis=0)    # days t-W ... t-1: never day t itself
    return H


def ewma_cov(Y, lam=0.94, init_window=250):
    """EWMA (RiskMetrics) covariance forecast      (Equation 8).

        H_t = (1 - lam) * y_{t-1} y_{t-1}' + lam * H_{t-1}

    Parameters
    ----------
    Y : pd.DataFrame or array (T, N)
        Daily returns in %.
    lam : float
        Decay factor (0.94 for daily data); half-life ln(0.5)/ln(lam) ~ 11 days.
    init_window : int
        Starting matrix = mean of y y' over the first `init_window` days.
        Its weight decays as lam^t: negligible after about one year.

    Returns
    -------
    array (T, N, N)
        Forecast for each day (built with returns up to the day before).

    Notes
    -----
    Each H_t is a positive mix of a positive definite matrix and a positive
    semi-definite one, so it stays positive definite; but with a short memory
    it can come close to singular in calm periods.
    """
    X = np.asarray(Y, dtype=float)
    T, N = X.shape
    H = np.empty((T, N, N))
    H[0] = X[:init_window].T @ X[:init_window] / init_window        # starting matrix
    for t in range(1, T):
        y = X[t - 1][:, None]                                       # yesterday's returns, column vector
        H[t] = (1 - lam) * (y @ y.T) + lam * H[t - 1]               # Equation 8
    return H


# =========================================================================== #
# 3. Correlations                                                             #
# =========================================================================== #
def cov_to_corr(H):
    """Correlation matrix from a covariance matrix      (Equation 3).

        rho_ij = h_ij / (sigma_i * sigma_j)

    Works for one matrix (N, N) or a stack (T, N, N).
    """
    H = np.asarray(H, dtype=float)
    sd = np.sqrt(np.diagonal(H, axis1=-2, axis2=-1))         # volatilities, shape (..., N)
    return H / (sd[..., :, None] * sd[..., None, :])         # divide h_ij by sigma_i * sigma_j


def pair_series(M, columns, a, b, index):
    """One element (a, b) of a stack of daily matrices, as a dated Series.

    Example: pair_series(cov_to_corr(H), Y.columns, "CAC40", "BUND", Y.index)
    gives the CAC40-Bund correlation for every day.
    """
    i, j = list(columns).index(a), list(columns).index(b)       # positions of the two assets
    return pd.Series(np.asarray(M)[:, i, j], index=index, name=f"{a}-{b}")
