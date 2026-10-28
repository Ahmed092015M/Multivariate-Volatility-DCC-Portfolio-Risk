"""
Step 8 - Evaluating covariance forecasts out of sample.

The true covariance is never observed; y_t y_t' is a noisy but unbiased proxy.
QLIKE ranks forecasts correctly even with a noisy proxy (Patton, 2011).

Sections
--------
1. Loss functions       qlike_matrix, qlike_portfolio     (theory B6, Equation 14)
2. Significance         diebold_mariano                   (theory B6, Equation 15)
"""
import numpy as np
from scipy import stats

__all__ = ["qlike_matrix", "qlike_portfolio", "diebold_mariano"]


# =========================================================================== #
# 1. Loss functions                                                           #
# =========================================================================== #
def qlike_matrix(H, Y):
    """Matrix QLIKE loss per day      (Equation 14).

        L_t = log|H_t| + y_t' H_t^{-1} y_t        (lower is better)

    Judges every direction of the forecast, including all correlations.
    -1/2 (L_t + N log 2 pi) is the Gaussian log-density of y_t.

    Parameters
    ----------
    H : array (T, N, N)
        Covariance forecasts.
    Y : array or pd.DataFrame (T, N)
        Realised returns on the same days.

    Returns
    -------
    array (T,)
    """
    X = np.asarray(Y, dtype=float)
    _, logdet = np.linalg.slogdet(H)                                              # log|H_t|
    quad = np.einsum("ti,ti->t", X, np.linalg.solve(H, X[:, :, None])[:, :, 0])   # y' H^{-1} y (no inverse)
    return logdet + quad


def qlike_portfolio(var_p, r_p):
    """Portfolio QLIKE per day: log(sigma2_p) + r_p^2 / sigma2_p   (lower is better).

    Judges only the direction w that the investor holds.

    Parameters
    ----------
    var_p : array (T,)
        Portfolio variance forecasts (w' H_t w).
    r_p : array (T,)
        Realised portfolio returns.
    """
    v = np.asarray(var_p, dtype=float)
    return np.log(v) + np.asarray(r_p, dtype=float) ** 2 / v


# =========================================================================== #
# 2. Significance of loss differences                                         #
# =========================================================================== #
def diebold_mariano(loss_a, loss_b, lags=10):
    """Diebold-Mariano test of equal accuracy, Newey-West standard error  (Equation 15).

        d_t = loss_a_t - loss_b_t,    DM = mean(d) / se(mean(d))  ~ N(0, 1)

    Daily loss differences are autocorrelated (volatility clusters), so the
    standard error uses Bartlett weights up to `lags` days.

    Returns
    -------
    (DM statistic, two-sided p-value)
        Negative DM: model A is more accurate. (nan, nan) if the two loss
        series are identical (e.g. a model compared with itself).
    """
    d = np.asarray(loss_a, dtype=float) - np.asarray(loss_b, dtype=float)
    T = len(d)
    u = d - d.mean()
    lrv = np.sum(u * u) / T                                              # variance of d
    for k in range(1, lags + 1):
        lrv += 2 * (1 - k / (lags + 1)) * np.sum(u[k:] * u[:-k]) / T      # Bartlett-weighted autocovariances
    if lrv <= 0:
        return np.nan, np.nan                                            # identical losses: no test
    dm = d.mean() / np.sqrt(lrv / T)
    return dm, 2 * stats.norm.sf(abs(dm))
