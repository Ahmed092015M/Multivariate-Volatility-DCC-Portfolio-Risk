"""
Step 4 - Univariate volatility: GJR-GARCH(1,1) with Student-t shocks.

Gives each asset's volatility sigma_{i,t} and standardised residuals
eps_{i,t} = y_{i,t} / sigma_{i,t}, the inputs of CCC and DCC.
The variance stored at day t uses returns up to t-1 only.

Written fresh for this project (same model as Project 2) and checked against
the `arch` package: estimates agree to 3-4 decimals.

Theory: notes/theory.md, section B4.
"""
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import gammaln

__all__ = ["gjr_variance", "gjr_t_loglik", "fit_gjr_t"]


def gjr_variance(y, omega, alpha, gamma, beta, sigma2_start=None):
    """GJR-GARCH(1,1) variance forecast.

        sigma2_t = omega + (alpha + gamma * 1{y_{t-1} < 0}) * y_{t-1}^2 + beta * sigma2_{t-1}

    gamma > 0: a fall raises volatility more than a rise (equities);
    gamma < 0: the reverse (gold, which tends to rally in panics).

    Parameters
    ----------
    y : pd.Series or array
        Daily returns in %.
    omega, alpha, gamma, beta : float
        Model parameters.
    sigma2_start : float, optional
        Starting variance; default = mean of y^2.

    Returns
    -------
    pd.Series
        Variance forecast for each day.
    """
    x = np.asarray(y, dtype=float)
    s2 = np.empty(len(x))
    s2[0] = np.mean(x ** 2) if sigma2_start is None else sigma2_start
    for t in range(1, len(x)):
        down = 1.0 if x[t - 1] < 0 else 0.0                       # leverage indicator I_{t-1}
        s2[t] = omega + (alpha + gamma * down) * x[t - 1] ** 2 + beta * s2[t - 1]
    return pd.Series(s2, index=getattr(y, "index", None))


def gjr_t_loglik(params, y):
    """Student-t (variance 1) log-likelihood of the GJR-GARCH model, days 2..T.

    log f(y_t) = log G((nu+1)/2) - log G(nu/2) - 1/2 log(pi (nu-2))
                 - 1/2 log(sigma2_t) - (nu+1)/2 log(1 + y_t^2 / ((nu-2) sigma2_t))

    Parameters
    ----------
    params : tuple
        (omega, alpha, gamma, beta, nu).
    y : pd.Series or array
        Daily returns in %.

    Returns
    -------
    float
        Log-likelihood; -1e10 for impossible parameters (non-positive or
        infinite variance), so the optimiser moves away from them.
    """
    omega, alpha, gamma, beta, nu = params
    s2 = gjr_variance(y, omega, alpha, gamma, beta).to_numpy()[1:]
    x = np.asarray(y, dtype=float)[1:]
    if np.any(s2 <= 0) or not np.all(np.isfinite(s2)):
        return -1e10
    const = gammaln((nu + 1) / 2) - gammaln(nu / 2) - 0.5 * np.log(np.pi * (nu - 2))
    return np.sum(const - 0.5 * np.log(s2) - (nu + 1) / 2 * np.log(1 + x ** 2 / ((nu - 2) * s2)))


def fit_gjr_t(y, start=(0.02, 0.03, 0.10, 0.90, 8.0)):
    """Maximum-likelihood estimate of the GJR-GARCH Student-t model (SLSQP).

    Constraints: omega > 0, alpha >= 0, beta >= 0, alpha + gamma >= 0
    (positive reaction after a fall), alpha + gamma/2 + beta < 1 (stationarity),
    2 < nu <= 100.

    Parameters
    ----------
    y : pd.Series or array
        Daily returns in %.
    start : tuple
        Starting values (omega, alpha, gamma, beta, nu).

    Returns
    -------
    dict
        omega, alpha, gamma, beta, nu, loglik, converged.
    """
    bounds = [(1e-6, None),        # omega > 0
              (0.0, 1.0),          # alpha >= 0
              (-1.0, 1.0),         # gamma: sign free (see constraint below)
              (0.0, 1.0),          # beta >= 0
              (2.05, 100.0)]       # nu > 2: finite variance
    constraints = [
        {"type": "ineq", "fun": lambda p: 0.9999 - p[1] - p[2] / 2 - p[3]},   # stationarity
        {"type": "ineq", "fun": lambda p: p[1] + p[2]},                       # reaction to falls >= 0
    ]
    res = minimize(lambda p: -gjr_t_loglik(p, y), x0=np.array(start), method="SLSQP",
                   bounds=bounds, constraints=constraints)
    omega, alpha, gamma, beta, nu = res.x
    return {"omega": omega, "alpha": alpha, "gamma": gamma, "beta": beta, "nu": nu,
            "loglik": -res.fun, "converged": res.success}
