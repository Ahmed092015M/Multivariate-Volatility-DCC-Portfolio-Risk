"""
Step 9 - Portfolio VaR and ES from a covariance forecast, and backtests.

sigma_p,t = sqrt(w' H_t w); VaR_t = sigma_p,t * k_VaR; ES_t = sigma_p,t * k_ES,
with multipliers from the normal distribution or from past standardised
portfolio residuals (filtered historical simulation, as in Project 3).
VaR and ES are positive numbers (a loss), in %.

Sections
--------
1. VaR and ES           var_es_normal, var_es_fhs                         (theory B7, Equation 16)
2. Backtests            kupiec_test, christoffersen_test, acerbi_szekely_z2   (Project 3)
"""
import numpy as np
from scipy import stats
from scipy.special import xlogy

__all__ = ["var_es_normal", "var_es_fhs", "kupiec_test", "christoffersen_test", "acerbi_szekely_z2"]


# =========================================================================== #
# 1. VaR and ES                                                               #
# =========================================================================== #
def var_es_normal(sigma, p_var=0.01, p_es=0.025):
    """Normal VaR (at p_var) and ES (at p_es), both positive, in the units of sigma.

        VaR = -sigma * Phi^{-1}(p_var),   ES = sigma * phi(Phi^{-1}(p_es)) / p_es
    """
    sigma = np.asarray(sigma, dtype=float)
    q_es = stats.norm.ppf(p_es)
    return -sigma * stats.norm.ppf(p_var), sigma * stats.norm.pdf(q_es) / p_es


def var_es_fhs(returns, sigma, p_var=0.01, p_es=0.025, min_obs=250):
    """Filtered historical simulation VaR and ES with an expanding window of past residuals.

    z_s = r_s / sigma_s. On day t, the multipliers are the historical-simulation
    VaR and ES of z_0 ... z_{t-1} (k = floor(n p); VaR = -(k+1)-th worst,
    ES = -mean of the k worst), then VaR_t = sigma_t * k_VaR and ES_t = sigma_t * k_ES.

    Each covariance model gets its own residuals: a model with poor volatility
    forecasts also gets distorted residuals, and is judged on both.

    Parameters
    ----------
    returns, sigma : arrays (T,)
        Realised portfolio returns and volatility forecasts on the same days.
    p_var, p_es : float
        Tail probabilities for VaR and ES.
    min_obs : int
        First day with a forecast (needs at least this many past residuals).

    Returns
    -------
    VaR, ES : arrays (T,)
        NaN for the first `min_obs` days.
    """
    r = np.asarray(returns, dtype=float)
    s = np.asarray(sigma, dtype=float)
    z = r / s                                                    # standardised residuals
    T = len(r)
    var, es = np.full(T, np.nan), np.full(T, np.nan)
    for t in range(min_obs, T):
        past = np.sort(z[:t])                                    # residuals before day t only
        k_v, k_e = int(np.floor(t * p_var)), int(np.floor(t * p_es))
        var[t] = s[t] * -past[k_v]                               # (k+1)-th worst residual
        es[t] = s[t] * -past[:k_e].mean()                        # average of the k worst
    return var, es


# =========================================================================== #
# 2. Backtests                                                                #
# =========================================================================== #
def kupiec_test(hits, p):
    """Kupiec proportion-of-failures test (right NUMBER of breaches).

        LR_POF = -2 [log L(p) - log L(x/T)] ~ chi2(1)

    Returns
    -------
    (breaches, p-value)
    """
    h = np.asarray(hits, dtype=bool)
    T, x = len(h), int(h.sum())
    pi = x / T
    lr = -2 * (xlogy(T - x, 1 - p) + xlogy(x, p) - xlogy(T - x, 1 - pi) - xlogy(x, pi))
    return x, stats.chi2.sf(lr, df=1)


def christoffersen_test(hits, p):
    """Christoffersen conditional coverage (right number AND independent breaches).

        LR_cc = LR_POF + LR_ind ~ chi2(2)

    Returns
    -------
    (n11 = number of consecutive breaches, p-value of LR_cc)
    """
    h = np.asarray(hits, dtype=bool)
    prev, curr = h[:-1], h[1:]                                   # yesterday, today
    n00, n01 = np.sum(~prev & ~curr), np.sum(~prev & curr)
    n10, n11 = np.sum(prev & ~curr), np.sum(prev & curr)
    pi01 = n01 / (n00 + n01) if n00 + n01 else 0.0               # P(breach | no breach yesterday)
    pi11 = n11 / (n10 + n11) if n10 + n11 else 0.0               # P(breach | breach yesterday)
    pi = (n01 + n11) / (n00 + n01 + n10 + n11)
    lr_ind = -2 * (xlogy(n00 + n10, 1 - pi) + xlogy(n01 + n11, pi)
                   - xlogy(n00, 1 - pi01) - xlogy(n01, pi01) - xlogy(n10, 1 - pi11) - xlogy(n11, pi11))
    T, x = len(h), int(h.sum())
    lr_pof = -2 * (xlogy(T - x, 1 - p) + xlogy(x, p) - xlogy(T - x, 1 - x / T) - xlogy(x, x / T))
    return int(n11), stats.chi2.sf(lr_pof + lr_ind, df=2)


def acerbi_szekely_z2(returns, var, es, p=0.025):
    """Acerbi-Szekely Z2 statistic for ES.

        Z2 = sum_t r_t 1{r_t < -VaR_t} / (T p ES_t) + 1

    0 = correct; < 0 = risk underestimated. The 5% critical value is about
    -0.7 for T = 250 and shrinks like 1/sqrt(T) (about -0.17 for T ~ 3,350).
    """
    r, v, e = (np.asarray(x, dtype=float) for x in (returns, var, es))
    return np.sum(r * (r < -v) / e) / (len(r) * p) + 1
