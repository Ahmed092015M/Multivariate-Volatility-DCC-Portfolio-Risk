"""
Step 1 - Input data for the multivariate volatility project.

Builds the inputs every model uses: daily SIMPLE returns in EUR, in percent,
on the days where all prices exist, for CAC 40, DAX, S&P 500, German Bunds
(total return) and gold, plus the return of a fixed-weight portfolio.

Why a total return for the Bund?  A bond ETF's return is mostly its coupons.
Yahoo's adjusted prices missed 8 payouts of the Bund ETF before 2013, which
looked like -2% to -3.5% "crashes"; the total return is rebuilt from close
prices and payouts with `total_return`.

Sections
--------
1. Total return         total_return, return_index     (theory B0, Equation 2)
2. EUR returns          eur_returns                    (theory B0, Equation 1)
3. Portfolio            portfolio_returns              (theory B1)
"""
import pandas as pd

__all__ = ["total_return", "return_index", "eur_returns", "portfolio_returns"]


# =========================================================================== #
# 1. Total return                                                             #
# =========================================================================== #
def total_return(close, payouts):
    """Daily simple total return in %, adding back payouts on their ex-dates.

        R_t = (P_t + D_t) / P_{t-1} - 1          (Equation 2)

    On an ex-date the price falls by the payout D_t but the holder receives D_t
    in cash, so adding it back gives the true return.

    Parameters
    ----------
    close : pd.Series
        Traded (unadjusted) prices.
    payouts : pd.Series
        Cash paid per share, indexed by ex-date (plain dates, no time zone).

    Returns
    -------
    pd.Series
        Total return in %, from the second price on.
    """
    p = close.dropna()
    d = payouts.reindex(p.index).fillna(0.0)          # payout on its ex-date, 0 on other days
    return 100 * ((p + d) / p.shift(1) - 1).dropna()  # price change + cash received


def return_index(returns_pct, first_date, base=100.0):
    """Price index from daily returns in %, starting at `base` on `first_date`.

        I_t = I_{t-1} * (1 + R_t / 100)

    Turns a total-return series (the Bund with payouts added back) into a
    price-like series, so it can be aligned with other prices on a common
    calendar and compounded correctly across holidays.
    """
    growth = (1 + returns_pct / 100).cumprod() * base
    start = pd.Series([base], index=[pd.Timestamp(first_date)])
    return pd.concat([start, growth])


# =========================================================================== #
# 2. Returns in EUR on a common calendar                                      #
# =========================================================================== #
def eur_returns(prices, fx, usd_assets):
    """Daily simple returns in EUR, in %, on the days where all prices exist.

        R_EUR = (1 + R_USD) / (1 + R_X) - 1,   X = USD per 1 EUR     (Equation 1)

    Parameters
    ----------
    prices : pd.DataFrame
        One price (or total-return index) column per asset.
    fx : pd.DataFrame
        EUR/USD rate, column 'EURUSD'.
    usd_assets : list of str
        Columns priced in USD, to convert into EUR.

    Returns
    -------
    pd.DataFrame
        Simple returns in %, one column per asset, common days only.

    Notes
    -----
    Keeping only common days makes every return cover the same time interval
    for every asset, which a covariance matrix needs. Filling holidays with the
    last price would create fake 0% returns and lower volatility and correlation.
    """
    data = prices.join(fx, how="inner").dropna()      # common days: every price and the FX rate
    r = data.pct_change().dropna()                    # return from one common day to the next

    out = r.drop(columns="EURUSD")
    for a in usd_assets:
        out[a] = (1 + r[a]) / (1 + r["EURUSD"]) - 1   # Equation 1
    return 100 * out


# =========================================================================== #
# 3. Portfolio                                                                #
# =========================================================================== #
def portfolio_returns(Y, weights):
    """Portfolio simple return in %, constant weights (daily rebalancing).

        R_p = sum_i w_i * R_i          (exact for simple returns)

    Parameters
    ----------
    Y : pd.DataFrame
        Simple returns in %, one column per asset.
    weights : dict
        Weight per asset; must sum to 1.

    Returns
    -------
    pd.Series
        Daily portfolio return in %, named 'Portfolio'.

    Raises
    ------
    ValueError
        If the weights do not sum to 1.
    """
    w = pd.Series(weights)[Y.columns]                  # same order as the columns
    if abs(w.sum() - 1) > 1e-10:
        raise ValueError(f"weights sum to {w.sum()}, not 1")
    return (Y * w).sum(axis=1).rename("Portfolio")
