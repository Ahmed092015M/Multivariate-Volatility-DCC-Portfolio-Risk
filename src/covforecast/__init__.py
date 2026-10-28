"""
covforecast - Multivariate volatility: rolling, EWMA, CCC and DCC covariance forecasts.

Project 4 of the Financial Risk Forecasting series (Danielsson, Ch. 3).

Modules
-------
data         Step 1      Bund total return, EUR returns on a common calendar, portfolio
covariance   Steps 2-3   portfolio variance, rolling and EWMA covariance, correlations
univariate   Step 4      GJR-GARCH(1,1) Student-t per asset
mgarch       Steps 4-7   CCC, DCC (recursion, likelihood, estimation), rolling out-of-sample forecasts
evaluation   Step 8      matrix and portfolio QLIKE, Diebold-Mariano test
risk         Step 9      portfolio VaR / ES (normal, FHS) and backtests
portfolio    Step 10     minimum-variance weights and turnover
plots        all steps   every chart of the project

Conventions
-----------
* Returns are SIMPLE returns in percent, in EUR; variances in percent squared.
* Arrays of daily matrices have shape (T, N, N).
* No look-ahead: the forecast stored at day t uses returns up to t-1 only.
"""

__version__ = "1.0.0"
