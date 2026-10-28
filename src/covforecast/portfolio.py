"""
Step 10 - Minimum-variance portfolios: an economic test of covariance forecasts.

The minimum-variance portfolio depends only on the covariance matrix: if a
model's forecasts are better, its minimum-variance portfolio should be less
volatile out of sample (Engle and Colacito, 2006). Turnover measures the cost
of following it.

Theory: notes/theory.md, section B8 (Equations 17-18).
"""
import numpy as np

__all__ = ["min_variance_weights", "turnover"]


def min_variance_weights(H):
    """Minimum-variance weights, fully invested, short sales allowed   (Equation 17).

        w = H^{-1} 1 / (1' H^{-1} 1)

    Uses H^{-1}, so it exploits every correlation, including near-singular
    directions (e.g. CAC 40 vs DAX): errors in the forecast become extreme
    long-short positions ("error maximisation").

    Parameters
    ----------
    H : array (N, N) or (T, N, N)

    Returns
    -------
    array (N,) or (T, N)
    """
    H = np.asarray(H, dtype=float)
    ones = np.ones(H.shape[-1])
    x = np.linalg.solve(H, np.broadcast_to(ones, H.shape[:-1])[..., None])[..., 0]   # H^{-1} 1, no explicit inverse
    return x / x.sum(axis=-1, keepdims=True)                                        # scale so weights sum to 1


def turnover(W):
    """Daily turnover sum_i |w_{i,t} - w_{i,t-1}| of a weight path (T, N); first day = NaN.

    A proxy for trading costs: at c basis points per unit traded, the yearly
    cost is about mean turnover x 252 x c.
    """
    W = np.asarray(W, dtype=float)
    out = np.full(len(W), np.nan)
    out[1:] = np.abs(np.diff(W, axis=0)).sum(axis=1)
    return out
