"""
Charts for the multivariate volatility project.

Every function draws one chart from results computed in the notebook and
returns the Figure, so the notebook can save it before showing it.

Sections
--------
1. Shared settings          CRISES, MODEL_COLORS
2. Correlations over time   plot_correlations                     (Steps 2-5)
3. Forecast accuracy        plot_scores                           (Step 8)
4. Risk                     plot_var_backtest                     (Step 9)
5. Portfolios               plot_mv_tradeoff                      (Step 10)
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

__all__ = ["CRISES", "MODEL_COLORS", "plot_correlations", "plot_scores",
           "plot_var_backtest", "plot_mv_tradeoff"]


# =========================================================================== #
# 1. Shared settings                                                          #
# =========================================================================== #
# Stress periods shaded on time charts: (start, end, label)
CRISES = [
    ("2008-09-01", "2009-03-31", "GFC"),
    ("2011-07-01", "2011-12-31", "Euro crisis"),
    ("2020-02-20", "2020-04-30", "Covid"),
    ("2022-01-01", "2022-12-31", "Inflation"),
    ("2026-03-01", "2026-09-30", "2026 regime"),
]

# One fixed colour per covariance model in every chart
MODEL_COLORS = {"Rolling 250": "#7f7f7f", "EWMA": "#2ca02c", "CCC": "#ff7f0e", "DCC": "#1f77b4"}


def _clean(ax):
    """Remove the top and right frame lines."""
    ax.spines[["top", "right"]].set_visible(False)


# =========================================================================== #
# 2. Correlations over time                                                   #
# =========================================================================== #
def plot_correlations(series, title, shade=CRISES):
    """Line chart of one or more correlation series, with stress periods shaded.

    Parameters
    ----------
    series : dict of {label: pd.Series}
        Correlation series to draw.
    title : str
        Chart title.
    shade : list of (start, end, label), optional
        Periods to shade in grey.

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=(12, 4.5))
    for start, end, text in shade or []:                       # grey bands behind the lines
        ax.axvspan(pd.Timestamp(start), pd.Timestamp(end), color="grey", alpha=0.15, lw=0)
        ax.text(pd.Timestamp(start), 0.93, text, fontsize=8, va="top")
    for label, s in series.items():
        ax.plot(s.index, s.values, lw=1.2, label=label)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_ylim(-1, 1)
    ax.set_ylabel("correlation")
    ax.set_title(title, fontsize=11)
    ax.legend(loc="lower left", fontsize=8)
    _clean(ax)
    fig.tight_layout()
    return fig


# =========================================================================== #
# 3. Forecast accuracy                                                        #
# =========================================================================== #
def plot_scores(scores):
    """Bar charts of the out-of-sample matrix and portfolio QLIKE per model.

    Parameters
    ----------
    scores : pd.DataFrame
        Index = model names; columns 'matrix QLIKE', 'portfolio QLIKE' and
        'DM vs DCC (matrix)' (as built in Step 8.2). Lower QLIKE = better.

    Returns
    -------
    matplotlib.figure.Figure
    """
    models = list(scores.index)
    colors = [MODEL_COLORS.get(m, "grey") for m in models]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, col, title in [(axes[0], "matrix QLIKE", "Whole covariance matrix (matrix QLIKE)"),
                           (axes[1], "portfolio QLIKE", "60/30/10 portfolio (portfolio QLIKE)")]:
        values = scores[col].astype(float)
        ax.bar(models, values, color=colors)
        for i, v in enumerate(values):
            ax.text(i, v, f"{v:.3f}", ha="center", va="bottom" if v >= 0 else "top", fontsize=9)
        ax.axhline(0, color="black", lw=0.8)
        ax.set_title(title, fontsize=11)
        _clean(ax)
    dm = scores["DM vs DCC (matrix)"].astype(float)
    note = ", ".join(f"{m} {v:.1f}" for m, v in dm.items() if np.isfinite(v))
    fig.suptitle(f"Out-of-sample accuracy, lower is better  |  DM vs DCC (matrix): {note}", fontsize=10)
    fig.tight_layout()
    return fig


# =========================================================================== #
# 4. Risk                                                                     #
# =========================================================================== #
def plot_var_backtest(bt, expected):
    """VaR 99% breaches per covariance model, normal vs FHS tails, against the expected number.

    Parameters
    ----------
    bt : pd.DataFrame
        Index = (model, tail) with tail in {'normal', 'FHS'}; column 'breaches 99%'
        and 'avg VaR 99% (EUR k)' (as built in Step 9.2).
    expected : float
        Expected number of breaches (T * 1%).

    Returns
    -------
    matplotlib.figure.Figure
    """
    models = list(dict.fromkeys(bt.index.get_level_values(0)))
    x = np.arange(len(models))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    ax = axes[0]
    for k, (tail, hatch) in enumerate([("normal", "//"), ("FHS", "")]):
        vals = [float(bt.loc[(m, tail), "breaches 99%"]) for m in models]
        ax.bar(x + (k - 0.5) * 0.38, vals, width=0.38, label=f"{tail} tail", hatch=hatch,
               color=[MODEL_COLORS.get(m, "grey") for m in models], edgecolor="white")
        for xi, v in zip(x + (k - 0.5) * 0.38, vals):
            ax.text(xi, v + 0.8, f"{v:.0f}", ha="center", fontsize=8)
    ax.axhline(expected, color="black", ls="--", lw=1, label=f"expected ({expected:.1f})")
    ax.set_xticks(x, models)
    ax.set_title("VaR 99% breaches (hatched = normal tail)", fontsize=11)
    ax.legend(fontsize=8)
    _clean(ax)

    ax = axes[1]
    vals = [float(bt.loc[(m, "FHS"), "avg VaR 99% (EUR k)"]) for m in models]
    ax.bar(models, vals, color=[MODEL_COLORS.get(m, "grey") for m in models])
    for i, v in enumerate(vals):
        ax.text(i, v + 1, f"{v:.0f}k", ha="center", fontsize=9)
    ax.set_title("Average VaR 99%, FHS tail (EUR 10m)", fontsize=11)
    _clean(ax)
    fig.tight_layout()
    return fig


# =========================================================================== #
# 5. Portfolios                                                               #
# =========================================================================== #
def plot_mv_tradeoff(mv, costs):
    """Minimum-variance strategies: realised volatility vs turnover, and net return vs trading cost.

    Parameters
    ----------
    mv : pd.DataFrame
        Index = model; columns 'realised vol % (annual)' and 'avg turnover % / day' (Step 10.1).
    costs : pd.DataFrame
        Index = (cost in bp, model); column 'annual return % (net)' (Step 10.2).

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    ax = axes[0]
    for m in mv.index:
        x, y = float(mv.loc[m, "avg turnover % / day"]), float(mv.loc[m, "realised vol % (annual)"])
        ax.scatter(x, y, s=120, color=MODEL_COLORS.get(m, "grey"))
        ax.annotate(m, (x, y), xytext=(7, 4), textcoords="offset points")
    ax.set_xlabel("average turnover, % of the portfolio per day")
    ax.set_ylabel("realised volatility, % a year")
    ax.set_title("Same risk, very different trading", fontsize=11)
    _clean(ax)

    ax = axes[1]
    bps = sorted(set(costs.index.get_level_values(0)))
    for m in mv.index:
        vals = [float(costs.loc[(bp, m), "annual return % (net)"]) for bp in bps]
        ax.plot(bps, vals, marker="o", color=MODEL_COLORS.get(m, "grey"), label=m)
    ax.set_xlabel("trading cost, basis points per unit traded")
    ax.set_ylabel("annual return net of costs, %")
    ax.set_title("After costs, the slow forecast wins", fontsize=11)
    ax.legend(fontsize=8)
    _clean(ax)
    fig.suptitle("Minimum-variance portfolios built from each covariance forecast (2012-2026)", fontsize=10)
    fig.tight_layout()
    return fig
