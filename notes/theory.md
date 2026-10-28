# Project 4 – Multivariate Volatility: EWMA, CCC and DCC – Theory Notes

*Methods follow Daníelsson, Financial Risk Forecasting, Chapter 3 (multivariate volatility models), used as a guide only. This file goes in the repo as `notes/theory.md`.*

These notes use the same frameworks as Projects 1–3:

- **Part A** applies the problem-solving framework to the project as a whole: what to determine, information and assumptions, method, why it works, how to verify.
- **Part B** applies the deep-learning framework to each technique: concept, why it was developed, the maths (with a definition for every equation and every variable), when it is valid, its limitations, and how to derive, implement and validate it. Each technique ends with a small real-life example.
- **Part C** is a one-page summary of all the equations.
- **Part D** is the development plan, step by step.

The worked examples use round, illustrative numbers. Our own results will come from our data.

**Units and conventions.** Returns are in **percent**, variances and covariances in **percent squared**. Vectors are in bold lower case ($\mathbf{y}_t$, $\mathbf{w}$), matrices in upper case ($H_t$, $R_t$). $\mathbf{w}'$ is the transpose of $\mathbf{w}$. As in Projects 2–3, the value for day *t* uses information up to day *t − 1* only.

---

## Decisions for this project

| Topic | Decision |
|---|---|
| Assets | **Multi-asset:** CAC 40, DAX, S&P 500, a **US Treasury bond** index (7–10 years) and **gold**, all as daily returns in **EUR** for a European investor |
| Data (to confirm in Step 1) | Equities from Project 1; EUR/USD from Project 3 (FRED); bonds: iShares 7–10 Year Treasury ETF (IEF, from 2002, total return via adjusted prices); gold: COMEX gold futures, front month (GC=F, from 2000). The final choice is made after the data check, as in Project 1 |
| Models | Rolling-window covariance (benchmark), **EWMA** (RiskMetrics), **CCC-GARCH** (constant correlation), **DCC-GARCH** (dynamic correlation), estimated by our own maximum likelihood |
| Evaluation | Covariance forecast loss (**matrix QLIKE**) with Diebold–Mariano tests; **portfolio VaR/ES** backtested with the Project 3 tools; **minimum-variance portfolio** realised volatility |
| Portfolios | A diversified buy-side portfolio (weights fixed in Step 1, for example equities 60%, bonds 30%, gold 10%) and the minimum-variance portfolio rebuilt from each model's forecast |
| Rule | Every forecast for day *t* uses information up to day *t − 1* only |

---

## Part A – The project's problem

### A1. What do I need to determine?
Project 3 measured risk one series at a time. It found two problems that only a **joint** model can solve: separately estimated ES were not consistent (the portfolio ES exceeded the sum of its parts on 18 days), and diversification collapsed when correlations jumped in 2008 and 2020. This project answers:

> **How should a buy-side risk team forecast the covariance matrix of a multi-asset portfolio, and does a dynamic-correlation model give better risk numbers and better portfolios than constant or simple rolling correlations?**

More precisely:

1. How the correlations between equities, bonds and gold change over time, and whether they move together in crises (equities) or flip sign (stocks vs bonds).
2. Which covariance model forecasts best out of sample: rolling window, EWMA, CCC or DCC.
3. Whether a better covariance model gives a better **portfolio VaR and ES** (Project 3 backtests).
4. Whether it gives a **lower realised volatility** for the minimum-variance portfolio.
5. Whether correlations are really time-varying (DCC vs CCC test).

### A2. What information and assumptions do I have?
- **Data:** daily prices of the three equity indices (Project 1), EUR/USD (Project 3), a US Treasury bond ETF and gold futures (new, Step 1).
- **Results from earlier projects:** GJR-t is the best univariate volatility model (Project 2); filtered historical simulation is the best VaR/ES method (Project 3).
- **Assumptions:**
  - Daily means are close to zero compared with volatility; they are set to zero.
  - Returns can be written as $\mathbf{y}_t = H_t^{1/2}\mathbf{z}_t$: a covariance matrix that changes over time times shocks with identity covariance.
  - Parameters are stable within an estimation window.
- **Known difficulties:**
  - A covariance matrix must stay **positive definite** (no portfolio can have negative variance).
  - The number of parameters grows quickly with the number of assets (the "curse of dimensionality").
  - Markets close at different times (Europe, New York, COMEX), which lowers measured daily correlations.
  - A front-month futures series has small jumps at each contract roll.

### A3. Which method is appropriate?
| Question | Method |
|---|---|
| Benchmark | Rolling-window sample covariance |
| Simple dynamic covariance | EWMA (RiskMetrics), one parameter λ |
| Separate volatilities, constant correlation | CCC-GARCH (Bollerslev, 1990) |
| Separate volatilities, dynamic correlation | DCC-GARCH (Engle, 2002), two-step maximum likelihood |
| Is correlation time-varying? | Likelihood-ratio test DCC vs CCC |
| Which forecast is best? | Matrix QLIKE loss, Diebold–Mariano test |
| Does it matter for risk? | Portfolio VaR/ES from $\mathbf{w}'H_t\mathbf{w}$, Project 3 backtests |
| Does it matter for allocation? | Minimum-variance portfolio, realised out-of-sample volatility |

### A4. Why should the method work?
- Each asset's volatility clusters (Projects 1–2), and so do the co-movements: big joint falls are followed by more joint falls. A model that updates the covariance with yesterday's shocks uses this information.
- DCC separates the problem into two easy parts: one GARCH per asset for the volatilities, then a small model for the correlations of the standardised shocks. This keeps the number of parameters small (two for all the correlations) and the matrix positive definite.
- A covariance forecast can be judged without knowing the true covariance: the matrix QLIKE loss ranks models correctly even with the noisy proxy $\mathbf{y}_t\mathbf{y}_t'$.

### A5. How can I verify the answer?
- **Formulas:** portfolio variance against direct simulation; EWMA covariance against pandas' `ewm().cov()`; matrix log-likelihood against `scipy.stats.multivariate_normal`.
- **Simulation:** simulate a DCC process with known parameters and check that the estimation recovers them; simulate constant correlations and check that DCC estimates a ≈ 0.
- **Positive definiteness:** smallest eigenvalue of every forecast matrix > 0.
- **Consistency:** with two assets, CCC and DCC correlations must agree with rolling correlations on average; with a = b = 0, DCC must equal CCC.
- **Out of sample:** a model is better only if it wins on days it was not estimated on.

---

## Part B – The techniques

### B0. Multi-asset returns in EUR

**1. Concept.** The investor holds equities, bonds and gold. All returns are measured in EUR, so the three USD assets (S&P 500, US Treasuries, gold) include the EUR/USD move.

**2. Why it was developed.** A European buy-side portfolio is measured in EUR. The currency changes the correlations: in a crisis the USD often rises, which raises the EUR return of every USD asset at the same time.

**3. Maths.**

**Variables used in this section:**

- $P_{i,t}$: price of asset *i* on day *t* (in its own currency).
- $X_t$: exchange rate, USD per 1 EUR.
- $R_{i,t}$: simple return; $R^{EUR}_{i,t}$: simple return in EUR.

**Equation 1 – Return in EUR of a USD asset** (as in Project 3):
$$R^{EUR}_{i,t} = \frac{1 + R_{i,t}}{1 + R_{X,t}} - 1.$$
*Definition:* the EUR return combines the asset's own return and the currency return.

**Equation 2 – Total return of an ETF:** $R_{t} = P^{adj}_t / P^{adj}_{t-1} - 1$, where $P^{adj}$ is the price adjusted for dividends (coupons for a bond ETF).
*Definition:* a bond's return is mainly its coupon plus the price change from moving yields; without the adjustment the coupons would be lost.

**4. Valid when.** Prices are taken on common days (all series exist), as in Project 3.

**5. Limitations.**
- Non-synchronous closes: European indices 17:30 CET, US 22:00 CET, gold futures and the FX fixing at other times.
- A front-month futures series jumps slightly at each roll date; these jumps are not real returns.
- ETF returns include fees (about 0.15% a year), negligible on a daily scale.

**6. Derive, implement, validate.**
- *Implement:* reuse `eur_returns` logic for any list of USD assets.
- *Validate:* constant FX gives the USD return; the roll days of the gold series are checked for unusual moves.

**Example.** On a crisis day the S&P 500 falls 3%, US Treasuries rise 1%, gold rises 2% and the USD rises 1.5% (EUR/USD −1.5%). In EUR: S&P 500 $0.97/0.985 - 1 = -1.5\%$, Treasuries $+2.5\%$, gold $+3.6\%$. The currency turns a large loss into a small one and amplifies the hedges.

---

### B1. The covariance matrix and portfolio variance

**1. Concept.** With *N* assets, risk is described by an $N \times N$ **covariance matrix**: variances on the diagonal, covariances off the diagonal. The variance of any portfolio follows from it.

**2. Why it was developed.** Markowitz (1952): a portfolio's risk depends not only on each asset's risk but on how the assets move together. This is the basis of diversification and of every parametric portfolio VaR.

**3. Maths.**

**Variables used in this section:**

- $H_t$: conditional covariance matrix for day *t* ($N \times N$); $h_{ij,t}$ its elements.
- $\sigma_{i,t} = \sqrt{h_{ii,t}}$: volatility of asset *i*.
- $\rho_{ij,t}$: correlation between assets *i* and *j*.
- $D_t = \text{diag}(\sigma_{1,t},\dots,\sigma_{N,t})$: diagonal matrix of volatilities.
- $R_t$: correlation matrix (ones on the diagonal).
- $\mathbf{w}$: vector of portfolio weights.

**Equation 3 – Covariance = volatilities × correlations:**
$$H_t = D_t R_t D_t, \qquad h_{ij,t} = \rho_{ij,t}\,\sigma_{i,t}\,\sigma_{j,t}.$$
*Definition:* any covariance matrix splits into the volatilities and the correlations. CCC and DCC model the two parts separately.

**Equation 4 – Portfolio variance:**
$$\sigma^2_{p,t} = \mathbf{w}' H_t \mathbf{w} = \sum_i \sum_j w_i w_j h_{ij,t}.$$
*Definition:* the portfolio's variance is the weighted sum of all variances and covariances.

**Equation 5 – Positive definiteness:** $\mathbf{w}'H_t\mathbf{w} > 0$ for every non-zero $\mathbf{w}$; equivalently, all eigenvalues of $H_t$ are positive.
*Definition:* no combination of assets can have zero or negative variance. A forecast that breaks this can give a negative portfolio variance, an impossible VaR, or an optimiser that "finds" a risk-free portfolio of risky assets.

**4. Valid when.** The second moments exist; the matrix is estimated with enough observations ($T \gg N$).

**5. Limitations.**
- Correlation measures only **linear** co-movement; joint crashes can be stronger than the correlation suggests (tail dependence, Project 7 territory).
- With many assets, sample covariance matrices are noisy (estimation error grows with $N^2$).

**6. Derive, implement, validate.**
- *Derive:* $\text{Var}(\sum_i w_i y_i) = \sum_i\sum_j w_i w_j \text{Cov}(y_i, y_j)$.
- *Implement:* `portfolio_variance(H, w)` with `w @ H @ w`; works for a single matrix or a stack of daily matrices.
- *Validate:* against the variance of simulated portfolio returns; against `np.cov` of a weighted sum.

**Example.** 60% equities (volatility 1.2%) and 40% bonds (0.5%). If the stock–bond correlation is **−0.3**, portfolio volatility is 0.687% and the 99% normal VaR on EUR 10m is EUR 160k. If the correlation turns **+0.3** (as in 2022, when inflation hit both), volatility is 0.803% and VaR EUR 187k: **+17% risk with the same positions**, only because one correlation changed sign.

---

### B2. The multivariate model and the rolling-window benchmark

**1. Concept.** The multivariate version of $y_t = \sigma_t z_t$: today's return vector is today's covariance "square root" times a vector of independent standard shocks. The simplest forecast of $H_t$ is the sample covariance of the last *W* days.

**2. Why it was developed.** It extends the univariate framework of Projects 2–3 to portfolios, and the rolling window is the benchmark every bank used before EWMA.

**3. Maths.**

**Variables used in this section:**

- $\mathbf{y}_t$: vector of the *N* returns on day *t*.
- $\mathbf{z}_t$: vector of shocks with mean 0 and identity covariance $I_N$.
- $H_t^{1/2}$: any matrix with $H_t^{1/2}(H_t^{1/2})' = H_t$ (for example the Cholesky factor).
- $W$: window length.

**Equation 6 – Multivariate model:** $\mathbf{y}_t = H_t^{1/2}\mathbf{z}_t$, so $E[\mathbf{y}_t\mathbf{y}_t' \mid \mathcal{F}_{t-1}] = H_t$.
*Definition:* all predictability is in $H_t$; the shocks are unpredictable and uncorrelated.

**Equation 7 – Rolling-window covariance:**
$$\hat H_t = \frac{1}{W}\sum_{k=1}^{W} \mathbf{y}_{t-k}\mathbf{y}_{t-k}'.$$
*Definition:* the average of the last *W* outer products (zero-mean version), shifted one day so that day *t* is not used.

**4. Valid when.** The covariance is constant within the window.

**5. Limitations.** The same ghost effect and slow reaction as the moving average in Project 2 and HS in Project 3: every day in the window has the same weight.

**6. Derive, implement, validate.**
- *Implement:* `rolling_cov(Y, window)` returning a $(T, N, N)$ array.
- *Validate:* one date against `np.cov(..., bias=True)` on de-meaned data with zero mean imposed.

**Example.** With *W* = 250, the March 2020 crash stays in the covariance forecast until March 2021, then disappears in one day.

---

### B3. EWMA covariance (RiskMetrics)

**1. Concept.** Exponentially weighted covariance: yesterday's cross-product gets weight $1-\lambda$, older days fade by $\lambda$ per day.

**2. Why it was developed.** RiskMetrics (J.P. Morgan, 1994) needed one method for thousands of risk factors that reacts quickly and is always positive definite, without any estimation.

**3. Maths.**

**Variables used in this section:**

- $\lambda$: decay factor (0.94 for daily data).

**Equation 8 – EWMA covariance:**
$$H_t = (1-\lambda)\,\mathbf{y}_{t-1}\mathbf{y}_{t-1}' + \lambda\,H_{t-1}.$$
*Definition:* each element follows the EWMA of Project 2: $h_{ij,t} = (1-\lambda)y_{i,t-1}y_{j,t-1} + \lambda h_{ij,t-1}$.

**4. Valid when.** The same decay suits every variance and every covariance; no mean reversion is needed.

**5. Limitations.**
- One λ for all assets: bonds, equities and gold must forget at the same speed.
- No long-run level: after a shock, forecasts never revert to an average (the multi-day forecast is flat).
- Positive definite only if the starting matrix is, and only semi-definite while fewer than *N* days have passed.

**6. Derive, implement, validate.**
- *Implement:* `ewma_cov(Y, lam=0.94)` in a loop over days.
- *Validate:* each element against Project 2's `ewma_variance` (diagonal) and against pandas' `ewm(alpha=0.06, adjust=False).cov()` (off-diagonal).

**Example.** Yesterday's covariance between stocks and bonds is 0.10. Today stocks rise 2% and bonds fall 1%. New covariance: $0.94 \times 0.10 + 0.06 \times (2 \times -1) = -0.026$. One day with opposite moves turns the covariance negative.

---

### B4. CCC-GARCH: constant conditional correlation

**1. Concept.** Model each asset's volatility with its own univariate GARCH, and keep the correlations constant.

**2. Why it was developed.** Bollerslev (1990): full multivariate GARCH models (VECH, BEKK) need hundreds of parameters for a few assets. CCC needs only the univariate GARCH parameters plus one correlation matrix, which is the correlation of the standardised residuals.

**3. Maths.**

**Variables used in this section:**

- $\sigma^2_{i,t}$: GARCH (or GJR) variance of asset *i* (Project 2).
- $\boldsymbol{\varepsilon}_t = D_t^{-1}\mathbf{y}_t$: standardised residuals ($\varepsilon_{i,t} = y_{i,t}/\sigma_{i,t}$).
- $\bar R$: constant correlation matrix.

**Equation 9 – CCC:**
$$H_t = D_t \bar R D_t, \qquad \bar R = \text{corr}(\boldsymbol{\varepsilon}_t) \text{ over the estimation window}.$$
*Definition:* volatilities change every day, correlations never do.

**4. Valid when.** Correlations really are constant; each univariate GARCH is well specified.

**5. Limitations.** Cannot capture rising correlations in crises or sign changes (stocks vs bonds in 2022).

**6. Derive, implement, validate.**
- *Implement:* fit GJR-t (or GARCH) per asset with Project 2 code, then `np.corrcoef` of the residuals.
- *Validate:* $\bar R$ is a valid correlation matrix (ones on the diagonal, positive definite); CCC is the special case a = b = 0 of DCC.

**Example.** Stock–bond correlation of residuals = −0.25 over 2014–2021. In 2022 CCC keeps −0.25, while the realised correlation was positive: CCC would have overstated the hedge from bonds exactly when inflation hit both assets.

---

### B5. DCC-GARCH: dynamic conditional correlation

**1. Concept.** Keep CCC's univariate GARCH volatilities, but let the correlation matrix follow its own GARCH-like recursion driven by yesterday's standardised shocks.

**2. Why it was developed.** Engle (2002) and Engle and Sheppard (2001): a model flexible enough for time-varying correlations, with only **two** extra parameters whatever the number of assets, estimable in two easy steps, and positive definite by construction.

**3. Maths.**

**Variables used in this section:**

- $Q_t$: "quasi-correlation" matrix (a covariance of the standardised shocks).
- $\bar Q$: unconditional covariance of $\boldsymbol{\varepsilon}_t$ (correlation targeting).
- $a$: reaction of correlations to yesterday's shocks; $b$: persistence.
- $\text{diag}(Q_t)$: diagonal matrix with the diagonal of $Q_t$.

**Equation 10 – DCC recursion:**
$$Q_t = (1 - a - b)\,\bar Q + a\,\boldsymbol{\varepsilon}_{t-1}\boldsymbol{\varepsilon}_{t-1}' + b\,Q_{t-1}.$$
*Definition:* like a GARCH(1,1) for every element of the correlation structure: a long-run level, a reaction to yesterday's joint shock and a persistence term.

**Equation 11 – Rescaling to a correlation matrix:**
$$R_t = \text{diag}(Q_t)^{-1/2}\,Q_t\,\text{diag}(Q_t)^{-1/2}, \qquad H_t = D_t R_t D_t.$$
*Definition:* $Q_t$ does not have exact ones on its diagonal, so it is rescaled; $\rho_{ij,t} = q_{ij,t}/\sqrt{q_{ii,t}q_{jj,t}}$.

**Equation 12 – Two-step estimation (Gaussian quasi-likelihood).** Step 1: one univariate GARCH per asset (Project 2), giving $D_t$ and $\boldsymbol{\varepsilon}_t$. Step 2: maximise the correlation part of the log-likelihood over $(a, b)$:
$$\ell_C(a, b) = -\frac12\sum_t\left[\ln|R_t| + \boldsymbol{\varepsilon}_t' R_t^{-1}\boldsymbol{\varepsilon}_t - \boldsymbol{\varepsilon}_t'\boldsymbol{\varepsilon}_t\right],$$
subject to $a \ge 0$, $b \ge 0$, $a + b < 1$.
*Definition:* the full Gaussian log-likelihood splits into a volatility part (step 1) and a correlation part (step 2). The term $\boldsymbol{\varepsilon}_t'\boldsymbol{\varepsilon}_t$ does not depend on $(a,b)$ and is kept only so that $\ell_C = 0$ when $R_t = I$.

**Equation 13 – Testing constant correlation:** $LR = 2\,[\ell_C(\hat a, \hat b) - \ell_C(0, 0)]$, where $(0,0)$ is CCC.
*Definition:* compares DCC with CCC on the same residuals. Because $a = 0$ lies on the boundary, the χ²(2) distribution is conservative; a very large LR is still clear evidence.

**4. Valid when.** The univariate models are good (step 1 errors carry into step 2); all correlations share the same dynamics $(a, b)$.

**5. Limitations.**
- Same $(a, b)$ for every pair: stock–stock and stock–gold correlations must react at the same speed.
- Two-step estimation is consistent but less efficient than joint estimation; standard errors need a correction.
- Symmetric: a joint fall and a joint rise move correlations equally (asymmetric DCC fixes this).
- Gaussian quasi-likelihood: fat tails handled through the univariate step and FHS-type VaR, not in the correlation step.

**6. Derive, implement, validate.**
- *Derive:* $\ln|H_t| = 2\ln|D_t| + \ln|R_t|$ and $\mathbf{y}_t'H_t^{-1}\mathbf{y}_t = \boldsymbol{\varepsilon}_t'R_t^{-1}\boldsymbol{\varepsilon}_t$ split the likelihood.
- *Implement:* `dcc_correlations(eps, a, b)`, `dcc_loglik(params, eps)`, `fit_dcc(eps)` with `scipy.optimize.minimize` (SLSQP), and a rolling version as in Project 2.
- *Validate:* simulate DCC data with known $(a, b) = (0.05, 0.93)$ and recover them; with constant correlation, $\hat a \approx 0$; the log-likelihood equals `scipy.stats.multivariate_normal.logpdf` up to the constant; every $R_t$ positive definite.

**Example.** Long-run correlation 0.50, $a = 0.05$, $b = 0.93$, yesterday $q_{12} = 0.50$ with $q_{11} = q_{22} = 1$. A joint crash day gives shocks $\varepsilon_1 = -3$, $\varepsilon_2 = -2.5$. Then $q_{12} = 0.02 \times 0.5 + 0.05 \times 7.5 + 0.93 \times 0.5 = 0.85$, $q_{11} = 1.40$, $q_{22} = 1.26$, so $\rho_{12} = 0.85/\sqrt{1.40 \times 1.26} = 0.64$. **One crash day raises the correlation from 0.50 to 0.64**: diversification shrinks exactly when it is needed.

---

### B6. Evaluating covariance forecasts: matrix QLIKE and Diebold–Mariano

**1. Concept.** The true covariance is never observed. As in Project 2, forecasts are compared with a noisy proxy, here the outer product $\mathbf{y}_t\mathbf{y}_t'$, using a loss function that ranks models correctly.

**2. Why it was developed.** Patton (2011) and Laurent, Rombouts and Violante (2013) showed that only some loss functions give the same ranking with a noisy proxy as with the truth. The multivariate QLIKE is one of them, and it is the negative Gaussian log-likelihood up to constants.

**3. Maths.**

**Variables used in this section:**

- $L_t^{(m)}$: loss of model *m* on day *t*.
- $d_t$: loss difference between two models.

**Equation 14 – Matrix QLIKE:**
$$L_t = \ln|H_t| + \mathbf{y}_t' H_t^{-1}\mathbf{y}_t.$$
*Definition:* penalises a forecast that is too large (first term) and one that is too small or has the wrong correlations (second term). Lower is better.

**Equation 15 – Diebold–Mariano test** (Project 2): $DM = \bar d / \widehat{se}(\bar d) \approx N(0,1)$ with a Newey–West standard error.
*Definition:* tests whether the average loss difference between two models is significant.

**4. Valid when.** $E[\mathbf{y}_t\mathbf{y}_t' \mid \mathcal{F}_{t-1}] = H_t^{true}$ (unbiased proxy); $H_t$ invertible.

**5. Limitations.** The daily proxy is very noisy, so differences need long samples; the loss treats all directions in the portfolio space equally, whereas a manager cares about specific portfolios.

**6. Derive, implement, validate.**
- *Implement:* `qlike_matrix(H, Y)` with `np.linalg.slogdet` and `np.linalg.solve` (no explicit inverse).
- *Validate:* for *N* = 1 it equals Project 2's QLIKE; $-\frac12(L_t + N\ln 2\pi)$ equals scipy's multivariate normal log-density.

**Example.** On a flight-to-quality day stocks fall 2% and bonds rise 0.8%. Model A forecasts a stock–bond correlation of −0.3, model B +0.3 (same volatilities 1.2% and 0.5%). QLIKE: A 2.99, B 6.51. The model with the right **sign** of correlation wins clearly, even with identical volatilities.

---

### B7. Portfolio VaR and ES from a covariance forecast

**1. Concept.** Turn each day's covariance forecast into the portfolio's volatility, then into VaR and ES with the Project 3 methods, and backtest them.

**2. Why it was developed.** This is how a covariance model is used in practice: a risk system computes $\mathbf{w}'H_t\mathbf{w}$ for every portfolio from one shared covariance forecast. The joint model gives consistent numbers for any portfolio and any sub-portfolio.

**3. Maths.**

**Variables used in this section:**

- $\sigma_{p,t} = \sqrt{\mathbf{w}'H_t\mathbf{w}}$: portfolio volatility forecast.
- $z_{p,t} = R_{p,t}/\sigma_{p,t}$: portfolio standardised residual.

**Equation 16 – Portfolio VaR and ES:**
$$\text{VaR}_t = \sigma_{p,t}\,k_{VaR}, \qquad \text{ES}_t = \sigma_{p,t}\,k_{ES},$$
where the multipliers come from the normal, the Student-t or the empirical residuals $z_{p,t}$ (filtered historical simulation, Project 3).
*Definition:* the covariance model sets the level of risk; the multiplier sets the shape of the tail.

**4. Valid when.** The portfolio return is $\mathbf{w}'\mathbf{y}_t$ (fixed weights, simple returns).

**5. Limitations.** A covariance only describes linear dependence; with normal multipliers, joint tail events are understated.

**6. Derive, implement, validate.**
- *Implement:* `portfolio_vol(H, w)`, then the Project 3 functions (`var_t`, `var_es_hs` on residuals) and backtests (`kupiec_test`, `christoffersen_test`, `acerbi_szekely_z2`).
- *Validate:* for a one-asset "portfolio" ($\mathbf{w} = (1, 0, \dots)$), the result equals the univariate VaR.

**Example.** If DCC forecasts a portfolio volatility of 0.70% and the FHS multiplier is 2.6, VaR 99% on EUR 10m is EUR 182k. The same day, CCC forecasts 0.62% (it misses the correlation rise): VaR EUR 161k, 12% lower.

---

### B8. The minimum-variance portfolio: an economic test

**1. Concept.** The portfolio with the lowest forecast variance depends only on the covariance matrix. If a model's covariance forecasts are better, its minimum-variance portfolio should have a lower **realised** volatility out of sample.

**2. Why it was developed.** Engle and Colacito (2006) proposed this as an economic evaluation of covariance models: it measures what a better forecast is worth to an investor, independently of expected returns.

**3. Maths.**

**Variables used in this section:**

- $\mathbf{1}$: vector of ones.
- $\mathbf{w}^{MV}_t$: minimum-variance weights chosen with the forecast $H_t$.

**Equation 17 – Minimum-variance weights (fully invested, short sales allowed):**
$$\mathbf{w}^{MV}_t = \frac{H_t^{-1}\mathbf{1}}{\mathbf{1}'H_t^{-1}\mathbf{1}}.$$
*Definition:* the weights that minimise $\mathbf{w}'H_t\mathbf{w}$ under $\mathbf{1}'\mathbf{w} = 1$ (Lagrange multiplier).

**Equation 18 – Realised volatility of the strategy:** $\hat\sigma^{MV} = \text{std}\left(\mathbf{w}^{MV\prime}_t\mathbf{y}_t\right)$ over the test period, plus the **turnover** $\sum_t\sum_i |w_{i,t} - w_{i,t-1}|$.
*Definition:* the model whose portfolio is really least volatile has the best covariance forecasts for this purpose; turnover measures how costly it is to follow.

**4. Valid when.** $H_t$ is invertible; trading costs are small relative to the risk reduction.

**5. Limitations.**
- Short sales may be unrealistic; a long-only version needs a quadratic optimiser.
- Noisy covariances give extreme weights (the "error maximisation" problem of mean–variance optimisation).
- Low volatility is not the only goal of an investor.

**6. Derive, implement, validate.**
- *Derive:* minimise $\mathbf{w}'H\mathbf{w} - \mu(\mathbf{1}'\mathbf{w} - 1)$; the first-order condition gives $\mathbf{w} \propto H^{-1}\mathbf{1}$.
- *Implement:* `min_variance_weights(H)` with `np.linalg.solve`.
- *Validate:* weights sum to 1; the portfolio variance is below that of any other test portfolio for the same $H$; for two assets it equals the closed form $w_1 = (\sigma_2^2 - \sigma_{12})/(\sigma_1^2 + \sigma_2^2 - 2\sigma_{12})$.

**Example.** Stocks 1.2%, bonds 0.5%, correlation −0.3. Minimum-variance weights: **21% stocks, 79% bonds**, portfolio volatility 0.40%, lower than bonds alone (0.50%) thanks to the negative correlation.

---

## Part C – Summary of the equations

| # | Equation | What it gives |
|---|---|---|
| 1 | $R^{EUR} = (1+R)/(1+R_X) - 1$ | Return in EUR of a USD asset |
| 2 | $P^{adj}_t/P^{adj}_{t-1} - 1$ | Total return of an ETF |
| 3 | $H_t = D_t R_t D_t$ | Covariance = volatilities × correlations |
| 4 | $\sigma^2_p = \mathbf{w}'H\mathbf{w}$ | Portfolio variance |
| 5 | eigenvalues of $H > 0$ | Positive definiteness |
| 6 | $\mathbf{y}_t = H_t^{1/2}\mathbf{z}_t$ | Multivariate model |
| 7 | $\frac1W\sum \mathbf{y}\mathbf{y}'$ | Rolling-window covariance |
| 8 | $(1-\lambda)\mathbf{y}\mathbf{y}' + \lambda H_{t-1}$ | EWMA covariance |
| 9 | $D_t\bar R D_t$ | CCC-GARCH |
| 10 | $(1-a-b)\bar Q + a\boldsymbol\varepsilon\boldsymbol\varepsilon' + bQ_{t-1}$ | DCC recursion |
| 11 | $\text{diag}(Q)^{-1/2}Q\,\text{diag}(Q)^{-1/2}$ | DCC correlation matrix |
| 12 | $-\frac12\sum[\ln|R| + \boldsymbol\varepsilon'R^{-1}\boldsymbol\varepsilon - \boldsymbol\varepsilon'\boldsymbol\varepsilon]$ | DCC correlation log-likelihood |
| 13 | $2[\ell_C(\hat a,\hat b) - \ell_C(0,0)]$ | Test DCC vs CCC |
| 14 | $\ln|H| + \mathbf{y}'H^{-1}\mathbf{y}$ | Matrix QLIKE loss |
| 15 | $\bar d/\widehat{se}(\bar d)$ | Diebold–Mariano |
| 16 | $\sigma_p k_{VaR}$, $\sigma_p k_{ES}$ | Portfolio VaR and ES |
| 17 | $H^{-1}\mathbf{1}/(\mathbf{1}'H^{-1}\mathbf{1})$ | Minimum-variance weights |
| 18 | std of $\mathbf{w}^{MV\prime}_t\mathbf{y}_t$, turnover | Economic evaluation |

---

## Part D – Development plan (step by step, one function at a time)

| Step | Task | New functions | Check |
|---|---|---|---|
| 1 | Data: bond ETF and gold, EUR returns on a common calendar, portfolio weights | `load_yahoo`, `eur_returns` (N assets) | history start dates, roll days, constant FX |
| 2 | Covariance basics: portfolio variance, rolling covariance, correlations over time | `portfolio_variance`, `rolling_cov`, `cov_to_corr` | simulation, `np.cov` |
| 3 | EWMA covariance | `ewma_cov` | pandas `ewm().cov()`, Project 2 diagonal |
| 4 | Univariate GJR-t per asset and standardised residuals; CCC | reuse Project 2, `ccc_cov` | `arch`, residual correlation matrix valid |
| 5 | DCC: recursion, log-likelihood, estimation | `dcc_correlations`, `dcc_loglik`, `fit_dcc` | simulation recovers (a, b); scipy log-density |
| 6 | DCC vs CCC test; correlations in crises | — | LR test; stock–bond sign change |
| 7 | Rolling out-of-sample forecasts for all models | `rolling_dcc` | no look-ahead, positive definite |
| 8 | Matrix QLIKE and Diebold–Mariano | `qlike_matrix` | N = 1 equals Project 2 |
| 9 | Portfolio VaR/ES and backtests | `portfolio_vol` + Project 3 | one-asset portfolio equals univariate |
| 10 | Minimum-variance portfolios | `min_variance_weights` | closed form for 2 assets |
| 11 | Conclusions, report with interview questions | — | — |
