# # Quantitative Risk Management - Statistical Distributions (Python)



## Overview

Python implementations of quantitative risk management techniques using statistical distributions. The project covers foundational market risk measures and advanced topics including extreme value theory, derivative hedging, and credit risk pricing.

All exercises are based on the academic curriculum of the course **Risk Management (2023)** , University of Piraeus, and are implemented with Monte Carlo verification and full visualizations.


### Exercise A1 - VaR Subadditivity and the Diversification Effect

**Problem:** Given two normally distributed loss variables X₁ ~ N(μ₁, σ₁²) and X₂ ~ N(μ₂, σ₂²) with correlation ρ, compare the Value at Risk of the combined portfolio VaR_p(X₁+X₂) against the sum of individual VaRs VaR_p(X₁) + VaR_p(X₂).

**Key result:** Since ρ ≤ 1, it always holds that:

```
VaR_p(X₁+X₂) ≤ VaR_p(X₁) + VaR_p(X₂)
```

This is known as **subadditivity** - the capital reserve required for a combined portfolio is always less than or equal to the sum of capital reserves for the individual portfolios separately. Equality holds only when ρ = 1, meaning the two losses are perfectly correlated (no diversification benefit).

**Implementation:** The diversification benefit is plotted across all correlations from ρ = -1 to ρ = +1, verified with 100,000 Monte Carlo simulations. The result confirms the mathematical proof and demonstrates why diversification is a fundamental principle of risk management.

### Exercise A3 - Empirical VaR and Expected Shortfall

**Problem:** A portfolio has n = 100 loss observations over independent time periods. The 7 largest losses (in thousands of euros) are: 12, 15, 17, 19, 23, 27, 32. Calculate the empirical Value at Risk and Expected Shortfall at confidence levels p = 96% and p = 98.5%.

**Key formulas:**

- **Empirical VaR:** VaR̂_p(L) = L_{⌈n·p⌉:n} — the order statistic at position ⌈n·p⌉
- **Empirical ES:** EŜ_p(L) = average of all losses at or above the VaR threshold

**Results:**

| Confidence Level | Empirical VaR | Empirical ES |
|---|---|---|
| p = 96% | 17k€ | 23.6k€ |
| p = 98.5% | 27k€ | 29.5k€ |

**Why ES matters:** VaR tells you the minimum loss in the worst p% of scenarios. Expected Shortfall tells you the average loss given that you are already in the tail — it is a more complete picture of tail risk and is the preferred risk measure under Basel III/IV regulatory frameworks.


### Exercise A5 - Extreme Value Distributions: Frechet and Weibull

**Problem:** Given n independent identically distributed random variables, identify the limiting distribution of the normalized sample maximum Mₙ = max{X₁,...,Xₙ} as n → ∞.

**(a) Frechet Distribution:**
For X ~ F(x) = 1 - x⁻ᵃ, x > 1, a > 0 (heavy-tailed Pareto-type):

With normalizing constants cₙ = n^(1/a) and dₙ = 0:
```
P((Mₙ - 0) / n^(1/a) ≤ x) → exp(-x⁻ᵃ)    [Frechet distribution Φ_a(x)]
```

**(b) Weibull Distribution:**
For X ~ F(x) = (1-x)ᵃ, x ∈ (0,1), a > 0 (bounded support):

With normalizing constants cₙ = n^(-1/a) and dₙ = 1:
```
P((Mₙ - 1) / n^(-1/a) ≤ x) → exp(-(-x)ᵃ)    [Weibull distribution Ψ_a(x)]
```

**Why this matters:** Extreme Value Theory (EVT) provides the statistical framework for modelling rare but catastrophic events - floods, market crashes, insurance losses. The Frechet distribution arises from heavy-tailed data (financial returns, insurance claims), while the Weibull arises from bounded distributions. Together with the Gumbel, these three form the Generalized Extreme Value (GEV) family.

**Implementation:** Both convergence results are verified with 10,000 simulations of n = 1,000 samples, showing that the empirical distribution of normalized maxima matches the theoretical limiting distribution.


### Exercise B1 - Portfolio VaR and ES for Two Correlated Stocks

**Problem:** Stocks A and B have normally distributed monthly returns with zero mean, standard deviations σ_A = 0.10 and σ_B = 0.20, and correlation ρ_AB = 0.25. Current prices are S_A = 20€ and S_B = 40€. A portfolio holds a = 2,000 shares of A and b = 1,000 shares of B. Calculate the monthly VaR and ES at p = 99%.

**Portfolio variance:**
```
V(Π) = (a·S_A)²·σ_A² + (b·S_B)²·σ_B² + 2·a·S_A·b·S_B·ρ_AB·σ_A·σ_B
```

**Results (analytical):**
- Monthly VaR₉₉% ≈ **22,829€**
- Monthly ES₉₉% ≈ **26,111€**

**Implementation:** The analytical results are verified against 100,000 Monte Carlo simulations using correlated return generation via Cholesky decomposition of the covariance matrix.


### Exercise B6 - Portfolio Hedging with Put Options and Delta-Neutral Strategy

**Problem:** A portfolio holds n = 100 shares of a company (current price S_t = 80€, daily return R_t ~ N(0, 0.02²)) and m = 240 put options on those shares (Delta Δ_t = -0.5).

**(a) Portfolio profit approximation:**
```
Π_t ≈ S_t · R_t · (n + m · Δ_t)
```

**(b) Distribution of daily profit:**
Since R_t is normally distributed, Π_t follows:
```
Π_t ~ N(0, S_t²·(n + m·Δ_t)²·σ²)
```

**(c) VaR₉₉.₉% with and without hedging:**

| Strategy | Net Delta | VaR₉₉.₉% |
|---|---|---|
| No hedge (m=0) | 100 | 480€ |
| Hedged (m=240) | 20 | 96€ |
| Delta neutral (m*=200) | 0 | ≈ 0€ |

**(d) Delta-neutral portfolio:**
Setting n + m* · Δ_t = 0 gives m* = -n/Δ_t = 200 options. The portfolio becomes essentially riskless, reducing VaR to approximately zero.

**Why this matters:** Delta hedging is the core mechanism behind options risk management. The delta-neutral condition eliminates first-order sensitivity to price movements. This is foundational to derivatives pricing theory and is used daily by options traders and risk managers at banks and hedge funds.

**Implementation:** All three strategies are verified with 100,000 Monte Carlo simulations and compared visually.

---

### Exercise G1 - Bond Default Probability and Fair Value (Hazard Rate Model)

**Problem:** A zero-coupon bond (maturity T = 2 years) is issued by an entity with hazard rate λ(t) = 3θt² and market interest rate r(t) = a + 2bt (a = 0.01, b = 0.005). The bond trades at a market yield of r_B = 10%.

**Key results derived:**

**(a) Survival probability** (probability the bond is fully repaid):
```
P(τ > T) = exp(-θ·T³)
```

**(b) Present value discount factor:**
```
B(0,T)⁻¹ = exp(-(a·T + b·T²))
```

**(c) Fair value of the bond:**
```
Z(0,T) = exp(-(a·T + b·T²)) · exp(-θ·T³)
```

**(d) Implied θ from market yield r_B:**

Using the pricing condition Z(0,T) · (1 + r_B · T) = 1:
```
θ̂ = [ln(1 + r_B·T) - (a·T + b·T²)] / T³ ≈ 0.01779
```

**(e) Estimated survival probability:**
```
P̂(τ > T) = exp(-θ̂·T³) ≈ 0.8673 (86.73%)
```

**Why this matters:** The hazard rate (or default intensity) model is the standard framework for pricing credit-risky bonds and credit derivatives. Extracting implied default probabilities from market prices is a core task in credit risk management and fixed income analysis.


### Exercise G4 - CDS Pricing and Implied Default Probability

**Problem:** A Credit Default Swap (CDS) with maturity T = 4 years, recovery rate R = 40%, notional K = 100,000€, and continuous premium payments. The annual premium paid is K·S₀ = 6,000€, giving a credit spread of S₀ = 0.06 (600 basis points). The reference entity has a constant hazard rate λ and the risk-free rate is constant at r.

**What is a CDS?** A Credit Default Swap is a financial contract where the protection buyer pays a periodic premium (the credit spread) to the protection seller. In return, if the reference entity defaults, the seller pays (1-R) times the notional to the buyer. It functions as insurance against default — widely used by banks, hedge funds, and institutional investors to manage credit exposure.

**(a) CDS present value formula:**
```
V(0) = K · ((1-R)·λ - S₀) · (1 - exp(-(r+λ)·T)) / (r+λ)
```

**(b) Breakeven credit spread (fair value condition V(0) = 0):**

Setting V(0) = 0 gives the neutral condition:
```
(1-R)·λ = S₀  →  λ̂ = S₀/(1-R) = 0.06/0.60 = 0.10
```

**Probability of full repayment over 4 years:**
```
P̂(τ > 4) = exp(-λ̂·T) = exp(-0.4) ≈ 67%
```

**Why this matters:** CDS pricing and implied default probability extraction from market spreads is fundamental to credit risk management. The breakeven spread formula connects market prices to the underlying credit quality of the reference entity — the same logic used by banks to price loans and by investors to assess sovereign and corporate credit risk.

---

## Key Concepts Demonstrated

| Concept | Exercise | Relevance |
|---|---|---|
| Value at Risk (VaR) | A1, A3, B1, B6 | Core risk measure — Basel III/IV |
| Expected Shortfall (ES) | A3, B1 | Preferred tail risk measure — Basel III/IV |
| VaR Subadditivity | A1 | Proves diversification reduces risk |
| Monte Carlo Simulation | A1, B1, B6, A5 | Verification of analytical results |
| Extreme Value Theory | A5 | Modelling catastrophic tail events |
| Frechet / Weibull distributions | A5 | GEV family - insurance and finance |
| Delta Hedging | B6 | Options risk management |
| Delta-Neutral Portfolio | B6 | Eliminating first-order price risk |
| Hazard Rate Model | G1, G4 | Credit risk - bond and CDS pricing |
| Implied Default Probability | G1, G4 | Extracting credit quality from prices |
| Credit Default Swap (CDS) | G4 | Credit derivative pricing |


## How to Run

Install dependencies:
```bash
pip install numpy scipy matplotlib
```

Run the scripts:
```bash
python risk_management.py
```


**Oresti Janko**
BSc Statistics and Insurance Science - University of Piraeus
Focus: Quantitative risk management, statistical modelling, Python
