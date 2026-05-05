
# RISK MANAGEMENT — Statistical Distributions

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import norm
import warnings
from scipy.special import gamma
warnings.filterwarnings('ignore')

################################################################
# EXERCISE A3
# Empirical VaR and Expected Shortfall
################################################################
# Problem: Given n=100 loss observations, the 7 largest losses
# are: 12, 15, 17, 19, 23, 27, 32 (thousands of euros).
# Find the empirical VaR and ES at p=96% and p=98.5%
###############################################################

print("=" * 60)
print("EXERCISE A3 — Empirical VaR and Expected Shortfall")
print("=" * 60)

#Setup
n = 100
# Simulate 93 small losses (below the threshold) + 7 known large losses
np.random.seed(42)
small_losses = np.random.uniform(0, 11, size=93)
large_losses = np.array([12, 15, 17, 19, 23, 27, 32])

# Combine and sort all losses
all_losses = np.sort(np.concatenate([small_losses, large_losses]))

print(f"\nTotal observations: {len(all_losses)}")
print(f"7 largest losses (thousands €): {large_losses}")

#Empirical VaR
# VaR_p = L_{ceil(n*p):n}

def empirical_var(losses, p):
    """
    Empirical VaR at confidence level p.
    losses: sorted array of loss observations
    p: confidence level (e.g. 0.96)
    """
    n = len(losses)
    idx = int(np.ceil(n * p)) - 1  # Python is 0-indexed
    return losses[idx]

def empirical_es(losses, p):
    """
    Empirical Expected Shortfall at confidence level p.
    Average of all losses at or above the VaR threshold.
    ES_p = average of L_{ceil(n*p):n}, ..., L_{n:n}
    """
    n = len(losses)
    idx = int(np.ceil(n * p)) - 1
    tail_losses = losses[idx:]
    return tail_losses.mean()

# Compute at p=96%
p1 = 0.96
var_96 = empirical_var(all_losses, p1)
es_96 = empirical_es(all_losses, p1)

print(f"\n--- Confidence Level p = {p1:.0%} ---")
print(f"Empirical VaR:  {var_96:.2f} thousand €")
print(f"Empirical ES:   {es_96:.2f} thousand €")
print(f"(Expected from solution: VaR=17, ES=23.6)")

# Compute at p=98.5%
p2 = 0.985
var_985 = empirical_var(all_losses, p2)
es_985 = empirical_es(all_losses, p2)

print(f"\n--- Confidence Level p = {p2:.1%} ---")
print(f"Empirical VaR:  {var_985:.2f} thousand €")
print(f"Empirical ES:   {es_985:.2f} thousand €")
print(f"(Expected from solution: VaR=27, ES=29.5)")

#Visualization
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for ax, p, var, es, color in zip(
    axes,
    [p1, p2],
    [var_96, var_985],
    [es_96, es_985],
    ['steelblue', 'darkorange']
):
    ax.hist(all_losses, bins=20, color='lightgray', edgecolor='black',
            alpha=0.7, label='Loss distribution')
    ax.axvline(var, color=color, linewidth=2,
               linestyle='--', label=f'VaR({p:.1%}) = {var:.1f}k€')
    ax.axvline(es, color='red', linewidth=2,
               linestyle=':', label=f'ES({p:.1%}) = {es:.1f}k€')
    ax.set_title(f'Empirical VaR & ES at p={p:.1%}', fontsize=13)
    ax.set_xlabel('Loss (thousands €)')
    ax.set_ylabel('Frequency')
    ax.legend()

plt.tight_layout()
plt.savefig('empirical_var_es.png', dpi=150)
plt.show()

# EXERCISE A5
# Extreme Value Distributions — Frechet & Weibull
##############################################################################
# Problem:
# (a) X1,...,Xn ~ F(x) = 1 - x^{-a}, x > 1, a > 0
#     Show that (Mn - 0) / n^{1/a} converges to Frechet Phi_a(x)
#
# (b) X1,...,Xn ~ F(x) = (1-x)^a, x in (0,1), a > 0
#     Show that (Mn - 1) / n^{-1/a} converges to Weibull Psi_a(x)
#
# We simulate both cases and show convergence to the
# limiting extreme value distributions
##############################################################################
 
print("=" * 60)
print("EXERCISE A5 — Extreme Value Distributions")
print("=" * 60)
 
#Parameters
a    = 2.0       # shape parameter
n    = 1000      # sample size per simulation
n_sim = 10000   # number of simulations
 
#Part (a): Frechet
# F(x) = 1 - x^{-a}, x > 1
# Mn / n^{1/a} → Frechet: Phi_a(x) = exp(-x^{-a}), x > 0
# Simulate using inverse CDF: F^{-1}(u) = (1-u)^{-1/a}
 
print("\n--- Part (a): Frechet Distribution ---")
print(f"F(x) = 1 - x^(-{a}), x > 1")
print(f"Normalizing: cn = n^(1/{a}) = {n**(1/a):.4f}, dn = 0")
 
# Simulate maxima
maxima_frechet = []
cn = n ** (1/a)
 
for _ in range(n_sim):
    u = np.random.uniform(0, 1, n)
    # Inverse CDF of F(x) = 1 - x^{-a}
    x = (1 - u) ** (-1/a)
    Mn = np.max(x)
    maxima_frechet.append(Mn / cn)  # normalized maximum
 
maxima_frechet = np.array(maxima_frechet)
 
# Theoretical Frechet CDF: Phi_a(x) = exp(-x^{-a})
x_range = np.linspace(0.01, 5, 500)
frechet_cdf = np.exp(-x_range ** (-a))
 
print(f"Simulated mean of normalized Mn: {maxima_frechet.mean():.4f}")
print(f"Theoretical Frechet mean (Gamma(1-1/a)): {gamma(1 - 1/a):.4f}")
 
#Part (b): Weibull
# F(x) = (1-x)^a, x in (0,1)
# (Mn - 1) / n^{-1/a} → Weibull: Psi_a(x) = exp(-(-x)^a), x < 0
# Inverse CDF: F^{-1}(u) = 1 - u^{1/a}
 
print("\n--- Part (b): Weibull Distribution ---")
print(f"F(x) = (1-x)^{a}, x in (0,1)")
print(f"Normalizing: cn = n^(-1/{a}) = {n**(-1/a):.4f}, dn = 1")
 
maxima_weibull = []
cn_w = n ** (-1/a)
dn_w = 1
 
for _ in range(n_sim):
    u = np.random.uniform(0, 1, n)
    # Inverse CDF of F(x) = (1-x)^a
    x = 1 - u ** (1/a)
    Mn = np.max(x)
    maxima_weibull.append((Mn - dn_w) / cn_w)  # normalized maximum
 
maxima_weibull = np.array(maxima_weibull)
 
# Theoretical Weibull CDF: Psi_a(x) = exp(-(-x)^a), x < 0
x_range_w = np.linspace(-5, 0, 500)
weibull_cdf = np.exp(-(-x_range_w) ** a)
 
print(f"Simulated mean of normalized Mn: {maxima_weibull.mean():.4f}")
 
#Visualization
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
 
# Frechet
ax1 = axes[0]
ax1.hist(maxima_frechet, bins=80, density=True, alpha=0.6,
         color='steelblue', edgecolor='black', label='Simulated normalized Mn')
# Plot theoretical PDF (derivative of CDF)
frechet_pdf = a * x_range**(-a-1) * np.exp(-x_range**(-a))
ax1.plot(x_range, frechet_pdf, 'r-', linewidth=2,
         label=f'Frechet PDF (a={a})')
ax1.set_title(f'Exercise A5(a) — Frechet Distribution\n'
              f'F(x)=1-x^(-{a}), Mn/n^(1/{a}) → Φ_{a}(x)', fontsize=11)
ax1.set_xlabel('Normalized Maximum')
ax1.set_ylabel('Density')
ax1.set_xlim(0, 5)
ax1.legend()
ax1.grid(True, alpha=0.3)
 
# Weibull
ax2 = axes[1]
ax2.hist(maxima_weibull, bins=80, density=True, alpha=0.6,
         color='darkorange', edgecolor='black', label='Simulated normalized Mn')
# Theoretical Weibull PDF
weibull_pdf = a * (-x_range_w)**(a-1) * np.exp(-(-x_range_w)**a)
ax2.plot(x_range_w, weibull_pdf, 'r-', linewidth=2,
         label=f'Weibull PDF (a={a})')
ax2.set_title(f'Exercise A5(b) — Weibull Distribution\n'
              f'F(x)=(1-x)^{a}, (Mn-1)/n^(-1/{a}) → Ψ_{a}(x)', fontsize=11)
ax2.set_xlabel('Normalized Maximum')
ax2.set_ylabel('Density')
ax2.legend()
ax2.grid(True, alpha=0.3)
 
plt.tight_layout()
plt.show()

#######################################################################
# EXERCISE B1
# Portfolio VaR and ES — Two Correlated Stocks
#######################################################################
# Problem: Stocks A and B with normal monthly returns.
# muA=muB=0, sigmaA=0.1, sigmaB=0.2, correlation=0.25
# Current prices: S_A=20€, S_B=40€
# Portfolio: a=2000 shares of A, b=1000 shares of B
# Find VaR and ES at p=99% (Phi(2.33)=0.99)
#######################################################################

print("\n" + "=" * 60)
print("EXERCISE B1 — Portfolio VaR and ES (Correlated Stocks)")
print("=" * 60)

#Parameters
sigma_A = 0.10          # Std of monthly return for stock A
sigma_B = 0.20          # Std of monthly return for stock B
rho_AB  = 0.25          # Correlation between A and B
S_A     = 20            # Current price of stock A (€)
S_B     = 40            # Current price of stock B (€)
a       = 2000          # Number of shares of A
b       = 1000          # Number of shares of B
p_conf  = 0.99          # Confidence level
z_99    = norm.ppf(p_conf)  # Phi^{-1}(0.99) ≈ 2.33

# Portfolio weights (monetary exposure)
w_A = a * S_A   # = 40,000
w_B = b * S_B   # = 40,000

print(f"\nMonetary exposure to A: {w_A:,}€")
print(f"Monetary exposure to B: {w_B:,}€")
print(f"Total portfolio value:  {w_A + w_B:,}€")

#Portfolio variance
# V(Pi) = w_A^2 * sigma_A^2 + w_B^2 * sigma_B^2 + 2*w_A*w_B*rho*sigma_A*sigma_B
var_portfolio = (w_A**2 * sigma_A**2 +
                 w_B**2 * sigma_B**2 +
                 2 * w_A * w_B * rho_AB * sigma_A * sigma_B)

std_portfolio = np.sqrt(var_portfolio)

print(f"\nPortfolio variance:  {var_portfolio:,.0f}")
print(f"Portfolio std dev:   {std_portfolio:,.2f}€")

# --- VaR and ES ---
# Since E(Pi) = 0 (both stocks have mean return = 0):
# VaR_p(L) = -E(Pi) + sqrt(V(Pi)) * Phi^{-1}(p) = std * z
# ES_p(L)  = -E(Pi) + sqrt(V(Pi)) * phi(Phi^{-1}(p)) / (1-p)

phi_z99 = norm.pdf(z_99)   # phi(2.33) ≈ 0.0267

var_portfolio_risk = std_portfolio * z_99
es_portfolio_risk  = std_portfolio * phi_z99 / (1 - p_conf)

print(f"\n--- Portfolio Risk at p = {p_conf:.0%} ---")
print(f"VaR_{p_conf:.0%}:  {var_portfolio_risk:,.2f}€")
print(f"ES_{p_conf:.0%}:   {es_portfolio_risk:,.2f}€")
print(f"(Expected from solution: VaR≈22,829€, ES≈26,111€)")

#Simulation to verify analytically
np.random.seed(42)
n_sim = 100_000

# Simulate correlated returns using Cholesky decomposition
mean = [0, 0]
cov = [[sigma_A**2, rho_AB * sigma_A * sigma_B],
       [rho_AB * sigma_A * sigma_B, sigma_B**2]]

returns = np.random.multivariate_normal(mean, cov, n_sim)
R_A = returns[:, 0]
R_B = returns[:, 1]

# Portfolio profit
portfolio_profit = w_A * R_A + w_B * R_B
portfolio_loss   = -portfolio_profit

# Simulated VaR and ES
var_sim = np.quantile(portfolio_loss, p_conf)
es_sim  = portfolio_loss[portfolio_loss >= var_sim].mean()

print(f"\n--- Monte Carlo Verification ({n_sim:,} simulations) ---")
print(f"Simulated VaR_{p_conf:.0%}: {var_sim:,.2f}€")
print(f"Simulated ES_{p_conf:.0%}:  {es_sim:,.2f}€")

# --- Visualization ---
fig, ax = plt.subplots(figsize=(12, 5))
ax.hist(portfolio_loss, bins=100, color='steelblue',
        edgecolor='black', alpha=0.6, label='Portfolio Loss Distribution')
ax.axvline(var_sim, color='red', linewidth=2,
           linestyle='--', label=f'VaR(99%) = {var_sim:,.0f}€')
ax.axvline(es_sim, color='darkorange', linewidth=2,
           linestyle=':', label=f'ES(99%) = {es_sim:,.0f}€')
ax.set_title('Portfolio Loss Distribution — 2 Correlated Stocks\n'
             f'(σ_A={sigma_A}, σ_B={sigma_B}, ρ={rho_AB})', fontsize=13)
ax.set_xlabel('Monthly Loss (€)')
ax.set_ylabel('Frequency')
ax.legend()
plt.tight_layout()
plt.savefig('portfolio_var_es.png', dpi=150)
plt.show()


##################################################################
# EXERCISE B6
# Portfolio Hedging with Put Options — Delta Neutral Strategy
##################################################################
# Problem:
# Portfolio: n=100 shares (S0=80, R~N(0, 0.02^2))
#            m=240 put options (Delta = -0.5)
#
# (a) Approximate daily portfolio profit
# (b) Distribution of daily profit
# (c) VaR99.9% with and without hedging
# (d) Find m* for Delta-neutral portfolio and its VaR
###################################################################

print("\n" + "=" * 60)
print("EXERCISE B6 — Portfolio Hedging with Put Options")
print("=" * 60)
 
#Parameters 
n_shares = 100       # number of shares
m_options = 240      # number of put options
S_t = 80             # current stock price
Delta_t = -0.5       # Delta of put option
sigma = 0.02         # daily std of return (h*sigma_e = 0.02^2 so sigma=0.02)
mu = 0               # daily mean return
p_conf = 0.999       # confidence level
z_999 = norm.ppf(p_conf)  # Phi^{-1}(0.999) = 3
 
print(f"\nParameters:")
print(f"Shares: n={n_shares}, Stock price: S_t={S_t}")
print(f"Put options: m={m_options}, Delta: {Delta_t}")
print(f"Daily return: R_t ~ N({mu}, {sigma}^2)")
print(f"Confidence level: {p_conf} (z={z_999})")
 
#Part (a) & (b): Portfolio profit approximation
# Pi_t ≈ S_t * R_t * (n + m * Delta_t)
# This follows N(0, S_t^2 * (n + m*Delta_t)^2 * sigma^2)
 
delta_pos = n_shares + m_options * Delta_t   # net delta position
print(f"\n--- Part (a) & (b) ---")
print(f"Net delta position: n + m*Delta = {n_shares} + {m_options}*{Delta_t} = {delta_pos}")
 
E_profit = S_t * delta_pos * mu
V_profit = (S_t * delta_pos) ** 2 * sigma ** 2
std_profit = np.sqrt(V_profit)
 
print(f"E(Pi_t) = S_t*(n+m*Delta)*mu = {E_profit}")
print(f"Std(Pi_t) = S_t*|n+m*Delta|*sigma = {S_t}*|{delta_pos}|*{sigma} = {std_profit}")
print(f"Pi_t ~ N({E_profit}, {std_profit}^2)")
 
#Part (c): VaR with and without hedging
print(f"\n--- Part (c): VaR99.9% ---")
 
#With hedging (m=240 put options)
VaR_hedged = -E_profit + std_profit * z_999
print(f"VaR99.9% WITH hedging (m={m_options}): {VaR_hedged:.2f}")
print(f"(Expected from solution: 96)")
 
# Without hedging (m=0)
std_no_hedge = S_t * n_shares * sigma
VaR_no_hedge = std_no_hedge * z_999
print(f"VaR99.9% WITHOUT hedging (m=0): {VaR_no_hedge:.2f}")
print(f"(Expected from solution: 480)")
 
print(f"\nRisk reduction from hedging: {VaR_no_hedge - VaR_hedged:.2f} ({(1 - VaR_hedged/VaR_no_hedge)*100:.1f}%)")
 
# Part (d): Delta-neutral portfolio 
# For Delta-neutral: n + m* * Delta_t = 0
# m* = -n / Delta_t
print(f"\n--- Part (d): Delta-Neutral Portfolio ---")
m_star = -n_shares / Delta_t
print(f"Delta-neutral condition: n + m*Delta = 0")
print(f"m* = -n/Delta = -{n_shares}/{Delta_t} = {m_star}")
print(f"VaR99.9% (Delta-neutral) = 0 (zero risk)")
print(f"(Expected from solution: m*=200, VaR=0)")
 
# Simulation to verify
np.random.seed(42)
n_sim = 100000
R_t = np.random.normal(mu, sigma, n_sim)
 
profit_hedged   = S_t * R_t * (n_shares + m_options * Delta_t)
profit_no_hedge = S_t * R_t * n_shares
profit_neutral  = S_t * R_t * (n_shares + m_star * Delta_t)
 
loss_hedged   = -profit_hedged
loss_no_hedge = -profit_no_hedge
loss_neutral  = -profit_neutral
 
VaR_hedged_sim   = np.quantile(loss_hedged,   p_conf)
VaR_no_hedge_sim = np.quantile(loss_no_hedge, p_conf)
VaR_neutral_sim  = np.quantile(loss_neutral,  p_conf)
 
print(f"\n--- Monte Carlo Verification ({n_sim:,} simulations) ---")
print(f"VaR99.9% WITH hedging:     {VaR_hedged_sim:.2f}")
print(f"VaR99.9% WITHOUT hedging:  {VaR_no_hedge_sim:.2f}")
print(f"VaR99.9% DELTA NEUTRAL:    {VaR_neutral_sim:.2f}")
 
# Visualization 
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
 
# Loss distributions
ax1 = axes[0]
ax1.hist(loss_no_hedge, bins=100, density=True, alpha=0.5,
         color='red', label=f'No hedge (VaR={VaR_no_hedge_sim:.0f})')
ax1.hist(loss_hedged, bins=100, density=True, alpha=0.5,
         color='steelblue', label=f'Hedged m={m_options} (VaR={VaR_hedged_sim:.0f})')
ax1.hist(loss_neutral, bins=100, density=True, alpha=0.5,
         color='green', label=f'Delta neutral m*={m_star:.0f} (VaR≈{VaR_neutral_sim:.0f})')
ax1.axvline(VaR_no_hedge_sim, color='red', linewidth=2, linestyle='--')
ax1.axvline(VaR_hedged_sim, color='steelblue', linewidth=2, linestyle='--')
ax1.set_title('Exercise B6 — Loss Distributions\nHedging vs No Hedging vs Delta Neutral',
              fontsize=11)
ax1.set_xlabel('Daily Loss (€)')
ax1.set_ylabel('Density')
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3)
 
# VaR comparison bar chart
ax2 = axes[1]
strategies = ['No Hedge', f'Hedged\n(m={m_options})', f'Delta Neutral\n(m*={m_star:.0f})']
vars_vals   = [VaR_no_hedge_sim, VaR_hedged_sim, VaR_neutral_sim]
colors      = ['red', 'steelblue', 'green']
bars = ax2.bar(strategies, vars_vals, color=colors,
               edgecolor='black', alpha=0.8)
ax2.set_title('VaR99.9% by Hedging Strategy', fontsize=12)
ax2.set_ylabel('VaR (€)')
for bar, val in zip(bars, vars_vals):
    ax2.text(bar.get_x() + bar.get_width()/2,
             bar.get_height() + 2,
             f'{val:.1f}', ha='center', fontsize=10, fontweight='bold')
ax2.grid(True, alpha=0.3, axis='y')
 
plt.tight_layout()
plt.savefig('b6_portfolio_hedging.png', dpi=150)
plt.show()

######################################################################
# EXERCISE A1 — Extended
# VaR Subadditivity: Diversification Effect
#####################################################################
# Problem: X1~N(mu1, sigma1^2), X2~N(mu2, sigma2^2) with
# correlation rho. Show that:
# VaR_p(X1+X2) <= VaR_p(X1) + VaR_p(X2)
# with equality only when rho=1
# This is the mathematical proof that diversification reduces risk
#####################################################################

print("\n" + "=" * 60)
print("EXERCISE A1 — VaR Subadditivity & Diversification Effect")
print("=" * 60)

# Parameters
mu1    = 0
mu2    = 0
sigma1 = 0.10
sigma2 = 0.20
p_conf = 0.99
z      = norm.ppf(p_conf)

# Range of correlation values
rhos = np.linspace(-1, 1, 200)

#  VaR of individual positions 
var1 = mu1 + sigma1 * z
var2 = mu2 + sigma2 * z
sum_individual_var = var1 + var2

print(f"\nVaR_99%(X1) alone:         {var1:.4f}")
print(f"VaR_99%(X2) alone:         {var2:.4f}")
print(f"Sum of individual VaRs:    {sum_individual_var:.4f}")

# VaR of combined portfolio across correlations 
var_combined = []
for rho in rhos:
    var_sum = np.sqrt(sigma1**2 + sigma2**2 + 2 * rho * sigma1 * sigma2)
    var_combined.append(mu1 + mu2 + var_sum * z)

var_combined = np.array(var_combined)

# Diversification benefit
diversification_benefit = sum_individual_var - var_combined

print(f"\nAt rho = -1.0: combined VaR = {var_combined[0]:.4f}  "
      f"| Diversification benefit = {diversification_benefit[0]:.4f}")
print(f"At rho =  0.0: combined VaR = {var_combined[100]:.4f}  "
      f"| Diversification benefit = {diversification_benefit[100]:.4f}")
print(f"At rho = +1.0: combined VaR = {var_combined[-1]:.4f}  "
      f"| Diversification benefit = {diversification_benefit[-1]:.4f}")
print(f"\nAt rho=1, combined VaR = sum of individual VaRs: "
      f"{np.isclose(var_combined[-1], sum_individual_var, atol=0.001)}")

# Visualization 
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: VaR vs Correlation
ax1 = axes[0]
ax1.plot(rhos, var_combined, color='steelblue', linewidth=2,
         label='VaR(X1+X2) — Combined')
ax1.axhline(sum_individual_var, color='red', linewidth=2,
            linestyle='--', label='VaR(X1) + VaR(X2) — Sum of individual')
ax1.fill_between(rhos, var_combined, sum_individual_var,
                 alpha=0.2, color='green', label='Diversification benefit')
ax1.set_title('VaR Subadditivity:\nCombined vs Individual VaR', fontsize=13)
ax1.set_xlabel('Correlation ρ')
ax1.set_ylabel('VaR (99%)')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Diversification Benefit vs Correlation
ax2 = axes[1]
ax2.plot(rhos, diversification_benefit, color='green', linewidth=2)
ax2.axhline(0, color='black', linewidth=0.8, linestyle='--')
ax2.set_title('Diversification Benefit vs Correlation\n'
              '(Sum of VaRs − Combined VaR)', fontsize=13)
ax2.set_xlabel('Correlation ρ')
ax2.set_ylabel('Diversification Benefit')
ax2.fill_between(rhos, diversification_benefit, 0,
                 where=(diversification_benefit > 0),
                 alpha=0.3, color='green', label='Benefit > 0')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('var_subadditivity.png', dpi=150)
plt.show()
print("Chart saved: var_subadditivity.png")

#################################################################
# SUMMARY
#################################################################
print("\n" + "=" * 60 )
print("SUMMARY OF RESULTS")
print("=" * 60)
print(f"\nExercise A3 — Empirical Risk Measures:")
print(f"  VaR(96%)  = {var_96:.1f}k€  |  ES(96%)  = {es_96:.2f}k€")
print(f"  VaR(98.5%)= {var_985:.1f}k€  |  ES(98.5%)= {es_985:.2f}k€")
print(f"\nExercise B1 — Portfolio Risk (Correlated Stocks):")
print(f"  Analytical VaR(99%) = {var_portfolio_risk:,.2f}€")
print(f"  Analytical ES(99%)  = {es_portfolio_risk:,.2f}€")
print(f"  Simulated  VaR(99%) = {var_sim:,.2f}€  (Monte Carlo)")
print(f"  Simulated  ES(99%)  = {es_sim:,.2f}€  (Monte Carlo)")

print(f"\nExercise A1 — VaR Subadditivity:")
print(f"  Sum of individual VaRs: {sum_individual_var:.4f}")
print(f"  Combined VaR at ρ=0:    {var_combined[100]:.4f}")
print(f"  Combined VaR at ρ=1:    {var_combined[-1]:.4f} (equals sum — no diversification)")
print(f"  Max diversification benefit at ρ=-1: {diversification_benefit[0]:.4f}")

print("\nOutput files saved:")
print("  empirical_var_es.png")
print("  portfolio_var_es.png")
print("  var_subadditivity.png")
# EXERCISE G1 (Γ1)
# Bond Default Probability & Fair Value
# Hazard Rate: lambda(t) = 3*theta*t^2
# Interest Rate: r(t) = a + 2*b*t

 
print("\n" + "=" * 60)
print("EXERCISE G1 — Bond Default Probability & Fair Value")
print("=" * 60)
 
# Parameters from solution
T    = 2       # bond maturity (years)
r_B  = 0.10   # market yield (10%)
a    = 0.01   # interest rate parameter
b    = 0.005  # interest rate parameter
 
print(f"\nParameters: T={T}, r_B={r_B}, a={a}, b={b}")
 
# Part (a): Survival probability P(tau > T) 
# lambda(t) = 3*theta*t^2
# Integral of lambda from 0 to T = theta*T^3
# P(tau > T) = exp(-theta*T^3)
 
print("\n--- Part (a): Survival Probability ---")
print("P(tau > T) = exp(-theta * T^3)")
 
# Part (b): Present value B(0,T)^{-1} 
# r(t) = a + 2*b*t
# Integral of r from 0 to T = a*T + b*T^2
pv_factor = np.exp(-(a * T + b * T**2))
print(f"\n--- Part (b): Present Value Factor ---")
print(f"B(0,T)^(-1) = exp(-(a*T + b*T^2))")
print(f"B(0,T)^(-1) = exp(-({a}*{T} + {b}*{T}^2)) = {pv_factor:.6f}")
 
# Part (c): Fair value Z(0,T) 
# Z(0,T) = B(0,T)^{-1} * P(tau > T) = exp(-(a*T+b*T^2)) * exp(-theta*T^3)
print(f"\n--- Part (c): Fair Value of Bond ---")
print(f"Z(0,T) = B(0,T)^(-1) * P(tau > T)")
print(f"Z(0,T) = exp(-(a*T+b*T^2)) * exp(-theta*T^3)")
 
# Part (d): Implied theta from market yield r_B 
# Z(0,T) * (1 + r_B * T) = 1
# exp(-(a*T+b*T^2)) * exp(-theta*T^3) * (1 + r_B*T) = 1
# theta = [ln(1 + r_B*T) - (a*T + b*T^2)] / T^3
 
theta_hat = (np.log(1 + r_B * T) - (a * T + b * T**2)) / T**3
print(f"\n--- Part (d): Implied Theta ---")
print(f"theta_hat = [ln(1 + r_B*T) - (a*T + b*T^2)] / T^3")
print(f"theta_hat = [ln({1 + r_B*T}) - ({a*T + b*T**2})] / {T**3}")
print(f"theta_hat = {theta_hat:.6f}")
print(f"(Expected from solution: 0.01779)")
 
# Part (e): Estimated survival probability 
survival_prob = np.exp(-theta_hat * T**3)
print(f"\n--- Part (e): Estimated Survival Probability ---")
print(f"P(tau > T) = exp(-theta_hat * T^3)")
print(f"P(tau > T) = exp(-{theta_hat:.6f} * {T**3})")
print(f"P(tau > T) = {survival_prob:.4f}")
print(f"(Expected from solution: 0.8673)")
 
# Visualization: Survival probability vs theta
thetas = np.linspace(0, 0.05, 200)
survival_probs = np.exp(-thetas * T**3)
fair_values = pv_factor * survival_probs
 
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
 
ax1 = axes[0]
ax1.plot(thetas, survival_probs, color='steelblue', linewidth=2)
ax1.axvline(theta_hat, color='red', linewidth=2, linestyle='--',
            label=f'Implied θ = {theta_hat:.4f}')
ax1.axhline(survival_prob, color='green', linewidth=1.5, linestyle=':',
            label=f'P(τ>T) = {survival_prob:.4f}')
ax1.set_title(f'Exercise G1 — Survival Probability vs θ\n'
              f'λ(t)=3θt², T={T} years', fontsize=11)
ax1.set_xlabel('θ (default intensity parameter)')
ax1.set_ylabel('P(τ > T)')
ax1.legend()
ax1.grid(True, alpha=0.3)
 
ax2 = axes[1]
ax2.plot(thetas, fair_values, color='darkorange', linewidth=2)
ax2.axvline(theta_hat, color='red', linewidth=2, linestyle='--',
            label=f'Implied θ = {theta_hat:.4f}')
ax2.axhline(1/(1 + r_B*T), color='green', linewidth=1.5, linestyle=':',
            label=f'Market price = {1/(1+r_B*T):.4f}')
ax2.set_title(f'Exercise G1 — Bond Fair Value vs θ\n'
              f'r_B={r_B}, a={a}, b={b}', fontsize=11)
ax2.set_xlabel('θ (default intensity parameter)')
ax2.set_ylabel('Z(0,T) — Fair Value')
ax2.legend()
ax2.grid(True, alpha=0.3)
 
plt.tight_layout()
plt.savefig('g1_bond_default.png', dpi=150)
plt.show()
 
#############################################################
# EXERCISE G4 (Γ4)
# CDS Pricing with Constant Hazard Rate
#############################################################
# Problem:
# CDS with maturity T, credit spread S0, continuous premium,
# recovery rate R=40%, notional K, constant hazard rate lambda
# and constant interest rate r
#
# (a) Find CDS present value formula
# (b) Find implied lambda from market spread (breakeven condition)
#     Annual premium: K*S0 = 6000 => S0 = 0.06 (600bp)
#     Find implied lambda and P(tau > 4)
#####################################################################
 
print("\n" + "=" * 60)
print("EXERCISE G4 — CDS Pricing with Constant Hazard Rate")
print("=" * 60)
 
# Parameters
K   = 100000   # notional amount
R   = 0.40     # recovery rate
S0  = 0.06     # credit spread (6000/100000 per year)
T   = 4        # maturity in years
r   = 0.0      # risk-free rate (not specified so assume 0 for simplicity)
 
print(f"\nParameters:")
print(f"Notional K={K:,}€, Recovery R={R}")
print(f"Annual premium: K*S0 = {K*S0:,.0f}€ => S0 = {S0} ({S0*10000:.0f}bp)")
print(f"Maturity T={T} years")
 
# Part (a): CDS Present Value
# V(0) = K * ((1-R)*lambda - S0) * (1 - exp(-(r+lambda)*T)) / (r+lambda)
# For r=0: V(0) = K * ((1-R)*lambda - S0) * (1 - exp(-lambda*T)) / lambda
 
print(f"\n--- Part (a): CDS Present Value Formula ---")
print(f"V(0) = K * ((1-R)*λ - S0) * (1-exp(-(r+λ)*T)) / (r+λ)")
 
# Part (b): Breakeven condition V(0) = 0 
# (1-R)*lambda - S0 = 0
# lambda = S0 / (1-R)
print(f"\n--- Part (b): Breakeven Credit Spread ---")
print(f"V(0) = 0 => (1-R)*λ = S0")
print(f"λ_hat = S0 / (1-R) = {S0} / {1-R} = {S0/(1-R):.4f}")
 
lambda_hat = S0 / (1 - R)
print(f"\nImplied hazard rate: λ_hat = {lambda_hat:.4f}")
print(f"(Expected from solution: 0.1)")
 
# Survival probability over T years
survival_cds = np.exp(-lambda_hat * T)
print(f"\nP(τ > {T}) = exp(-λ_hat*T) = exp(-{lambda_hat}*{T}) = {survival_cds:.4f}")
print(f"(Expected from solution: e^(-0.4) ≈ 67%)")
 
# CDS value as function of lambda
lambdas = np.linspace(0.001, 0.30, 300)
 
# For r=0
def cds_value(lam, K, R, S0, T, r=0):
    if r == 0:
        return K * ((1-R)*lam - S0) * T * np.exp(-lam*T) / 1
    else:
        return K * ((1-R)*lam - S0) * (1 - np.exp(-(r+lam)*T)) / (r+lam)
 
# Approximate formula V(0) ≈ K*((1-R)*λ - S0)*T for small lambda
cds_vals_approx = K * ((1-R)*lambdas - S0) * T
 
# Survival probability over time
t_range = np.linspace(0, 10, 200)
survival_curve = np.exp(-lambda_hat * t_range)
 
# Visualization
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
 
# CDS value vs lambda
ax1 = axes[0]
ax1.plot(lambdas, cds_vals_approx / 1000, color='steelblue', linewidth=2,
         label='CDS Value (approx)')
ax1.axhline(0, color='black', linewidth=1, linestyle='--')
ax1.axvline(lambda_hat, color='red', linewidth=2, linestyle='--',
            label=f'Breakeven λ = {lambda_hat:.3f}')
ax1.fill_between(lambdas, cds_vals_approx/1000, 0,
                 where=(lambdas < lambda_hat),
                 alpha=0.2, color='red', label='Protection buyer loses')
ax1.fill_between(lambdas, cds_vals_approx/1000, 0,
                 where=(lambdas > lambda_hat),
                 alpha=0.2, color='green', label='Protection buyer gains')
ax1.set_title(f'Exercise G4 — CDS Value vs Hazard Rate\n'
              f'S0={S0} ({S0*10000:.0f}bp), R={R}, K={K:,}€', fontsize=11)
ax1.set_xlabel('Hazard Rate λ')
ax1.set_ylabel("CDS Value (thousands €)")
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3)
 
# Survival curve
ax2 = axes[1]
ax2.plot(t_range, survival_curve, color='steelblue', linewidth=2,
         label=f'P(τ > t) = exp(-{lambda_hat}t)')
ax2.axvline(T, color='red', linewidth=2, linestyle='--',
            label=f'T={T} years')
ax2.axhline(survival_cds, color='green', linewidth=1.5, linestyle=':',
            label=f'P(τ > {T}) = {survival_cds:.3f} ({survival_cds*100:.1f}%)')
ax2.set_title(f'Exercise G4 — Survival Probability Curve\n'
              f'λ = {lambda_hat} (implied from CDS spread)', fontsize=11)
ax2.set_xlabel('Time (years)')
ax2.set_ylabel('Survival Probability P(τ > t)')
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)
 
plt.tight_layout()
plt.savefig('g4_cds_pricing.png', dpi=150)
plt.show()