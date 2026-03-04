# Structured Products Simulator – Phoenix Autocallable

A **Monte Carlo simulator for Phoenix structured products**, built with **Python and Streamlit**.

The application allows users to simulate multi-asset structured products (e.g. **Phoenix Autocallables**) and analyze their **payoff distribution, pricing, and sensitivities**.

This project is designed as an **educational and analytical framework** to understand the mechanics and risk profile of structured products used in **Equity Derivatives desks**.

---

# Overview

Structured products such as **Phoenix Autocallables** are widely used in equity derivatives markets to offer investors enhanced yield with conditional capital protection.

Their payoff structure typically includes:

- Periodic coupon payments
- Autocall mechanisms
- Memory coupon features
- Downside protection barriers
- Basket structures (Worst-of, Best-of, Average, Weighted)

Because these products are **path-dependent and often multi-asset**, closed-form pricing solutions are generally not available.  
Instead, **Monte Carlo simulation** is used to estimate expected discounted payoffs.

---

# Key Features

### Product Mechanics

The simulator models the full lifecycle of a Phoenix structured product:

- Coupon payments conditional on barrier levels
- Memory coupon accumulation
- Early redemption through autocall triggers
- Capital protection via maturity barriers

### Multi-Asset Basket Support

The model supports several basket constructions:

- **Worst-of**
- **Best-of**
- **Arithmetic average**
- **Weighted basket**

### Market Data Integration

The application can automatically retrieve:

- Spot prices
- Historical volatility
- Correlation matrices

using **Yahoo Finance (yfinance)**.

Users can also manually specify market parameters.

### Monte Carlo Pricing

Asset paths are simulated using **correlated Geometric Brownian Motion (GBM)**:

\[
dS_t = S_t \left((r - q)dt + \sigma dW_t\right)
\]

Pricing is computed as the **expected discounted payoff** under the risk-neutral measure:

\[
Price = \mathbb{E}^{\mathbb{Q}} \left[\sum DF(t_k) \cdot CF(t_k)\right]
\]

### Risk Metrics

The application provides:

- Estimated product price
- Autocall probability
- Payoff distribution
- Tail risk statistics
- Expected autocall timing

Optional **Greeks sensitivities** can be computed using **bump-and-revalue Monte Carlo**.

---

# Visualizations

The application generates several visual outputs:

- Payoff distribution histogram
- Autocall timing distribution
- Simulated basket paths
- Market data preview

These visualizations help understand the **path-dependent risk profile** of the product.

---

# Project Structure
