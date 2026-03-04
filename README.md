# Structured Products Simulator – Phoenix Autocallable

A **Monte Carlo simulator for Phoenix structured products**, built with **Python and Streamlit**.

The application allows users to simulate multi-asset structured products (e.g. **Phoenix Autocallables**) and analyze their **payoff distribution, pricing, and sensitivities**.

This project is designed as an **educational and analytical framework** to understand the mechanics and risk profile of structured products used in **Equity Derivatives desks**.

---

## Overview

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

## Key Features

### Product Mechanics
The simulator models the lifecycle of a Phoenix structured product:
- Coupon payments conditional on barrier levels
- Memory coupon accumulation
- Early redemption through autocall triggers
- Capital protection via maturity barriers

### Multi-Asset Basket Support
Supported basket constructions:
- **Worst-of**
- **Best-of**
- **Arithmetic average**
- **Weighted basket**

### Market Data Integration
The application can retrieve:
- Spot prices
- Historical volatility
- Correlation matrices  
using **Yahoo Finance (yfinance)**.

Users can also manually specify parameters.

### Monte Carlo Pricing
Asset paths are simulated using **correlated Geometric Brownian Motion (GBM)**.

Risk-neutral dynamics (plain text):

- dS/S = (r - q) dt + sigma dW

Where:
- `r` = risk-free rate
- `q` = dividend yield
- `sigma` = volatility
- `W` = Brownian motion (with correlations across assets)

**Pricing idea (plain text):**

- Price ≈ average over simulations of: sum over cashflow dates of [ DiscountFactor(t) * Cashflow(t) ]

With a flat rate:
- DiscountFactor(t) = exp(-r * t)

### Risk Metrics
The application provides:
- Estimated product price (PV)
- Autocall probability
- Payoff distribution
- Tail risk statistics (quantiles)
- Expected autocall timing

Optional **Greeks sensitivities** can be computed using **bump-and-revalue Monte Carlo**.

---

## Visualizations
The application generates:
- Payoff distribution histogram
- Autocall timing distribution
- Simulated basket paths
- Market data preview

---

## Project Structure
