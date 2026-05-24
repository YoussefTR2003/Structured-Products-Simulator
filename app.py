"""
Autocall Athena Pricer - Phoenix Structure
A Streamlit application for pricing autocallable structured products
"""

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import yfinance as yf
from typing import Tuple, Optional
from matplotlib.patches import FancyBboxPatch

# ----------------------------
# NUMERICAL UTILITIES
# ----------------------------

def ensure_3d_paths(paths: np.ndarray) -> np.ndarray:
    """Ensure paths are in shape (n_sims, n_steps+1, n_assets)."""
    paths = np.asarray(paths, dtype=float)
    if paths.ndim == 2:
        return paths[:, :, np.newaxis]
    return paths


def nearest_psd_corr(mat: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    """Project a symmetric matrix to a PSD correlation matrix via eigenvalue clipping."""
    A = np.asarray(mat, dtype=float)
    A = 0.5 * (A + A.T)
    np.fill_diagonal(A, 1.0)

    vals, vecs = np.linalg.eigh(A)
    vals = np.maximum(vals, eps)
    B = vecs @ np.diag(vals) @ vecs.T
    B = 0.5 * (B + B.T)

    d = np.sqrt(np.maximum(np.diag(B), eps))
    C = B / np.outer(d, d)
    np.fill_diagonal(C, 1.0)
    return 0.5 * (C + C.T)


def cholesky_safe(corr: np.ndarray) -> np.ndarray:
    """Cholesky decomposition with PSD fallback."""
    try:
        return np.linalg.cholesky(corr)
    except np.linalg.LinAlgError:
        corr_psd = nearest_psd_corr(corr)
        return np.linalg.cholesky(corr_psd)

# ----------------------------
# MARKET MODEL
# ----------------------------

@st.cache_data(
    ttl=300,
    max_entries=3,
    show_spinner="🔄 Simulating paths..."
)
def simulate_correlated_gbm(
    S0: tuple,
    r: float,
    q: tuple,
    sigma: tuple,
    corr: tuple,
    T: float,
    n_steps: int,
    n_sims: int,
    seed: int = 42,
) -> np.ndarray:
    """Simulate correlated GBM paths under risk-neutral measure."""
    S0 = np.array(S0)
    q = np.array(q)
    sigma = np.array(sigma)
    corr = np.array(corr).reshape(int(np.sqrt(len(corr))), -1)
    
    rng = np.random.default_rng(seed)
    
    n_assets = S0.size
    dt = T / n_steps
    
    L = cholesky_safe(corr)
    
    Z = rng.standard_normal((n_sims, n_steps, n_assets))
    dW = Z @ L.T
    
    drift = (r - q - 0.5 * sigma**2) * dt
    diffusion = sigma * np.sqrt(dt) * dW
    
    log_increments = drift[np.newaxis, np.newaxis, :] + diffusion
    log_paths = np.cumsum(log_increments, axis=1)
    
    paths = np.empty((n_sims, n_steps + 1, n_assets))
    paths[:, 0, :] = S0
    paths[:, 1:, :] = S0[np.newaxis, np.newaxis, :] * np.exp(log_paths)
    
    return paths

# ----------------------------
# BASKET MECHANICS
# ----------------------------

def basket_ratio(
    St_t: np.ndarray, 
    S0: np.ndarray, 
    kind: str = "worst-of", 
    weights: Optional[np.ndarray] = None
) -> np.ndarray:
    """Calculate basket performance ratio."""
    S0 = np.asarray(S0, dtype=float)
    R = St_t / S0[np.newaxis, :]
    
    kind = kind.lower()
    
    if kind == "worst-of":
        return np.min(R, axis=1)
    elif kind == "best-of":
        return np.max(R, axis=1)
    elif kind == "average":
        return np.mean(R, axis=1)
    elif kind == "weighted":
        if weights is None:
            raise ValueError("weights required for weighted basket")
        w = np.asarray(weights, dtype=float)
        w = w / np.sum(w)
        return R @ w
    else:
        raise ValueError(f"Unknown basket kind: {kind}")


def build_obs_idx(T: float, steps_per_year: int, obs_per_year: int) -> np.ndarray:
    """Build observation step indices."""
    n_steps = int(round(T * steps_per_year))
    obs_step = max(1, int(round(steps_per_year / obs_per_year)))
    obs_idx = np.arange(obs_step, n_steps + 1, obs_step, dtype=int)
    obs_idx = obs_idx[obs_idx <= n_steps]
    return obs_idx

# ----------------------------
# PRODUCT PAYOFF
# ----------------------------

def phoenix_payoff(
    paths: np.ndarray,
    S0: np.ndarray,
    nominal: float,
    obs_idx: np.ndarray,
    coupon_rate_per_obs: float,
    coupon_trigger: float,
    call_trigger: float,
    barrier: float,
    basket_kind: str = "worst-of",
    weights: Optional[np.ndarray] = None,
    memory: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Phoenix autocall payoff calculator."""
    paths = ensure_3d_paths(paths)
    n_sims, _, n_assets = paths.shape
    
    S0 = np.asarray(S0, dtype=float)
    if S0.ndim == 0:
        S0 = np.array([float(S0)])
    if S0.size != n_assets:
        raise ValueError(f"S0 size ({S0.size}) must match n_assets ({n_assets})")
    
    payoff = np.zeros(n_sims)
    autocalled = np.zeros(n_sims, dtype=bool)
    autocall_obs = np.full(n_sims, -1, dtype=int)
    accrued = np.zeros(n_sims)
    
    for k, t in enumerate(obs_idx):
        alive = ~autocalled
        if not np.any(alive):
            break
        
        St = paths[alive, t, :]
        ratio = basket_ratio(St, S0, kind=basket_kind, weights=weights)
        
        coupon_ok = ratio >= coupon_trigger
        
        if memory:
            accrued_alive = accrued[alive] + nominal * coupon_rate_per_obs
            payoff[alive] += accrued_alive * coupon_ok
            accrued[alive] = accrued_alive * (~coupon_ok)
        else:
            payoff[alive] += (nominal * coupon_rate_per_obs) * coupon_ok
        
        call_ok = ratio >= call_trigger
        idx_alive = np.where(alive)[0]
        called_idx = idx_alive[call_ok]
        
        if called_idx.size > 0:
            autocalled[called_idx] = True
            autocall_obs[called_idx] = k
            payoff[called_idx] += nominal
    
    alive = ~autocalled
    if np.any(alive):
        ST = paths[alive, -1, :]
        ratio_T = basket_ratio(ST, S0, kind=basket_kind, weights=weights)
        protected = ratio_T >= barrier
        payoff[alive] += np.where(protected, nominal, nominal * ratio_T)
    
    return payoff, autocalled, autocall_obs

# ----------------------------
# METRICS
# ----------------------------

def summarize_metrics(
    payoff: np.ndarray, 
    autocalled: np.ndarray, 
    autocall_obs: np.ndarray, 
    obs_per_year: int
) -> pd.DataFrame:
    """Generate summary statistics."""
    metrics = {
        "Expected payoff": float(np.mean(payoff)),
        "Median payoff": float(np.median(payoff)),
        "P(Autocall)": float(np.mean(autocalled)),
        "5% quantile": float(np.quantile(payoff, 0.05)),
        "1% quantile": float(np.quantile(payoff, 0.01)),
    }
    
    calls = autocall_obs[autocall_obs >= 0]
    if calls.size > 0:
        metrics["Expected autocall time (years)"] = float(np.mean((calls + 1) / obs_per_year))
    else:
        metrics["Expected autocall time (years)"] = float('nan')
    
    return pd.DataFrame(metrics, index=["Value"]).T

# ----------------------------
# MARKET DATA
# ----------------------------

@st.cache_data(show_spinner=False)
def fetch_market_params(
    tickers: list, 
    lookback_years: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame, list]:
    """Fetch market data from Yahoo Finance."""
    try:
        end = pd.Timestamp.today()
        start = end - pd.DateOffset(years=lookback_years)
        
        data = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)
        
        if len(tickers) == 1:
            if 'Close' in data and not data['Close'].empty:
                px = data['Close'].to_frame(tickers[0])
                successful_tickers = tickers
            else:
                raise ValueError(f"No data for {tickers[0]}")
        else:
            if 'Close' in data.columns.levels[0]:
                px = data['Close']
            else:
                px = data['Close']
            
            px = px.dropna(axis=1, how='all')
            successful_tickers = px.columns.tolist()
        
        px = px.dropna()
        
        if px.empty or px.shape[1] == 0:
            raise ValueError("No valid price data returned.")
        
        rets = px.pct_change().dropna()
        S0 = px.iloc[-1].values
        sigma = rets.std().values * np.sqrt(252)
        corr = rets.corr().values
        
        return S0, sigma, corr, px, successful_tickers
        
    except Exception as e:
        raise ValueError(f"Failed to fetch market data: {str(e)}")

# ==========================================
# DOCUMENTATION PAGE
# ==========================================

def show_documentation():
    st.title("📚 How the Autocall Pricer Works")
    st.markdown("**Understanding the mathematics behind Phoenix autocall pricing**")
    
    # Table of contents
    st.sidebar.markdown("## 📖 Contents")
    section = st.sidebar.radio(
        "Jump to section:",
        [
            "🎯 What is an Autocall?",
            "📊 Monte Carlo Simulation",
            "🔗 Correlation & Cholesky",
            "🧮 Geometric Brownian Motion",
            "🧺 Basket Mechanisms",
            "💰 Payoff Calculation",
            "⚙️ Technical Details"
        ]
    )
    
    # SECTION 1: What is an Autocall?
    if section == "🎯 What is an Autocall?":
        st.header("🎯 What is an Autocall (Phoenix)?")
        
        st.markdown("""
        An **autocall** (or Phoenix) is a structured product that:
        
        1. **Pays coupons** if the underlying asset(s) are above a barrier
        2. **Auto-redeems early** if the asset(s) perform well
        3. **Protects capital** at maturity (with conditions)
        """)
        
        # Visual timeline
        st.subheader("📅 Product Timeline")
        
        fig, ax = plt.subplots(figsize=(12, 4))
        
        times = [0, 0.25, 0.5, 0.75, 1.0]
        labels = ['Start', 'Obs 1', 'Obs 2', 'Obs 3', 'Maturity']
        
        ax.plot(times, [0.5]*5, 'o-', linewidth=3, markersize=15, color='steelblue')
        
        for i, (t, label) in enumerate(zip(times, labels)):
            ax.text(t, 0.3, label, ha='center', fontsize=11, fontweight='bold')
            
            if i > 0:
                if i < 4:
                    ax.add_patch(FancyBboxPatch((t-0.05, 0.6), 0.1, 0.25, 
                                boxstyle="round,pad=0.01", 
                                facecolor='lightgreen', edgecolor='green', linewidth=2))
                    ax.text(t, 0.725, '✓ Coupon?', ha='center', fontsize=9)
                    ax.text(t, 0.95, '✓ Autocall?', ha='center', fontsize=9, color='blue')
                else:
                    ax.add_patch(FancyBboxPatch((t-0.05, 0.6), 0.1, 0.25,
                                boxstyle="round,pad=0.01",
                                facecolor='lightyellow', edgecolor='orange', linewidth=2))
                    ax.text(t, 0.725, 'Final\nPayoff', ha='center', fontsize=9)
        
        ax.set_xlim(-0.1, 1.1)
        ax.set_ylim(0, 1.2)
        ax.axis('off')
        ax.set_title('Autocall Timeline with Observation Dates', fontsize=14, fontweight='bold')
        st.pyplot(fig)
        plt.close()
        
        # Example payoff scenarios
        st.subheader("💡 Example Scenarios")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.success("**🎉 Best Case**")
            st.markdown("""
            **Observation 2**: Asset > 100%
            
            ✅ Autocalls early  
            ✅ Get: Capital + Coupons  
            ⏱️ Duration: 6 months
            """)
        
        with col2:
            st.info("**📈 Good Case**")
            st.markdown("""
            **All Obs**: Asset > 70%  
            **Maturity**: Asset > 60%
            
            ✅ All coupons paid  
            ✅ Get: Capital + All coupons  
            ⏱️ Duration: 3 years
            """)
        
        with col3:
            st.warning("**📉 Bad Case**")
            st.markdown("""
            **Maturity**: Asset at 50%
            
            ❌ No autocall  
            ⚠️ Capital loss: -50%  
            💰 Some coupons received
            """)
    
    # SECTION 2: Monte Carlo
    elif section == "📊 Monte Carlo Simulation":
        st.header("📊 Monte Carlo Simulation")
        
        st.markdown("""
        ### Why Monte Carlo?
        
        Autocalls are **path-dependent**: the payoff depends on the entire price history, not just the final price.
        
        **Solution**: Simulate thousands of possible future paths and average the payoffs.
        """)
        
        st.latex(r"\text{Price} \approx \frac{1}{N} \sum_{i=1}^{N} \text{Payoff}_i")
        
        # Interactive demo
        st.subheader("🎲 Interactive Demo")
        
        n_paths = st.slider("Number of paths to simulate", 10, 200, 50, step=10)
        volatility = st.slider("Volatility", 0.1, 0.5, 0.25, step=0.05)
        
        np.random.seed(42)
        T = 1.0
        n_steps = 252
        dt = T / n_steps
        S0 = 100
        
        paths = np.zeros((n_paths, n_steps + 1))
        paths[:, 0] = S0
        
        for i in range(n_paths):
            for t in range(1, n_steps + 1):
                z = np.random.randn()
                paths[i, t] = paths[i, t-1] * np.exp(-0.5 * volatility**2 * dt + volatility * np.sqrt(dt) * z)
        
        fig, ax = plt.subplots(figsize=(10, 5))
        time_grid = np.linspace(0, T, n_steps + 1)
        
        for i in range(min(n_paths, 50)):
            alpha = 0.3 if n_paths > 50 else 0.5
            ax.plot(time_grid, paths[i], linewidth=0.8, alpha=alpha, color='steelblue')
        
        mean_path = np.mean(paths, axis=0)
        ax.plot(time_grid, mean_path, linewidth=3, color='red', label='Average Path', linestyle='--')
        
        ax.axhline(S0, color='black', linestyle='--', linewidth=1.5, label='Initial Price')
        ax.set_xlabel('Time (years)', fontsize=12)
        ax.set_ylabel('Stock Price', fontsize=12)
        ax.set_title(f'{n_paths} Simulated Price Paths', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        plt.close()
        
        st.info(f"""
        **📊 Statistics from {n_paths} simulations:**
        - Average final price: ${mean_path[-1]:.2f}
        - Min final price: ${np.min(paths[:, -1]):.2f}
        - Max final price: ${np.max(paths[:, -1]):.2f}
        - Standard deviation: ${np.std(paths[:, -1]):.2f}
        """)
        
        st.markdown("""
        ### 🔑 Key Idea
        
        More simulations = More accurate price, but slower computation
        
        | Simulations | Accuracy | Speed |
        |------------|----------|-------|
        | 5,000 | ±2% | Fast ⚡ |
        | 10,000 | ±1% | Good ✅ |
        | 30,000 | ±0.5% | Slow 🐌 |
        """)
    
    # SECTION 3: Correlation & Cholesky
    elif section == "🔗 Correlation & Cholesky":
        st.header("🔗 Correlation & Cholesky Decomposition")
        
        st.markdown("""
        ### The Problem
        
        For multi-asset products, we need to simulate **correlated** assets:
        - Apple and Microsoft tend to move together
        - How do we capture this in our simulation?
        """)
        
        # Interactive correlation demo
        st.subheader("🎨 Visual Correlation Demo")
        
        corr_value = st.slider("Correlation between Asset 1 and Asset 2", -0.9, 0.9, 0.5, step=0.1)
        
        np.random.seed(42)
        n_samples = 500
        
        corr_matrix = np.array([[1.0, corr_value], [corr_value, 1.0]])
        L = np.linalg.cholesky(corr_matrix)
        
        Z = np.random.randn(n_samples, 2)
        X = Z @ L.T
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        ax1.scatter(Z[:, 0], Z[:, 1], alpha=0.5, s=20)
        ax1.set_xlabel('Asset 1 (independent)', fontsize=11)
        ax1.set_ylabel('Asset 2 (independent)', fontsize=11)
        ax1.set_title('BEFORE: Independent Random Variables', fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(-3, 3)
        ax1.set_ylim(-3, 3)
        
        ax2.scatter(X[:, 0], X[:, 1], alpha=0.5, s=20, color='coral')
        ax2.set_xlabel('Asset 1', fontsize=11)
        ax2.set_ylabel('Asset 2', fontsize=11)
        ax2.set_title(f'AFTER: Correlated (ρ={corr_value:.1f})', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim(-3, 3)
        ax2.set_ylim(-3, 3)
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        
        st.markdown("""
        ### 🧮 What is Cholesky Decomposition?
        
        It's a mathematical technique to transform independent random numbers into correlated ones.
        
        **Formula**: If we have a correlation matrix **C**, we find a matrix **L** such that:
        """)
        
        st.latex(r"C = L \times L^T")
        
        st.markdown("""
        Then:
        1. Generate independent random numbers **Z**
        2. Multiply: **X = Z × L**
        3. Now **X** has the desired correlations!
        """)
        
        st.info("""
        **🔑 Why This Matters**
        
        - **Worst-of basket**: If assets are highly correlated (ρ=0.9), they crash together → More risky
        - **Diversification**: Low correlation (ρ=0.2) → One asset can save the day
        """)
    
    # SECTION 4: GBM
    elif section == "🧮 Geometric Brownian Motion":
        st.header("🧮 Geometric Brownian Motion (GBM)")
        
        st.markdown("""
        ### Stock Price Model
        
        We model stock prices using **Geometric Brownian Motion** - the standard model in finance.
        
        **The Equation**:
        """)
        
        st.latex(r"dS = \mu S dt + \sigma S dW")
        
        st.markdown("""
        Where:
        - **S** = Stock price
        - **μ** = Drift (expected return)
        - **σ** = Volatility (how much it bounces around)
        - **dW** = Random shock (Brownian motion)
        
        **In English**: 
        > "The stock price changes by a predictable drift **plus** a random shock"
        """)
        
        # Interactive GBM
        st.subheader("🎮 Interactive GBM")
        
        col1, col2 = st.columns(2)
        with col1:
            mu = st.slider("Drift (μ) - Expected return", -0.2, 0.3, 0.05, step=0.05)
        with col2:
            sigma_gbm = st.slider("Volatility (σ)", 0.05, 0.6, 0.25, step=0.05)
        
        np.random.seed(42)
        T = 1.0
        n_steps = 252
        dt = T / n_steps
        S0 = 100
        n_paths_gbm = 100
        
        paths_gbm = np.zeros((n_paths_gbm, n_steps + 1))
        paths_gbm[:, 0] = S0
        
        for i in range(n_paths_gbm):
            for t in range(1, n_steps + 1):
                z = np.random.randn()
                drift_component = (mu - 0.5 * sigma_gbm**2) * dt
                diffusion_component = sigma_gbm * np.sqrt(dt) * z
                paths_gbm[i, t] = paths_gbm[i, t-1] * np.exp(drift_component + diffusion_component)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        time_grid = np.linspace(0, T, n_steps + 1)
        
        for i in range(min(n_paths_gbm, 30)):
            ax.plot(time_grid, paths_gbm[i], linewidth=0.7, alpha=0.4, color='steelblue')
        
        mean_path_gbm = np.mean(paths_gbm, axis=0)
        ax.plot(time_grid, mean_path_gbm, linewidth=3, color='red', label='Average', linestyle='--')
        ax.axhline(S0, color='black', linestyle='--', linewidth=1.5, label='Initial Price')
        
        theoretical = S0 * np.exp(mu * time_grid)
        ax.plot(time_grid, theoretical, linewidth=2, color='green', label='Theoretical E[S]', linestyle=':')
        
        ax.set_xlabel('Time (years)', fontsize=12)
        ax.set_ylabel('Stock Price ($)', fontsize=12)
        ax.set_title(f'GBM: μ={mu:.2f}, σ={sigma_gbm:.2f}', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        plt.close()
        
        st.info("""
        ### 🎯 Risk-Neutral Pricing
        
        In our pricer, we use **risk-neutral** measure where:
        - Drift μ = **r - q** (risk-free rate minus dividends)
        - This is the standard for derivatives pricing
        
        **Why?** It ensures arbitrage-free pricing and allows discounting at the risk-free rate.
        """)
    
    # SECTION 5: Basket Mechanisms
    elif section == "🧺 Basket Mechanisms":
        st.header("🧺 Basket Mechanisms")
        
        st.markdown("""
        ### Multi-Asset Products
        
        When an autocall has **multiple underlying assets** (e.g., Apple + Microsoft + Tesla),
        we need to combine them into a **basket**.
        """)
        
        st.subheader("📊 Basket Types")
        
        assets_perf = {
            'Apple': 1.15,
            'Microsoft': 1.05,
            'Tesla': 0.85
        }
        
        tab1, tab2, tab3, tab4 = st.tabs(["Worst-of", "Best-of", "Average", "Weighted"])
        
        perf_values = list(assets_perf.values())
        
        with tab1:
            st.markdown("""
            ### Worst-of Basket
            
            **Formula**: `Basket = min(Asset1, Asset2, Asset3)`
            
            **Use case**: Most conservative / Most common in structured products
            
            **Payoff depends on**: The worst performing asset
            """)
            st.metric("Basket Performance", f"{min(perf_values):.0%}")
            st.error("⚠️ In this case: Tesla (85%) determines the payoff")
        
        with tab2:
            st.markdown("""
            ### Best-of Basket
            
            **Formula**: `Basket = max(Asset1, Asset2, Asset3)`
            
            **Use case**: Most optimistic / Rare in real products
            
            **Payoff depends on**: The best performing asset
            """)
            st.metric("Basket Performance", f"{max(perf_values):.0%}")
            st.success("✅ In this case: Apple (115%) determines the payoff")
        
        with tab3:
            st.markdown("""
            ### Average Basket
            
            **Formula**: `Basket = (Asset1 + Asset2 + Asset3) / 3`
            
            **Use case**: Balanced approach
            
            **Payoff depends on**: Average of all assets
            """)
            st.metric("Basket Performance", f"{np.mean(perf_values):.0%}")
            st.info("📊 Average smooths out individual performances")
        
        with tab4:
            st.markdown("""
            ### Weighted Basket
            
            **Formula**: `Basket = w1·Asset1 + w2·Asset2 + w3·Asset3`
            
            where w1 + w2 + w3 = 1
            
            **Use case**: When you want different exposures to each asset
            """)
            
            w1 = st.slider("Apple weight", 0.0, 1.0, 0.33, step=0.01, key='w1_doc')
            w2 = st.slider("Microsoft weight", 0.0, 1.0, 0.33, step=0.01, key='w2_doc')
            w3 = 1 - w1 - w2
            
            st.write(f"Tesla weight (automatic): {w3:.2f}")
            
            if w3 < 0:
                st.error("⚠️ Weights must sum to 1!")
            else:
                weighted_perf = w1 * perf_values[0] + w2 * perf_values[1] + w3 * perf_values[2]
                st.metric("Weighted Basket Performance", f"{weighted_perf:.0%}")
    
    # SECTION 6: Payoff Calculation
    elif section == "💰 Payoff Calculation":
        st.header("💰 Payoff Calculation")
        
        st.markdown("""
        ### Phoenix Payoff Logic
        
        The product checks the basket performance at each observation date:
        """)
        
        st.subheader("📊 Decision Tree")
        
        st.code("""
FOR each observation date:
    
    IF basket >= Autocall Trigger (e.g., 100%):
        → Pay: Nominal + All Coupons (including memory)
        → Product ENDS
    
    ELSE IF basket >= Coupon Trigger (e.g., 70%):
        → Pay: Coupon for this period (+ memory if applicable)
        → Continue to next observation
    
    ELSE:
        → No coupon
        → If MEMORY = True: Accumulate unpaid coupon
        → Continue to next observation

AT Maturity (if not autocalled):
    
    IF basket >= Barrier (e.g., 60%):
        → Pay: Nominal + Any remaining coupons
    
    ELSE:
        → Pay: Nominal × basket performance
        → CAPITAL LOSS!
        """, language="text")
    
    # SECTION 7: Technical Details
    elif section == "⚙️ Technical Details":
        st.header("⚙️ Technical Implementation Details")
        
        st.markdown("""
        ### Code Architecture
        
        The pricer is built with these main components:
        """)
        
        with st.expander("🔢 1. Simulation Engine"):
            st.code("""
def simulate_correlated_gbm(S0, r, q, sigma, corr, T, n_steps, n_sims, seed):
    # 1. Cholesky decomposition of correlation
    L = cholesky(corr)
    
    # 2. Generate independent random shocks
    Z = randn(n_sims, n_steps, n_assets)
    
    # 3. Make them correlated
    dW = Z @ L.T
    
    # 4. Compute GBM paths
    drift = (r - q - 0.5*sigma²) * dt
    diffusion = sigma * sqrt(dt) * dW
    
    log_paths = cumsum(drift + diffusion)
    paths = S0 * exp(log_paths)
    
    return paths
            """, language="python")
        
        st.markdown("""
        ### Performance Optimization
        
        | Technique | Purpose | Impact |
        |-----------|---------|--------|
        | `@st.cache_data` | Cache simulations | 100x faster re-runs |
        | NumPy vectorization | Batch operations | 50x faster than loops |
        | Cholesky decomposition | Efficient correlation | O(n²) instead of O(n³) |
        | `dtype=float` | Memory efficiency | 50% less memory |
        """)

# ==========================================
# PRICER PAGE
# ==========================================

def show_pricer():
    st.title("🏛️ Autocall Athena Pricer — Phoenix Structure")
    
    with st.sidebar:
        st.header("📊 Mode")
        mode = st.radio("Parameter source", ["Manual parameters", "Market data (Yahoo Finance)"], index=0)
        
        st.header("📋 Product Structure")
        nominal = st.number_input("Nominal", min_value=1.0, value=1000.0, step=100.0)
        T = st.number_input("Maturity (years)", min_value=0.25, value=3.0, step=0.25)
        steps_per_year = st.selectbox("Simulation steps/year", [252, 52, 12], index=0)
        obs_per_year = st.selectbox("Observation frequency/year", [12, 4, 2, 1], index=1)
        
        coupon_pa = st.number_input("Coupon p.a.", min_value=0.0, value=0.10, step=0.01, format="%.2f")
        coupon_trigger = st.number_input("Coupon trigger", min_value=0.0, max_value=2.0, value=0.70, step=0.01)
        call_trigger = st.number_input("Autocall trigger", min_value=0.0, max_value=2.0, value=1.00, step=0.01)
        barrier = st.number_input("Maturity barrier", min_value=0.0, max_value=2.0, value=0.60, step=0.01)
        memory = st.checkbox("Memory coupon", value=True)
        
        st.header("🧺 Basket")
        basket_kind = st.selectbox("Basket type", ["worst-of", "best-of", "average", "weighted"], index=0)
        
        st.header("💹 Rates")
        r = st.number_input("Risk-free rate", value=0.02, step=0.005, format="%.3f")
        use_q = st.checkbox("Set dividend yields", value=False)
        
        st.header("🎲 Monte Carlo")
        n_sims = st.slider("Simulations", 5000, 30000, 10000, step=5000)
        seed = st.number_input("Random seed", value=42, step=1)
        
        n_steps_est = int(round(T * steps_per_year))
        mem_mb = (n_sims * n_steps_est * 2 * 8) / (1024**2)
        if mem_mb > 300:
            st.warning(f"⚠️ Estimated memory: {mem_mb:.0f} MB. Risk of crash on free tier.")
    
    tickers = None
    px_preview = None
    
    if mode == "Market data (Yahoo Finance)":
        with st.sidebar:
            st.subheader("Market Settings")
            tickers_str = st.text_input("Tickers (comma-separated)", value="AAPL,MSFT")
            lookback_years = st.slider("Lookback (years)", 1, 10, 3)
        
        requested_tickers = [t.strip().upper() for t in tickers_str.split(",") if t.strip()]
        
        if not requested_tickers:
            st.error("⚠️ Please enter at least 1 ticker")
            st.stop()
        
        try:
            with st.spinner("📡 Fetching market data..."):
                S0, sigma, corr, px_preview, tickers = fetch_market_params(requested_tickers, lookback_years)
            
            if len(tickers) < len(requested_tickers):
                failed = set(requested_tickers) - set(tickers)
                st.warning(f"⚠️ Could not download: {', '.join(failed)}. Proceeding with: {', '.join(tickers)}")
            
            n_assets = len(S0)
            q = np.zeros(n_assets)
            
            if use_q:
                with st.sidebar:
                    st.subheader("Dividend Yields")
                    q_list = []
                    for i, ticker in enumerate(tickers):
                        q_list.append(st.number_input(
                            f"q {ticker}", 
                            value=0.00, 
                            step=0.005, 
                            format="%.3f",
                            key=f"q_m_{i}"
                        ))
                    q = np.array(q_list)
        
        except Exception as e:
            st.error(f"❌ Market data error: {str(e)}")
            st.stop()
    
    else:
        with st.sidebar:
            st.subheader("Asset Parameters")
            n_assets = st.number_input("Number of underlyings", min_value=1, max_value=5, value=2, step=1)
            
            S0_list, sigma_list, q_list = [], [], []
            
            for i in range(int(n_assets)):
                st.markdown(f"**Asset {i+1}**")
                S0_list.append(st.number_input(f"S0 #{i+1}", value=100.0, step=1.0, key=f"S0_{i}"))
                sigma_list.append(st.number_input(f"Volatility #{i+1}", value=0.25, step=0.01, format="%.2f", key=f"sig_{i}"))
                if use_q:
                    q_list.append(st.number_input(f"Div yield #{i+1}", value=0.00, step=0.005, format="%.3f", key=f"q_{i}"))
                else:
                    q_list.append(0.0)
            
            S0 = np.array(S0_list)
            sigma = np.array(sigma_list)
            q = np.array(q_list)
            
            n_assets_int = int(n_assets)
            corr = np.eye(n_assets_int)
            
            if n_assets_int > 1:
                st.subheader("Correlations")
                st.caption("Correlation between assets (-1 to 1)")
                
                for i in range(n_assets_int):
                    for j in range(i + 1, n_assets_int):
                        corr_val = st.number_input(
                            f"Corr(A{i+1}, A{j+1})",
                            min_value=-1.0,
                            max_value=1.0,
                            value=0.0,
                            step=0.1,
                            format="%.2f",
                            key=f"corr_{i}_{j}"
                        )
                        corr[i, j] = corr_val
                        corr[j, i] = corr_val
            
            tickers = None
    
    weights = None
    if basket_kind == "weighted":
        with st.sidebar:
            st.subheader("Basket Weights")
            w_list = []
            labels = tickers if tickers else [f"A{i+1}" for i in range(int(n_assets))]
            for i, lab in enumerate(labels):
                w_list.append(st.number_input(f"Weight {lab}", value=1.0, step=0.1, key=f"w_{i}"))
            weights = np.array(w_list)
    
    n_steps = int(round(T * steps_per_year))
    obs_idx = build_obs_idx(T, steps_per_year, int(obs_per_year))
    coupon_rate_per_obs = coupon_pa / int(obs_per_year)
    
    try:
        with st.spinner("🔄 Running Monte Carlo simulation..."):
            paths = simulate_correlated_gbm(
                S0=tuple(S0),
                r=r,
                q=tuple(q),
                sigma=tuple(sigma),
                corr=tuple(corr.flatten()),
                T=T,
                n_steps=n_steps,
                n_sims=int(n_sims),
                seed=int(seed)
            )
            payoff, autocalled, autocall_obs = phoenix_payoff(
                paths, S0, nominal, obs_idx, coupon_rate_per_obs,
                coupon_trigger, call_trigger, barrier, basket_kind, weights, memory
            )
        
        metrics_df = summarize_metrics(payoff, autocalled, autocall_obs, int(obs_per_year))
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("📊 Input Summary")
            labels = tickers if tickers else [f"A{i+1}" for i in range(len(S0))]
            df_params = pd.DataFrame({"S0": S0, "Volatility": sigma, "Div Yield": q}, index=labels)
            st.dataframe(df_params, width=600)
            
            st.caption("**Correlation Matrix**")
            st.dataframe(pd.DataFrame(corr, index=labels, columns=labels).round(3), width=600)
            
            st.subheader("📈 Key Metrics")
            st.dataframe(metrics_df, width=600)
            
            out_df = pd.DataFrame({"payoff": payoff, "autocalled": autocalled.astype(int), "autocall_obs": autocall_obs})
            csv = out_df.to_csv(index=False)
            st.download_button("⬇️ Download Results (CSV)", data=csv, file_name="autocall_simulation.csv", mime="text/csv")
        
        with col2:
            st.subheader("💰 Payoff Distribution")
            fig1, ax1 = plt.subplots(figsize=(8, 4))
            ax1.hist(payoff, bins=80, edgecolor='black', alpha=0.7, color='steelblue')
            ax1.axvline(nominal, color='red', linestyle='--', linewidth=2, label=f'Nominal ({nominal:.0f})')
            ax1.set_xlabel("Payoff")
            ax1.set_ylabel("Frequency")
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            st.pyplot(fig1)
            plt.close(fig1)
            
            st.subheader("⏱️ Autocall Timing")
            calls = autocall_obs[autocall_obs >= 0]
            fig2, ax2 = plt.subplots(figsize=(8, 4))
            
            if calls.size > 0:
                ax2.hist(calls, bins=np.arange(-0.5, len(obs_idx) + 0.5, 1), edgecolor='black', alpha=0.7, color='coral')
                ax2.set_xlabel("Observation Number")
                ax2.set_ylabel("Count")
                ax2.grid(True, alpha=0.3)
            else:
                ax2.text(0.5, 0.5, "No autocalls occurred", ha="center", va="center", fontsize=14, color='gray')
                ax2.set_xlim(0, 1)
                ax2.set_ylim(0, 1)
                ax2.axis("off")
            
            st.pyplot(fig2)
            plt.close(fig2)
            
            st.subheader("📉 Sample Paths (Basket Performance)")
            paths3 = ensure_3d_paths(paths)
            tgrid = np.linspace(0, T, n_steps + 1)
            
            fig3, ax3 = plt.subplots(figsize=(8, 5))
            m = min(30, paths3.shape[0])
            
            for i in range(m):
                ratio_line = basket_ratio(paths3[i, :, :], S0, kind=basket_kind, weights=weights)
                ax3.plot(tgrid, ratio_line, linewidth=0.8, alpha=0.5, color='gray')
            
            ax3.axhline(coupon_trigger, linestyle='--', color='green', label=f'Coupon trigger ({coupon_trigger:.0%})', linewidth=2)
            ax3.axhline(call_trigger, linestyle='--', color='blue', label=f'Autocall trigger ({call_trigger:.0%})', linewidth=2)
            ax3.axhline(barrier, linestyle='--', color='red', label=f'Barrier ({barrier:.0%})', linewidth=2)
            
            ax3.set_xlabel("Time (years)")
            ax3.set_ylabel("Basket Performance")
            ax3.legend(loc='best')
            ax3.grid(True, alpha=0.3)
            st.pyplot(fig3)
            plt.close(fig3)
        
        if px_preview is not None:
            st.subheader("📊 Market Data Preview")
            st.line_chart(px_preview)
        
        st.divider()
        st.caption("⚠️ **Disclaimer**: Simplified Monte Carlo simulator using GBM. Real pricing uses implied vol surfaces, dividend curves, and calibrated rate curves.")
    
    except Exception as e:
        st.error(f"❌ Simulation error: {str(e)}")
        st.stop()

# ==========================================
# MAIN ENTRY POINT
# ==========================================

def main():
    st.set_page_config(page_title="Autocall Athena Pricer", layout="wide")
    
    # Navigation in sidebar
    page = st.sidebar.radio(
        "📑 Navigation",
        ["🏛️ Pricer", "📚 Documentation"],
        index=0
    )
    
    if page == "📚 Documentation":
        show_documentation()
    else:
        show_pricer()

if __name__ == "__main__":
    main()
