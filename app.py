"""
Autocall Phoenix Pricer
A Streamlit application for pricing Phoenix autocallable structured products
"""

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import yfinance as yf
from typing import Tuple, Optional

# ----------------------------
# PAGE CONFIG
# ----------------------------
st.set_page_config(
    page_title="Autocall Phoenix Pricer",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Minimal CSS
st.markdown("""
<style>
    .big-price {
        font-size: 56px;
        font-weight: bold;
        text-align: center;
        padding: 30px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 8px;
        margin: 30px 0 10px 0;
    }
    .price-subtitle {
        font-size: 14px;
        text-align: center;
        color: #555;
        margin-bottom: 30px;
        font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)

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
    show_spinner="Simulating paths..."
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
# PRICER PAGE
# ==========================================

def show_pricer():
    st.title("Autocall Phoenix Pricer")
    st.markdown("**Initial Investment: $100 | Fair Value Pricing with Risk-Neutral Monte Carlo**")
    
    with st.sidebar:
        st.header("Mode")
        mode = st.radio("Parameter source", ["Manual parameters", "Market data (Yahoo Finance)"], index=0)
        
        st.header("Product Structure")
        T = st.number_input("Maturity (years)", min_value=0.25, value=3.0, step=0.25)
        steps_per_year = st.selectbox("Simulation steps/year", [252, 52, 12], index=0)
        obs_per_year = st.selectbox("Observation frequency/year", [12, 4, 2, 1], index=1)
        
        coupon_pa = st.number_input("Coupon p.a.", min_value=0.0, value=0.10, step=0.01, format="%.2f")
        coupon_trigger = st.number_input("Coupon trigger", min_value=0.0, max_value=2.0, value=0.70, step=0.01)
        call_trigger = st.number_input("Autocall trigger", min_value=0.0, max_value=2.0, value=1.00, step=0.01)
        barrier = st.number_input("Maturity barrier", min_value=0.0, max_value=2.0, value=0.60, step=0.01)
        memory = st.checkbox("Memory coupon", value=True)
        
        st.header("Basket")
        basket_kind = st.selectbox("Basket type", ["worst-of", "best-of", "average", "weighted"], index=0)
        
        st.header("Rates")
        r = st.number_input("Risk-free rate", value=0.02, step=0.005, format="%.3f")
        use_q = st.checkbox("Set dividend yields", value=False)
        
        st.header("Monte Carlo")
        n_sims = st.slider("Simulations", 1000, 10000, 10000, step=1000)
        seed = st.number_input("Random seed", value=42, step=1)
        
        nominal = 100.0
        
        n_steps_est = int(round(T * steps_per_year))
        mem_mb = (n_sims * n_steps_est * 2 * 8) / (1024**2)
        if mem_mb > 300:
            st.warning(f"Estimated memory: {mem_mb:.0f} MB")
    
    tickers = None
    px_preview = None
    
    if mode == "Market data (Yahoo Finance)":
        with st.sidebar:
            st.subheader("Market Settings")
            tickers_str = st.text_input("Tickers (comma-separated)", value="AAPL,MSFT")
            lookback_years = st.slider("Lookback (years)", 1, 10, 3)
        
        requested_tickers = [t.strip().upper() for t in tickers_str.split(",") if t.strip()]
        
        if not requested_tickers:
            st.error("Please enter at least 1 ticker")
            st.stop()
        
        try:
            with st.spinner("Fetching market data..."):
                S0, sigma, corr, px_preview, tickers = fetch_market_params(requested_tickers, lookback_years)
            
            if len(tickers) < len(requested_tickers):
                failed = set(requested_tickers) - set(tickers)
                st.warning(f"Could not download: {', '.join(failed)}")
            
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
            st.error(f"Error: {str(e)}")
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
        with st.spinner("Running Monte Carlo simulation..."):
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
        
        # ===== PRICING LOGIC =====
        discount_factor = np.exp(-r * T)
        expected_payoff = np.mean(payoff)
        estimated_price = expected_payoff * discount_factor
        
        median_payoff = np.median(payoff)
        median_price = median_payoff * discount_factor
        autocall_prob = np.mean(autocalled) * 100
        
        # ===== MAIN DISPLAY =====
        st.markdown(f"""
        <div class="big-price">
            ${estimated_price:.2f}
        </div>
        <div class="price-subtitle">
            Fair Value Today | Discounted at {r*100:.1f}% over {T:.1f} years | Autocall Probability: {autocall_prob:.1f}%
        </div>
        """, unsafe_allow_html=True)
        
        st.divider()
        
        # ===== INPUT SUMMARY - REDESIGNED =====
        st.subheader("Input Summary", divider="blue")
        
        labels = tickers if tickers else [f"A{i+1}" for i in range(len(S0))]
        
        # ===== TOP ROW: UNDERLYINGS & PRODUCT PARAMS =====
        col_under, col_product = st.columns([1.2, 1])
        
        with col_under:
            st.markdown("**Underlyings**")
            df_params = pd.DataFrame({
                "Price": [f"${x:.2f}" for x in S0],
                "Volatility": [f"{x:.1%}" for x in sigma]
            }, index=labels)
            st.dataframe(df_params, use_container_width=True, height=150)
        
        with col_product:
            st.markdown("**Product Structure**")
            product_data = {
                "Maturity": f"{T:.2f} years",
                "Coupon p.a.": f"{coupon_pa:.1%}",
                "Coupon Trigger": f"{coupon_trigger:.0%}",
                "Autocall Trigger": f"{call_trigger:.0%}",
                "Maturity Barrier": f"{barrier:.0%}",
                "Basket Type": basket_kind.capitalize(),
            }
            
            for key, val in product_data.items():
                st.markdown(f"<div style='padding: 8px 0;'><b>{key}:</b> <code>{val}</code></div>", unsafe_allow_html=True)
            
            st.markdown(f"<div style='padding: 8px 0; border-top: 1px solid #ddd; margin-top: 10px;'><b>Risk-free Rate:</b> <code>{r:.2%}</code></div>", unsafe_allow_html=True)
        
        st.divider()
        
        # ===== BOTTOM ROW: CORRELATION MATRIX =====
        st.markdown("**Correlation Matrix**")
        df_corr = pd.DataFrame(corr, index=labels, columns=labels)
        
        # Format correlation as heatmap-style with colors
        def color_corr(val):
            if val < 0:
                color = f'rgba(255, 0, 0, {abs(val) * 0.3})'
            elif val > 0:
                color = f'rgba(0, 102, 204, {val * 0.3})'
            else:
                color = 'rgba(0,0,0,0)'
            return f'background-color: {color}'
        
        styled_corr = df_corr.round(3).style.applymap(color_corr, subset=pd.IndexSlice[:, :])
        st.dataframe(styled_corr, use_container_width=True, height=180)
        
        st.divider()
        
        # ===== PAYOFF HISTOGRAM =====
        st.subheader("Payoff Analysis", divider="blue")
        
        fig1, ax1 = plt.subplots(figsize=(14, 6), dpi=100)
        ax1.hist(payoff, bins=60, edgecolor='black', alpha=0.65, color='steelblue')
        ax1.axvline(nominal, color='red', linestyle='--', linewidth=2.5, alpha=0.8, label='Nominal ($100)')
        ax1.axvline(expected_payoff, color='green', linestyle='--', linewidth=2.5, alpha=0.8, label='Expected Payoff')
        ax1.axvline(estimated_price, color='orange', linestyle='--', linewidth=3, alpha=0.9, label='Fair Value Today')
        
        ax1.set_xlabel('Value ($)', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Frequency', fontsize=12, fontweight='bold')
        ax1.set_title('Payoff Distribution at Maturity', fontsize=13, fontweight='bold', pad=15)
        ax1.legend(loc='upper right', fontsize=11)
        ax1.grid(True, alpha=0.2)
        
        st.pyplot(fig1, use_container_width=True)
        plt.close(fig1)
        
        st.divider()
        
        # ===== AUTOCALL TIMING =====
        st.subheader("Autocall Timing", divider="blue")
        calls = autocall_obs[autocall_obs >= 0]
        
        fig2, ax2 = plt.subplots(figsize=(14, 5.5), dpi=100)
        
        if calls.size > 0:
            ax2.hist(calls, bins=np.arange(-0.5, len(obs_idx) + 0.5, 1), edgecolor='black', alpha=0.7, color='coral')
            ax2.set_xlabel('Observation Number', fontsize=12, fontweight='bold')
            ax2.set_ylabel('Count', fontsize=12, fontweight='bold')
            ax2.set_title(f'Autocall Events ({autocall_prob:.1f}% probability)', fontsize=13, fontweight='bold', pad=15)
            ax2.grid(True, alpha=0.2, axis='y')
        else:
            ax2.text(0.5, 0.5, 'No Autocalls', ha='center', va='center', fontsize=18, color='gray', transform=ax2.transAxes)
            ax2.axis('off')
        
        st.pyplot(fig2, use_container_width=True)
        plt.close(fig2)
        
        st.divider()
        
        # ===== SAMPLE PATHS =====
        st.subheader("Sample Paths (30 simulations)", divider="blue")
        st.caption("Basket performance ratio over time - showing the decision boundaries")
        
        paths3 = ensure_3d_paths(paths)
        tgrid = np.linspace(0, T, n_steps + 1)
        
        fig3, ax3 = plt.subplots(figsize=(14, 6), dpi=100)
        m = min(30, paths3.shape[0])
        
        for i in range(m):
            ratio_line = basket_ratio(paths3[i, :, :], S0, kind=basket_kind, weights=weights)
            ax3.plot(tgrid, ratio_line, linewidth=0.7, alpha=0.35, color='steelblue')
        
        ax3.axhline(coupon_trigger, linestyle='--', color='green', linewidth=2.5, alpha=0.85, label=f'Coupon Trigger ({coupon_trigger:.0%})')
        ax3.axhline(call_trigger, linestyle='--', color='blue', linewidth=2.5, alpha=0.85, label=f'Autocall Trigger ({call_trigger:.0%})')
        ax3.axhline(barrier, linestyle='--', color='red', linewidth=2.5, alpha=0.85, label=f'Maturity Barrier ({barrier:.0%})')
        ax3.axhline(1.0, linestyle=':', color='black', linewidth=1.5, alpha=0.5)
        
        ax3.set_xlabel('Time (years)', fontsize=12, fontweight='bold')
        ax3.set_ylabel('Basket Performance Ratio', fontsize=12, fontweight='bold')
        ax3.set_title('Risk-Neutral Monte Carlo Paths', fontsize=13, fontweight='bold', pad=15)
        ax3.legend(loc='best', fontsize=11)
        ax3.grid(True, alpha=0.2)
        ax3.set_ylim(0, 1.5)
        
        st.pyplot(fig3, use_container_width=True)
        plt.close(fig3)
        
        if px_preview is not None:
            st.divider()
            st.subheader("Market Data Preview", divider="blue")
            
            fig4, ax4 = plt.subplots(figsize=(14, 5), dpi=100)
            for col in px_preview.columns:
                ax4.plot(px_preview.index, px_preview[col], label=col, linewidth=2)
            
            ax4.set_xlabel('Date', fontsize=12, fontweight='bold')
            ax4.set_ylabel('Price ($)', fontsize=12, fontweight='bold')
            ax4.set_title('Historical Prices', fontsize=13, fontweight='bold', pad=15)
            ax4.legend(fontsize=11)
            ax4.grid(True, alpha=0.2)
            
            st.pyplot(fig4, use_container_width=True)
            plt.close(fig4)
        
        st.divider()
        st.caption("Disclaimer: Fair value pricing uses risk-neutral Monte Carlo simulation with discount rate applied. Assumes constant volatility and rates.")
    
    except Exception as e:
        st.error(f"Error: {str(e)}")
        import traceback
        st.error(traceback.format_exc())

# ==========================================
# MAIN ENTRY POINT
# ==========================================

def main():
    page = st.sidebar.radio(
        "Navigation",
        ["Pricer"],
        index=0
    )
    
    show_pricer()

if __name__ == "__main__":
    main()
