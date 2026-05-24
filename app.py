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

# Initialize session state
if 'prev_n_assets' not in st.session_state:
    st.session_state.prev_n_assets = 2
if 'prev_basket_kind' not in st.session_state:
    st.session_state.prev_basket_kind = "worst-of"

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

@st.cache_data(show_spinner="🔄 Simulating paths...")
def simulate_correlated_gbm(
    S0: tuple,  # Changed to tuple for hashability
    r: float,
    q: tuple,   # Changed to tuple for hashability
    sigma: tuple,  # Changed to tuple for hashability
    corr: tuple,   # Changed to tuple for hashability
    T: float,
    n_steps: int,
    n_sims: int,
    seed: int = 42,
) -> np.ndarray:
    """Simulate correlated GBM paths under risk-neutral measure."""
    # Convert back to arrays
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

# ----------------------------
# STREAMLIT UI
# ----------------------------

def main():
    st.set_page_config(page_title="Autocall Athena Pricer", layout="wide")
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
        n_sims = st.slider("Simulations", 5000, 120000, 30000, step=5000)
        seed = st.number_input("Random seed", value=42, step=1)
    
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
            
            # Detect if n_assets changed
            if int(n_assets) != st.session_state.prev_n_assets:
                st.session_state.prev_n_assets = int(n_assets)
                st.rerun()
            
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
    
    # Detect if basket_kind changed for weights
    if basket_kind != st.session_state.prev_basket_kind:
        st.session_state.prev_basket_kind = basket_kind
        st.rerun()
    
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
            # Convert to tuples for caching
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


if __name__ == "__main__":
    main()
