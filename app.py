# app.py
# Streamlit app: Structured Products Simulator (Phoenix / Autocall mechanics) with:
# - Manual mode (user inputs S0, vol, corr)
# - Market data mode (Yahoo Finance via yfinance -> S0, hist vol, hist corr)
# - Multi-asset (1..5), basket (worst-of / best-of / average / weighted)
# - Phoenix-style payoff: conditional coupons (+ optional memory), autocall, maturity barrier

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import yfinance as yf

# ----------------------------
# Utils: numerics
# ----------------------------
def ensure_3d_paths(paths: np.ndarray) -> np.ndarray:
    paths = np.asarray(paths, float)
    if paths.ndim == 2:  # (n_sims, n_steps+1) -> (n_sims, n_steps+1, 1)
        return paths[:, :, None]
    return paths

def nearest_psd_corr(mat: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    """Project a symmetric matrix to a PSD correlation matrix (simple eigenvalue clipping)."""
    A = np.asarray(mat, float)
    A = 0.5 * (A + A.T)
    np.fill_diagonal(A, 1.0)

    vals, vecs = np.linalg.eigh(A)
    vals = np.clip(vals, eps, None)
    B = vecs @ np.diag(vals) @ vecs.T
    B = 0.5 * (B + B.T)

    # Re-normalize to correlation (diagonal = 1)
    d = np.sqrt(np.clip(np.diag(B), eps, None))
    C = B / (d[:, None] * d[None, :])
    np.fill_diagonal(C, 1.0)
    return 0.5 * (C + C.T)

def cholesky_safe(corr: np.ndarray) -> np.ndarray:
    """Cholesky with fallback to nearest PSD correlation matrix."""
    try:
        return np.linalg.cholesky(corr)
    except np.linalg.LinAlgError:
        corr_psd = nearest_psd_corr(corr)
        return np.linalg.cholesky(corr_psd)

# ----------------------------
# Market model: correlated GBM
# ----------------------------
def simulate_correlated_gbm(
    S0: np.ndarray,
    r: float,
    q: np.ndarray,
    sigma: np.ndarray,
    corr: np.ndarray,
    T: float,
    n_steps: int,
    n_sims: int,
    seed: int = 42,
) -> np.ndarray:
    """
    Returns paths of shape (n_sims, n_steps+1, n_assets).
    Risk-neutral GBM: dS = (r-q)S dt + sigma S dW
    """
    rng = np.random.default_rng(int(seed))

    S0 = np.asarray(S0, float)
    q = np.asarray(q, float)
    sigma = np.asarray(sigma, float)

    n_assets = S0.size
    dt = T / n_steps

    L = cholesky_safe(corr)

    Z = rng.standard_normal((n_sims, n_steps, n_assets))
    dW = Z @ L.T  # correlate across assets

    drift = (r - q - 0.5 * sigma**2) * dt                  # (n_assets,)
    diff = sigma * np.sqrt(dt) * dW                        # (n_sims, n_steps, n_assets)

    log_increments = drift[None, None, :] + diff
    log_paths = np.cumsum(log_increments, axis=1)          # (n_sims, n_steps, n_assets)

    paths = np.empty((n_sims, n_steps + 1, n_assets))
    paths[:, 0, :] = S0
    paths[:, 1:, :] = S0[None, None, :] * np.exp(log_paths)
    return paths

# ----------------------------
# Basket engine
# ----------------------------
def basket_ratio(St_t: np.ndarray, S0: np.ndarray, kind: str = "worst-of", weights=None) -> np.ndarray:
    """
    St_t: (n_sims, n_assets) prices at time t
    S0:   (n_assets,)
    returns: (n_sims,) basket performance ratios
    """
    S0 = np.asarray(S0, float)
    R = St_t / S0[None, :]

    kind = kind.lower()
    if kind == "worst-of":
        return np.min(R, axis=1)
    if kind == "best-of":
        return np.max(R, axis=1)
    if kind == "average":
        return np.mean(R, axis=1)
    if kind == "weighted":
        w = np.asarray(weights, float)
        w = w / np.sum(w)
        return R @ w

    raise ValueError("kind must be: worst-of / best-of / average / weighted")

def build_obs_idx(T: float, steps_per_year: int, obs_per_year: int) -> np.ndarray:
    n_steps = int(round(T * steps_per_year))
    obs_step = int(round(steps_per_year / obs_per_year))
    obs_idx = np.arange(obs_step, n_steps + 1, obs_step, dtype=int)
    obs_idx = obs_idx[obs_idx <= n_steps]
    return obs_idx

# ----------------------------
# Product: Phoenix (coupon + memory + autocall + maturity barrier)
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
    weights=None,
    memory: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    paths: (n_sims, n_steps+1, n_assets)
    Returns:
      payoff: (n_sims,)
      autocalled: (n_sims,) bool
      autocall_obs: (n_sims,) int obs number (0-based) or -1
    """
    paths = ensure_3d_paths(paths)
    n_sims, _, n_assets = paths.shape

    S0 = np.asarray(S0, float)
    if S0.ndim == 0:
        S0 = np.array([float(S0)])
    if S0.size != n_assets:
        raise ValueError("S0 must match n_assets in paths.")

    payoff = np.zeros(n_sims)
    autocalled = np.zeros(n_sims, dtype=bool)
    autocall_obs = np.full(n_sims, -1, dtype=int)

    accrued = np.zeros(n_sims)  # coupon memory bucket

    for k, t in enumerate(obs_idx):
        alive = ~autocalled
        if not np.any(alive):
            break

        St = paths[alive, t, :]
        ratio = basket_ratio(St, S0, kind=basket_kind, weights=weights)

        # Coupon condition
        coupon_ok = ratio >= coupon_trigger
        if memory:
            accrued_alive = accrued[alive] + nominal * coupon_rate_per_obs
            payoff[alive] += accrued_alive * coupon_ok
            accrued[alive] = accrued_alive * (~coupon_ok)
        else:
            payoff[alive] += (nominal * coupon_rate_per_obs) * coupon_ok

        # Autocall condition
        call_ok = ratio >= call_trigger
        idx_alive = np.where(alive)[0]
        called_idx = idx_alive[call_ok]
        if called_idx.size:
            autocalled[called_idx] = True
            autocall_obs[called_idx] = k
            payoff[called_idx] += nominal

    # Maturity payoff for remaining
    alive = ~autocalled
    if np.any(alive):
        ST = paths[alive, -1, :]
        ratio_T = basket_ratio(ST, S0, kind=basket_kind, weights=weights)
        protected = ratio_T >= barrier
        payoff[alive] += np.where(protected, nominal, nominal * ratio_T)

    return payoff, autocalled, autocall_obs

# ----------------------------
# Metrics
# ----------------------------
def summarize_metrics(payoff: np.ndarray, autocalled: np.ndarray, autocall_obs: np.ndarray, obs_per_year: int) -> pd.DataFrame:
    payoff = np.asarray(payoff, float)
    autocalled = np.asarray(autocalled, bool)

    out = {
        "Expected payoff": float(np.mean(payoff)),
        "Median payoff": float(np.median(payoff)),
        "P(Autocall)": float(np.mean(autocalled)),
        "5% quantile": float(np.quantile(payoff, 0.05)),
        "1% quantile": float(np.quantile(payoff, 0.01)),
    }
    calls = autocall_obs[autocall_obs >= 0]
    if calls.size:
        out["Expected autocall time (years)"] = float(np.mean((calls + 1) / obs_per_year))
    else:
        out["Expected autocall time (years)"] = np.nan

    return pd.DataFrame(out, index=["Value"]).T

# ----------------------------
# Market data (Yahoo Finance)
# ----------------------------
@st.cache_data(show_spinner=False)
def fetch_market_params(tickers: list[str], lookback_years: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    end = pd.Timestamp.today().tz_localize(None)
    start = end - pd.DateOffset(years=int(lookback_years))

    px = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)["Close"]
    if isinstance(px, pd.Series):  # single ticker returns Series
        px = px.to_frame()

    px = px.dropna(how="all").dropna(axis=1, how="any")
    if px.shape[1] == 0:
        raise ValueError("No usable price data returned (check tickers).")

    rets = px.pct_change().dropna()
    S0 = px.iloc[-1].values
    sigma = rets.std().values * np.sqrt(252)
    corr = rets.corr().values
    return S0, sigma, corr, px

# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(page_title="Structured Products Simulator", layout="wide")
st.title("Structured Products Simulator (Phoenix) — Manual + Market Data")

with st.sidebar:

    with st.form("params_form"):

        st.header("Mode")
        mode = st.radio("Parameter source", ["Manual parameters", "Market data (Yahoo Finance)"], index=0)

        st.header("Product")
        nominal = st.number_input("Nominal", min_value=1.0, value=1000.0, step=100.0)

        T = st.number_input("Maturity (years)", min_value=0.25, value=3.0, step=0.25)
        steps_per_year = st.selectbox("Simulation steps/year", [252, 52, 12], index=0)
        obs_per_year = st.selectbox("Observation frequency/year", [12, 4, 2, 1], index=1)

        coupon_pa = st.number_input("Coupon p.a.", min_value=0.0, value=0.10, step=0.01)
        coupon_trigger = st.number_input("Coupon trigger", min_value=0.0, max_value=2.0, value=0.70, step=0.01)
        call_trigger = st.number_input("Autocall trigger", min_value=0.0, max_value=2.0, value=1.00, step=0.01)
        barrier = st.number_input("Maturity barrier", min_value=0.0, max_value=2.0, value=0.60, step=0.01)

        memory = st.checkbox("Memory coupon", value=True)

        st.header("Basket")
        basket_kind = st.selectbox("Basket type", ["worst-of", "best-of", "average", "weighted"], index=0)

        r = st.number_input("Risk-free rate r", value=0.02, step=0.005)

        n_sims = st.slider("Number of simulations", 5000, 60000, 20000)
        seed = st.number_input("Random seed", value=42)

        run_simulation = st.form_submit_button("Run Simulation")
    T = st.number_input("Maturity (years)", min_value=0.25, value=3.0, step=0.25)
    steps_per_year = st.selectbox("Simulation steps/year", [252, 52, 12], index=0)
    obs_per_year = st.selectbox("Observation frequency/year", [12, 4, 2, 1], index=1)

    coupon_pa = st.number_input("Coupon p.a. (e.g. 0.10 = 10%)", min_value=0.0, value=0.10, step=0.01)
    coupon_trigger = st.number_input("Coupon trigger (ratio)", min_value=0.0, max_value=2.0, value=0.70, step=0.01)
    call_trigger = st.number_input("Autocall trigger (ratio)", min_value=0.0, max_value=2.0, value=1.00, step=0.01)
    barrier = st.number_input("Maturity barrier (ratio)", min_value=0.0, max_value=2.0, value=0.60, step=0.01)
    memory = st.checkbox("Memory coupon", value=True)

    st.header("Basket")
    basket_kind = st.selectbox("Basket type", ["worst-of", "best-of", "average", "weighted"], index=0)

    st.header("Rates / Dividends")
    r = st.number_input("Risk-free rate r", value=0.02, step=0.005)
    use_q = st.checkbox("Set dividend yields q", value=False)

    st.header("Monte Carlo")
    n_sims = st.slider("Number of simulations", 5000, 120000, 30000, step=5000)
    seed = st.number_input("Random seed", value=42, step=1)

# --- Inputs for assets (manual or market)
tickers = None
px_preview = None

if mode == "Market data (Yahoo Finance)":
    with st.sidebar:
        st.subheader("Market data settings")
        tickers_str = st.text_input("Tickers (comma-separated)", value="AAPL,MSFT")
        lookback_years = st.slider("Lookback (years)", 1, 10, 3)

    tickers = [t.strip().upper() for t in tickers_str.split(",") if t.strip()]
    if len(tickers) == 0:
        st.error("Please enter at least 1 ticker.")
        st.stop()

    try:
        S0, sigma, corr, px_preview = fetch_market_params(tickers, lookback_years)
    except Exception as e:
        st.error(f"Market data error: {e}")
        st.stop()

    n_assets = len(S0)
    q = np.zeros(n_assets)  # MVP: keep q = 0 (you can extend later)

    if use_q:
        # allow overriding q manually even in market mode
        with st.sidebar:
            st.subheader("Dividend yields q (override)")
            q_list = []
            for i in range(n_assets):
                q_list.append(st.number_input(f"q {tickers[i]}", value=0.00, step=0.005, key=f"q_m_{i}"))
            q = np.array(q_list, float)

else:
    with st.sidebar:
        st.subheader("Manual asset settings")
        n_assets = st.number_input("Number of underlyings", min_value=1, max_value=5, value=2, step=1)

        S0_list, sigma_list, q_list = [], [], []
        for i in range(int(n_assets)):
            st.markdown(f"**Asset {i+1}**")
            S0_list.append(st.number_input(f"S0 {i+1}", value=100.0, step=1.0, key=f"S0_{i}"))
            sigma_list.append(st.number_input(f"Vol (sigma) {i+1}", value=0.25, step=0.01, key=f"sig_{i}"))
            if use_q:
                q_list.append(st.number_input(f"q {i+1}", value=0.00, step=0.005, key=f"q_{i}"))
            else:
                q_list.append(0.0)

        S0 = np.array(S0_list, float)
        sigma = np.array(sigma_list, float)
        q = np.array(q_list, float)

        st.subheader("Correlation matrix")
        corr_df = pd.DataFrame(
            np.eye(int(n_assets)),
            index=[f"A{i+1}" for i in range(int(n_assets))],
            columns=[f"A{i+1}" for i in range(int(n_assets))],
        )
        corr_df = st.data_editor(corr_df, use_container_width=True, key="corr_editor")
        corr = corr_df.to_numpy()

# --- Basket weights (if weighted)
weights = None
if basket_kind == "weighted":
    with st.sidebar:
        st.subheader("Weights")
        w_list = []
        labels = tickers if tickers else [f"A{i+1}" for i in range(int(n_assets))]
        for i, lab in enumerate(labels):
            w_list.append(st.number_input(f"Weight {lab}", value=1.0, step=0.1, key=f"w_{i}"))
        weights = np.array(w_list, float)

# --- Run simulation & product
n_steps = int(round(T * steps_per_year))
obs_idx = build_obs_idx(T, steps_per_year, int(obs_per_year))
coupon_rate_per_obs = coupon_pa / int(obs_per_year)

if run_simulation:

    with st.spinner("Simulating paths and pricing..."):
        paths = simulate_correlated_gbm(
        S0=S0,
        r=float(r),
        q=q,
        sigma=sigma,
        corr=corr,
        T=float(T),
        n_steps=n_steps,
        n_sims=int(n_sims),
        seed=int(seed),
    )

    payoff, autocalled, autocall_obs = phoenix_payoff(
        paths=paths,
        S0=S0,
        nominal=float(nominal),
        obs_idx=obs_idx,
        coupon_rate_per_obs=float(coupon_rate_per_obs),
        coupon_trigger=float(coupon_trigger),
        call_trigger=float(call_trigger),
        barrier=float(barrier),
        basket_kind=basket_kind,
        weights=weights,
        memory=bool(memory),
    )

metrics_df = summarize_metrics(payoff, autocalled, autocall_obs, int(obs_per_year))

# ----------------------------
# Output section
# ----------------------------
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Inputs (summary)")
    labels = tickers if tickers else [f"A{i+1}" for i in range(len(S0))]
    df_params = pd.DataFrame(
        {"S0": S0, "sigma": sigma, "q": q},
        index=labels
    )
    st.dataframe(df_params, use_container_width=True)
    st.caption("Correlation matrix used (may be adjusted to PSD internally if needed).")
    st.dataframe(pd.DataFrame(corr, index=labels, columns=labels), use_container_width=True)

    st.subheader("Key metrics")
    st.dataframe(metrics_df, use_container_width=True)

    out_df = pd.DataFrame(
        {
            "payoff": payoff,
            "autocalled": autocalled.astype(int),
            "autocall_obs": autocall_obs,
        }
    )
    st.download_button(
        "Download simulation results (CSV)",
        data=out_df.to_csv(index=False),
        file_name="simulation_results.csv",
        mime="text/csv",
        use_container_width=True,
    )

with col2:
    st.subheader("Payoff distribution")
    fig = plt.figure()
    plt.hist(payoff, bins=80)
    plt.xlabel("Payoff")
    plt.ylabel("Frequency")
    st.pyplot(fig)

    st.subheader("Autocall timing (obs index)")
    calls = autocall_obs[autocall_obs >= 0]
    fig2 = plt.figure()
    if calls.size:
        plt.hist(calls, bins=np.arange(0, len(obs_idx) + 1) - 0.5)
        plt.xlabel("Observation number (0-based)")
        plt.ylabel("Count")
    else:
        plt.text(0.5, 0.5, "No autocalls in this run.", ha="center", va="center")
        plt.axis("off")
    st.pyplot(fig2)

    st.subheader("Sample paths (first 30 sims, basket ratio)")
    # plot basket ratio through time for a few simulations (cleaner than plotting all assets)
    paths3 = ensure_3d_paths(paths)
    tgrid = np.linspace(0, float(T), n_steps + 1)
    fig3 = plt.figure()
    m = min(30, paths3.shape[0])
    for i in range(m):
        ratio_line = basket_ratio(paths3[i, :, :], S0, kind=basket_kind, weights=weights)
        plt.plot(tgrid, ratio_line, linewidth=0.8)
    plt.axhline(coupon_trigger, linestyle="--")
    plt.axhline(call_trigger, linestyle="--")
    plt.axhline(barrier, linestyle="--")
    plt.xlabel("Time (years)")
    plt.ylabel("Basket ratio")
    st.pyplot(fig3)

if px_preview is not None:
    st.subheader("Market data preview (auto-adjusted close)")
    st.line_chart(px_preview)

st.caption(
    "Notes: This is a simplified Monte Carlo simulator (GBM + historical vol/corr in market mode). "
    "Real desk pricing uses implied vol surfaces, dividend curves, and calibrated rate curves."
)
