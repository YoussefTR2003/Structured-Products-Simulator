# app.py
# Structured Products Simulator (Phoenix) - robust Streamlit Cloud version
# - Inputs always defined first
# - Simulation runs only when user clicks "Run simulation"
# - Results stored in st.session_state to avoid recomputation on every rerun

from __future__ import annotations
from typing import Optional
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import yfinance as yf


# ----------------------------
# Utils
# ----------------------------
def ensure_3d_paths(paths: np.ndarray) -> np.ndarray:
    paths = np.asarray(paths, float)
    if paths.ndim == 2:
        return paths[:, :, None]
    return paths


def nearest_psd_corr(mat: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    A = np.asarray(mat, float)
    A = 0.5 * (A + A.T)
    np.fill_diagonal(A, 1.0)

    vals, vecs = np.linalg.eigh(A)
    vals = np.clip(vals, eps, None)
    B = vecs @ np.diag(vals) @ vecs.T
    B = 0.5 * (B + B.T)

    d = np.sqrt(np.clip(np.diag(B), eps, None))
    C = B / (d[:, None] * d[None, :])
    np.fill_diagonal(C, 1.0)
    return 0.5 * (C + C.T)


def cholesky_safe(corr: np.ndarray) -> np.ndarray:
    try:
        return np.linalg.cholesky(corr)
    except np.linalg.LinAlgError:
        return np.linalg.cholesky(nearest_psd_corr(corr))


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
    rng = np.random.default_rng(int(seed))

    S0 = np.asarray(S0, float)
    q = np.asarray(q, float)
    sigma = np.asarray(sigma, float)
    corr = np.asarray(corr, float)

    n_assets = S0.size
    dt = T / n_steps

    L = cholesky_safe(corr)

    Z = rng.standard_normal((n_sims, n_steps, n_assets))
    dW = Z @ L.T

    drift = (r - q - 0.5 * sigma**2) * dt
    diff = sigma * np.sqrt(dt) * dW

    log_increments = drift[None, None, :] + diff
    log_paths = np.cumsum(log_increments, axis=1)

    paths = np.empty((n_sims, n_steps + 1, n_assets))
    paths[:, 0, :] = S0
    paths[:, 1:, :] = S0[None, None, :] * np.exp(log_paths)
    return paths


# ----------------------------
# Basket
# ----------------------------
def basket_ratio(St_t: np.ndarray, S0: np.ndarray, kind: str = "worst-of", weights=None) -> np.ndarray:
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
    return obs_idx[obs_idx <= n_steps]


# ----------------------------
# Phoenix payoff
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
    # ---- PRICING PARAMS (NEW) ----
    r: float = 0.0,
    steps_per_year: int = 252,
    T: Optional[float] = None,  # if None, inferred from paths,   # if None, inferred from paths
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns discounted PV per simulation (price paths already simulated under Q).
    PV = sum_k DF(t_k)*CF(t_k) + DF(T)*CF(T)
    """
    paths = ensure_3d_paths(paths)
    n_sims, n_steps_plus1, n_assets = paths.shape

    S0 = np.asarray(S0, float)
    if S0.ndim == 0:
        S0 = np.array([float(S0)])
    if S0.size != n_assets:
        raise ValueError("S0 must match number of assets in paths.")

    # infer maturity if not provided
    if T is None:
        n_steps = n_steps_plus1 - 1
        T = n_steps / float(steps_per_year)

    pv = np.zeros(n_sims)  # discounted payoff
    autocalled = np.zeros(n_sims, dtype=bool)
    autocall_obs = np.full(n_sims, -1, dtype=int)

    accrued = np.zeros(n_sims)  # undiscounted accrued coupon amount (for memory)

    for k, step in enumerate(obs_idx):
        alive = ~autocalled
        if not np.any(alive):
            break

        time = step / float(steps_per_year)
        df = np.exp(-float(r) * float(time))

        ratio = basket_ratio(paths[alive, step, :], S0, kind=basket_kind, weights=weights)

        # ---- Coupons (paid at this observation date, discounted by df) ----
        coupon_ok = ratio >= coupon_trigger

        if memory:
            # accrue one coupon for alive paths (undiscounted amount)
            accrued_alive = accrued[alive] + nominal * coupon_rate_per_obs
            # if coupon condition met, pay all accrued now (discounted)
            pv[alive] += (accrued_alive * coupon_ok) * df
            # if not met, keep it in memory
            accrued[alive] = accrued_alive * (~coupon_ok)
        else:
            pv[alive] += (nominal * coupon_rate_per_obs) * coupon_ok * df

        # ---- Autocall (nominal paid at this observation date, discounted by df) ----
        call_ok = ratio >= call_trigger
        idx_alive = np.where(alive)[0]
        called_idx = idx_alive[call_ok]
        if called_idx.size:
            autocalled[called_idx] = True
            autocall_obs[called_idx] = k
            pv[called_idx] += nominal * df  # discount nominal repayment at call date

    # ---- Maturity cashflow (discounted at DF(T)) ----
    alive = ~autocalled
    if np.any(alive):
        dfT = np.exp(-float(r) * float(T))
        ratio_T = basket_ratio(paths[alive, -1, :], S0, kind=basket_kind, weights=weights)
        protected = ratio_T >= barrier
        maturity_cf = np.where(protected, nominal, nominal * ratio_T)
        pv[alive] += maturity_cf * dfT

    return pv, autocalled, autocall_obs


# ----------------------------
# Metrics
# ----------------------------
def summarize_metrics(payoff: np.ndarray, autocalled: np.ndarray, autocall_obs: np.ndarray, obs_per_year: int) -> pd.DataFrame:
    payoff = np.asarray(payoff, float)
    autocalled = np.asarray(autocalled, bool)

    out = {
        "Estimated price (PV)": float(np.mean(payoff)),
        "Median payoff": float(np.median(payoff)),
        "P(Autocall)": float(np.mean(autocalled)),
        "5% quantile": float(np.quantile(payoff, 0.05)),
        "1% quantile": float(np.quantile(payoff, 0.01)),
    }
    calls = autocall_obs[autocall_obs >= 0]
    out["Expected autocall time (years)"] = float(np.mean((calls + 1) / obs_per_year)) if calls.size else np.nan
    return pd.DataFrame(out, index=["Value"]).T


# ----------------------------
# Market data
# ----------------------------
@st.cache_data(show_spinner=False)
def fetch_market_params(tickers: list[str], lookback_years: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    end = pd.Timestamp.today().tz_localize(None)
    start = end - pd.DateOffset(years=int(lookback_years))

    df = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)

    if isinstance(df.columns, pd.MultiIndex):
        px = df["Close"] if "Close" in df.columns.get_level_values(0) else df["Adj Close"]
    else:
        px = df["Close"] if "Close" in df.columns else df

    if isinstance(px, pd.Series):
        px = px.to_frame()

    px = px.dropna(how="all").dropna(axis=1, how="any")
    if px.shape[1] == 0:
        raise ValueError("No usable price data returned (check tickers and lookback).")

    rets = px.pct_change().dropna()
    S0 = px.iloc[-1].values
    sigma = rets.std().values * np.sqrt(252)
    corr = rets.corr().values
    return S0, sigma, corr, px


# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(page_title="Structured Products Simulator made by Youssef Triki", layout="wide")
st.title("Structured Products Simulator (Phoenix) made by Youssef Triki")

if "results" not in st.session_state:
    st.session_state["results"] = None

with st.sidebar:
    st.header("Mode")
    mode = st.radio("Parameter source", ["Manual parameters", "Market data (Yahoo Finance)"], index=0)

    st.header("Product")
    nominal = st.number_input("Nominal", min_value=1.0, value=1000.0, step=100.0)
    T = st.number_input("Maturity (years)", min_value=0.25, value=3.0, step=0.25)
    steps_per_year = st.selectbox("Simulation steps/year", [252, 52, 12], index=0)
    obs_per_year = st.selectbox("Observation frequency/year", [12, 4, 2, 1], index=1)

    coupon_pa = st.number_input("Coupon p.a.", min_value=0.0, value=0.10, step=0.01)
    coupon_pa = coupon_pa_pct / 100
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
    # keep cloud-safe
    n_sims = st.slider("Number of simulations", 2000, 60000, 20000, step=1000)
    seed = st.number_input("Random seed", value=42, step=1)

    # Mode-specific inputs (still defined BEFORE running)
    px_preview = None
    tickers = None
    lookback_years = None

    if mode == "Market data (Yahoo Finance)":
        st.subheader("Market data settings")
        tickers_str = st.text_input("Tickers (comma-separated)", value="AAPL,MSFT")
        lookback_years = st.slider("Lookback (years)", 1, 10, 3)
        tickers = [t.strip().upper() for t in tickers_str.split(",") if t.strip()]

    else:
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

        S0_manual = np.array(S0_list, float)
        sigma_manual = np.array(sigma_list, float)
        q_manual = np.array(q_list, float)

        st.subheader("Correlation matrix")
        corr_df = pd.DataFrame(
            np.eye(int(n_assets)),
            index=[f"A{i+1}" for i in range(int(n_assets))],
            columns=[f"A{i+1}" for i in range(int(n_assets))],
        )
        corr_df = st.data_editor(corr_df, width="stretch", key="corr_editor")
        corr_manual = corr_df.to_numpy()

    weights = None
    if basket_kind == "weighted":
        st.subheader("Weights")
        if mode == "Market data (Yahoo Finance)":
            labels_for_w = tickers if tickers else ["A1"]
        else:
            labels_for_w = [f"A{i+1}" for i in range(int(n_assets))]
        w_list = []
        for i, lab in enumerate(labels_for_w):
            w_list.append(st.number_input(f"Weight {lab}", value=1.0, step=0.1, key=f"w_{i}"))
        weights = np.asarray(w_list, float)

    run = st.button("Run simulation", type="primary")

if run:
    try:
        if mode == "Market data (Yahoo Finance)":
            if not tickers:
                st.error("Please enter at least one ticker.")
                st.stop()

            S0, sigma, corr, px_preview = fetch_market_params(tickers, int(lookback_years))
            q = np.zeros_like(S0)
            if use_q:
                # keep q = 0 for now in market mode unless you extend with inputs
                pass
            labels = tickers

        else:
            S0, sigma, corr, q = S0_manual, sigma_manual, corr_manual, q_manual
            labels = [f"A{i+1}" for i in range(len(S0))]

        n_steps = int(round(float(T) * int(steps_per_year)))
        obs_idx = build_obs_idx(float(T), int(steps_per_year), int(obs_per_year))
        coupon_rate_per_obs = float(coupon_pa) / int(obs_per_year)

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
                paths=paths,S0=S0,nominal=float(nominal),obs_idx=obs_idx,coupon_rate_per_obs=float(coupon_rate_per_obs),
                coupon_trigger=float(coupon_trigger),call_trigger=float(call_trigger),barrier=float(barrier),
                basket_kind=basket_kind,weights=weights,memory=bool(memory),r=float(r),steps_per_year=int(steps_per_year),
                T=float(T),

            )

            metrics_df = summarize_metrics(payoff, autocalled, autocall_obs, int(obs_per_year))

        st.session_state["results"] = {
            "labels": labels,
            "S0": S0,
            "sigma": sigma,
            "q": q,
            "corr": corr,
            "payoff": payoff,
            "autocalled": autocalled,
            "autocall_obs": autocall_obs,
            "metrics_df": metrics_df,
            "paths": paths,
            "obs_idx": obs_idx,
            "n_steps": n_steps,
            "T": float(T),
            "basket_kind": basket_kind,
            "weights": weights,
            "coupon_trigger": float(coupon_trigger),
            "call_trigger": float(call_trigger),
            "barrier": float(barrier),
            "px_preview": px_preview,
        }

    except Exception as e:
        st.error(f"Run failed: {e}")
        st.stop()

res = st.session_state.get("results")
if res is None:
    st.info("Set parameters in the sidebar and click **Run simulation**.")
    st.stop()

labels = res["labels"]
S0 = res["S0"]
sigma = res["sigma"]
q = res["q"]
corr = res["corr"]
payoff = res["payoff"]
autocalled = res["autocalled"]
autocall_obs = res["autocall_obs"]
metrics_df = res["metrics_df"]
paths = res["paths"]
obs_idx = res["obs_idx"]
n_steps = res["n_steps"]
T_val = res["T"]
basket_kind = res["basket_kind"]
weights = res["weights"]
coupon_trigger = res["coupon_trigger"]
call_trigger = res["call_trigger"]
barrier = res["barrier"]
px_preview = res["px_preview"]

col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Inputs (summary)")
    st.dataframe(pd.DataFrame({"S0": S0, "sigma": sigma, "q": q}, index=labels), width="stretch")
    st.caption("Correlation matrix used (may be adjusted to PSD internally).")
    st.dataframe(pd.DataFrame(corr, index=labels, columns=labels), width="stretch")

    st.subheader("Key metrics")
    st.dataframe(metrics_df, width="stretch")

    out_df = pd.DataFrame({"payoff": payoff, "autocalled": autocalled.astype(int), "autocall_obs": autocall_obs})
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
    st.pyplot(fig, clear_figure=True)

    st.subheader("Autocall timing (observation number)")
    calls = autocall_obs[autocall_obs >= 0]
    fig2 = plt.figure()
    if calls.size:
        plt.hist(calls, bins=np.arange(0, len(obs_idx) + 1) - 0.5)
        plt.xlabel("Observation number (0-based)")
        plt.ylabel("Count")
    else:
        plt.text(0.5, 0.5, "No autocalls in this run.", ha="center", va="center")
        plt.axis("off")
    st.pyplot(fig2, clear_figure=True)

    st.subheader("Sample paths (basket ratio, first 30 sims)")
    paths3 = ensure_3d_paths(paths)
    tgrid = np.linspace(0, float(T_val), n_steps + 1)
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
    st.pyplot(fig3, clear_figure=True)

if px_preview is not None:
    st.subheader("Market data preview (auto-adjusted close)")
    st.line_chart(px_preview)

st.caption(
    "Notes: simplified GBM + historical vol/corr in market mode. "
    "Desk pricing typically uses implied vol surfaces, dividend curves, and calibrated rate curves."
)

if __name__ == "__main__":
    pass
