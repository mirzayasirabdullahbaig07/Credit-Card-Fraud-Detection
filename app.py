"""
Credit Card Fraud Detection — Streamlit App
Loads the artifacts produced by Credit_Card_Fraud_Detection_Colab.ipynb:
    - fraud_detection_model.pkl
    - scaler_amount.pkl
    - scaler_time.pkl
    - feature_columns.pkl

Built by Mirza Yasir Abdullah Baig
"""

import numpy as np
import pandas as pd
import streamlit as st
import joblib
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Fraud Radar — Credit Card Fraud Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

AUTHOR_NAME = "Mirza Yasir Abdullah Baig"
GITHUB_URL = "https://github.com/mirzayasirabdullahbaig07"
LINKEDIN_URL = "https://www.linkedin.com/in/mirza-yasir-abdullah-baig/"
KAGGLE_URL = "https://www.kaggle.com/myab07"

# ---------------------------------------------------------------------------
# Global styling — "security command-center" theme
# ---------------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700;800&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500;700&display=swap');

:root {
    --bg: #080B13;
    --surface: rgba(22, 29, 45, 0.55);
    --surface-solid: #121826;
    --surface-2: #171F30;
    --border: #232C40;
    --border-soft: rgba(45, 58, 85, 0.5);
    --cyan: #22D3EE;
    --cyan-dim: rgba(34, 211, 238, 0.15);
    --red: #F43F5E;
    --green: #34D399;
    --amber: #FBBF24;
    --text: #E9ECF3;
    --muted: #8B95A7;
}

html, body, [data-testid="stAppViewContainer"] {
    background:
        radial-gradient(circle at 12% -8%, rgba(34,211,238,0.07) 0%, transparent 38%),
        radial-gradient(circle at 90% 10%, rgba(244,63,94,0.05) 0%, transparent 35%),
        linear-gradient(180deg, #0A0F1C 0%, var(--bg) 100%) !important;
    color: var(--text);
    font-family: 'Inter', sans-serif;
}

[data-testid="stAppViewContainer"]::before {
    content: '';
    position: fixed; inset: 0; pointer-events: none; z-index: 0;
    background-image:
        linear-gradient(rgba(255,255,255,0.012) 1px, transparent 1px),
        linear-gradient(90deg, rgba(255,255,255,0.012) 1px, transparent 1px);
    background-size: 42px 42px;
}

[data-testid="stHeader"] { background: transparent !important; }
[data-testid="block-container"] { padding-top: 2rem; }

[data-testid="stSidebar"] {
    background: #060911 !important;
    border-right: 1px solid var(--border);
}
[data-testid="stSidebar"] * { font-family: 'Inter', sans-serif; }

h1, h2, h3, h4 { font-family: 'Space Grotesk', sans-serif !important; letter-spacing: 0.2px; }

.hero {
    display: flex;
    align-items: center;
    gap: 18px;
    padding: 6px 24px 24px 24px;
    margin: -6px -24px 30px -24px;
    border-bottom: 1px solid var(--border-soft);
    position: relative;
    overflow: hidden;
}
.hero::after {
    content: '';
    position: absolute; left: 0; right: 0; bottom: -1px; height: 1px;
    background: linear-gradient(90deg, transparent, var(--cyan), transparent);
    animation: sweep 4s ease-in-out infinite;
}
@keyframes sweep { 0%,100% { opacity: 0.15; transform: translateX(-30%); } 50% { opacity: 0.9; transform: translateX(30%); } }

.radar {
    position: relative;
    width: 50px; height: 50px;
    border-radius: 50%;
    border: 1px solid var(--cyan);
    flex-shrink: 0;
    overflow: hidden;
    background: #05070C;
    box-shadow: 0 0 22px rgba(34, 211, 238, 0.25), inset 0 0 12px rgba(34,211,238,0.08);
}
.radar::before {
    content: '';
    position: absolute; inset: 0;
    background: conic-gradient(from 0deg, rgba(34,211,238,0.75), transparent 42%);
    animation: spin 3.2s linear infinite;
}
.radar::after {
    content: '';
    position: absolute; inset: 40%;
    border-radius: 50%;
    background: var(--cyan);
    box-shadow: 0 0 10px var(--cyan), 0 0 22px rgba(34,211,238,0.6);
}
@keyframes spin { to { transform: rotate(360deg); } }

.hero-title {
    font-size: 30px; font-weight: 800; margin: 0; color: var(--text);
    background: linear-gradient(90deg, #F5F7FB 30%, var(--cyan) 100%);
    -webkit-background-clip: text; background-clip: text; -webkit-text-fill-color: transparent;
}
.hero-sub {
    font-family: 'JetBrains Mono', monospace;
    font-size: 12px; color: var(--muted); letter-spacing: 1.8px;
    margin: 5px 0 0 0; text-transform: uppercase;
}
.status-dot {
    display: inline-block; width: 8px; height: 8px; border-radius: 50%;
    background: var(--green); box-shadow: 0 0 8px var(--green);
    margin-right: 8px; animation: pulse 2s infinite;
}
@keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.3; } }

div[data-testid="stVerticalBlockBorderWrapper"] {
    background: var(--surface) !important;
    border: 1px solid var(--border-soft) !important;
    border-radius: 16px !important;
    backdrop-filter: blur(10px);
    box-shadow: 0 4px 24px rgba(0,0,0,0.25);
    transition: border-color 0.25s ease, box-shadow 0.25s ease;
}
div[data-testid="stVerticalBlockBorderWrapper"]:hover {
    border-color: rgba(34, 211, 238, 0.35) !important;
    box-shadow: 0 4px 28px rgba(34, 211, 238, 0.08);
}

div[data-testid="stMetric"] {
    background: var(--surface-2);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 16px 18px;
}
div[data-testid="stMetricLabel"] {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 11px !important;
    letter-spacing: 1.3px;
    text-transform: uppercase;
    color: var(--muted) !important;
}
div[data-testid="stMetricValue"] {
    font-family: 'JetBrains Mono', monospace !important;
    color: var(--text) !important;
}

.stButton > button {
    background: linear-gradient(135deg, #1B2A44, #0E1524) !important;
    color: var(--cyan) !important;
    border: 1px solid var(--cyan) !important;
    border-radius: 10px !important;
    font-family: 'JetBrains Mono', monospace !important;
    letter-spacing: 0.6px;
    font-weight: 500 !important;
    padding: 0.6rem 1.2rem !important;
    transition: all 0.2s ease;
}
.stButton > button:hover {
    background: var(--cyan) !important;
    color: #06131A !important;
    box-shadow: 0 0 22px rgba(34, 211, 238, 0.4);
    transform: translateY(-1px);
}

[data-testid="stSidebar"] .stRadio > label { font-family: 'JetBrains Mono', monospace; }

.gauge-wrap { display: flex; justify-content: center; padding: 10px 0 6px 0; animation: fadein 0.5s ease; }
@keyframes fadein { from { opacity: 0; transform: scale(0.92); } to { opacity: 1; transform: scale(1); } }
.gauge {
    width: 182px; height: 182px; border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    transition: background 0.6s ease;
    padding: 6px;
}
.gauge-inner {
    width: 138px; height: 138px; border-radius: 50%;
    background: var(--surface-solid);
    display: flex; flex-direction: column;
    align-items: center; justify-content: center;
    border: 1px solid var(--border);
}
.gauge-value { font-family: 'JetBrains Mono', monospace; font-size: 31px; font-weight: 700; }
.gauge-label {
    font-family: 'JetBrains Mono', monospace; font-size: 10px;
    letter-spacing: 2px; color: var(--muted); margin-top: 6px;
}

.verdict {
    text-align: center; font-family: 'JetBrains Mono', monospace;
    font-size: 13px; letter-spacing: 1.5px; text-transform: uppercase;
    padding: 10px; border-radius: 10px; margin-top: 12px; font-weight: 700;
    animation: fadein 0.5s ease;
}
.verdict-fraud { background: rgba(244, 63, 94, 0.12); color: var(--red); border: 1px solid rgba(244,63,94,0.4); box-shadow: 0 0 20px rgba(244,63,94,0.12); }
.verdict-legit { background: rgba(52, 211, 153, 0.12); color: var(--green); border: 1px solid rgba(52,211,153,0.4); box-shadow: 0 0 20px rgba(52,211,153,0.12); }

.eyebrow {
    font-family: 'JetBrains Mono', monospace; font-size: 11px;
    letter-spacing: 2px; text-transform: uppercase; color: var(--cyan);
    margin-bottom: 4px;
}

.profile-card {
    border: 1px solid var(--border-soft);
    border-radius: 14px;
    padding: 16px 14px;
    background: linear-gradient(160deg, rgba(34,211,238,0.06), rgba(255,255,255,0.01));
    margin-top: 6px;
}
.profile-avatar {
    width: 40px; height: 40px; border-radius: 10px;
    background: linear-gradient(135deg, var(--cyan), #0E7490);
    display: flex; align-items: center; justify-content: center;
    font-family: 'Space Grotesk', sans-serif; font-weight: 700; font-size: 15px;
    color: #06131A; flex-shrink: 0;
}
.profile-name {
    font-family: 'Space Grotesk', sans-serif; font-weight: 700;
    font-size: 13.5px; color: var(--text); line-height: 1.25; margin: 0;
}
.profile-role {
    font-family: 'JetBrains Mono', monospace; font-size: 10px;
    color: var(--muted); letter-spacing: 0.8px; margin: 2px 0 0 0;
}
.profile-links { display: flex; gap: 8px; margin-top: 12px; }
.profile-link {
    flex: 1; text-align: center; text-decoration: none !important;
    font-family: 'JetBrains Mono', monospace; font-size: 10.5px; font-weight: 700;
    letter-spacing: 0.5px; color: var(--muted) !important;
    border: 1px solid var(--border-soft); border-radius: 8px;
    padding: 7px 4px; transition: all 0.2s ease;
}
.profile-link:hover {
    color: var(--cyan) !important; border-color: var(--cyan);
    background: var(--cyan-dim); transform: translateY(-1px);
}

.app-footer {
    margin-top: 48px; padding: 22px 4px 6px 4px;
    border-top: 1px solid var(--border-soft);
    display: flex; align-items: center; justify-content: space-between;
    flex-wrap: wrap; gap: 10px;
}
.app-footer .left {
    font-family: 'JetBrains Mono', monospace; font-size: 11px; color: var(--muted);
    letter-spacing: 0.4px;
}
.app-footer .left b { color: var(--text); }
.app-footer .links { display: flex; gap: 14px; }
.app-footer .links a {
    font-family: 'JetBrains Mono', monospace; font-size: 11px;
    color: var(--muted); text-decoration: none; letter-spacing: 0.4px;
    border-bottom: 1px solid transparent; padding-bottom: 2px;
    transition: all 0.2s ease;
}
.app-footer .links a:hover { color: var(--cyan); border-color: var(--cyan); }

hr { border-color: var(--border) !important; }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Load artifacts
# ---------------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    model = joblib.load("fraud_detection_model.pkl")
    scaler_amount = joblib.load("scaler_amount.pkl")
    scaler_time = joblib.load("scaler_time.pkl")
    feature_columns = joblib.load("feature_columns.pkl")
    return model, scaler_amount, scaler_time, feature_columns


try:
    model, scaler_amount, scaler_time, feature_columns = load_artifacts()
    ARTIFACTS_OK = True
except FileNotFoundError:
    ARTIFACTS_OK = False

V_COLS = [f"V{i}" for i in range(1, 29)]


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["scaled_amount"] = scaler_amount.transform(df[["Amount"]])
    df["scaled_time"] = scaler_time.transform(df[["Time"]])
    df = df.drop(columns=["Amount", "Time"])
    return df[feature_columns]


def predict(df_raw: pd.DataFrame, threshold: float):
    X = preprocess(df_raw)
    probs = model.predict_proba(X)[:, 1]
    preds = (probs >= threshold).astype(int)
    return preds, probs


def gauge_color(prob: float) -> str:
    if prob >= 0.7:
        return "var(--red)"
    elif prob >= 0.4:
        return "var(--amber)"
    return "var(--green)"


def render_gauge(prob: float):
    pct = prob * 100
    deg = pct * 3.6
    color = gauge_color(prob)
    st.markdown(f"""
    <div class="gauge-wrap">
      <div class="gauge" style="background: conic-gradient({color} {deg}deg, #1A2233 {deg}deg);">
        <div class="gauge-inner">
          <div class="gauge-value" style="color:{color};">{pct:.1f}%</div>
          <div class="gauge-label">FRAUD RISK</div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)


def render_hero():
    st.markdown("""
    <div class="hero">
      <div class="radar"></div>
      <div>
        <p class="hero-title">Fraud Radar</p>
        <p class="hero-sub"><span class="status-dot"></span>credit card transaction screening system</p>
      </div>
    </div>
    """, unsafe_allow_html=True)


def render_footer():
    st.markdown(f"""
    <div class="app-footer">
        <div class="left">Fraud Radar · built by <b>{AUTHOR_NAME}</b> · scikit-learn + Streamlit</div>
        <div class="links">
            <a href="{GITHUB_URL}" target="_blank">GitHub ↗</a>
            <a href="{LINKEDIN_URL}" target="_blank">LinkedIn ↗</a>
            <a href="{KAGGLE_URL}" target="_blank">Kaggle ↗</a>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("""
    <div style="display:flex;align-items:center;gap:10px;padding:6px 0 18px 0;">
        <div class="radar" style="width:32px;height:32px;"></div>
        <span style="font-family:'Space Grotesk';font-weight:800;font-size:18px;">Fraud Radar</span>
    </div>
    """, unsafe_allow_html=True)

    page = st.radio(
        "NAVIGATE",
        ["🎯  Single Transaction", "📊  Batch Prediction", "ℹ️  About"],
        label_visibility="visible",
    )

    st.markdown("<div class='eyebrow' style='margin-top:22px;'>Decision Threshold</div>", unsafe_allow_html=True)
    threshold = st.slider(
        " ", min_value=0.05, max_value=0.95, value=0.50, step=0.05,
        label_visibility="collapsed",
        help="Lower = catch more fraud but more false alarms. Higher = fewer false alarms, may miss fraud.",
    )
    st.caption(f"Flagging transactions with predicted fraud probability ≥ {threshold:.0%}")

    st.markdown("---")
    st.caption("Model artifacts loaded from disk · trained on SMOTE-balanced data, evaluated on real class distribution.")

    initials = "".join([w[0] for w in AUTHOR_NAME.split()[:2]]).upper()
    st.markdown(f"""
    <div class="profile-card">
        <div style="display:flex;align-items:center;gap:10px;">
            <div class="profile-avatar">{initials}</div>
            <div>
                <p class="profile-name">{AUTHOR_NAME}</p>
                <p class="profile-role">ML / DATA SCIENCE</p>
            </div>
        </div>
        <div class="profile-links">
            <a class="profile-link" href="{GITHUB_URL}" target="_blank">GitHub</a>
            <a class="profile-link" href="{LINKEDIN_URL}" target="_blank">LinkedIn</a>
            <a class="profile-link" href="{KAGGLE_URL}" target="_blank">Kaggle</a>
        </div>
    </div>
    """, unsafe_allow_html=True)

if not ARTIFACTS_OK:
    render_hero()
    st.error(
        "Model artifacts not found. Make sure `fraud_detection_model.pkl`, "
        "`scaler_amount.pkl`, `scaler_time.pkl`, and `feature_columns.pkl` "
        "are in the same folder as this app (produced by the Colab notebook)."
    )
    st.stop()

# ---------------------------------------------------------------------------
# Page: Single Transaction
# ---------------------------------------------------------------------------
if page.startswith("🎯"):
    render_hero()
    st.markdown("<div class='eyebrow'>Scan · Single Transaction</div>", unsafe_allow_html=True)
    st.markdown("#### Check a transaction against the model")
    st.caption(
        "`V1`–`V28` are anonymized PCA components from the original dataset. "
        "If you don't have real values, leave them at 0."
    )

    left, right = st.columns([1.3, 1], gap="large")

    with left:
        with st.container(border=True):
            st.markdown("<div class='eyebrow'>Transaction Details</div>", unsafe_allow_html=True)
            c1, c2 = st.columns(2)
            with c1:
                time_val = st.number_input("Time (seconds since first transaction)", value=0.0, step=1.0)
            with c2:
                amount_val = st.number_input("Amount ($)", value=0.0, min_value=0.0, step=1.0)

            with st.expander("Advanced — V1–V28 PCA features (optional)"):
                v_values = {}
                cols = st.columns(4)
                for i, v in enumerate(V_COLS):
                    with cols[i % 4]:
                        v_values[v] = st.number_input(v, value=0.0, format="%.4f", key=v)

            run = st.button("🔍  Scan Transaction", type="primary", use_container_width=True)

    with right:
        with st.container(border=True):
            st.markdown("<div class='eyebrow' style='text-align:center;'>Result</div>", unsafe_allow_html=True)
            if run:
                row = {"Time": time_val, "Amount": amount_val, **v_values}
                df_input = pd.DataFrame([row])
                preds, probs = predict(df_input, threshold)
                pred, prob = preds[0], probs[0]

                render_gauge(prob)
                if pred == 1:
                    st.markdown("<div class='verdict verdict-fraud'>🚨 Fraud Detected</div>", unsafe_allow_html=True)
                else:
                    st.markdown("<div class='verdict verdict-legit'>✅ Legitimate</div>", unsafe_allow_html=True)
                st.caption(f"Decision threshold: {threshold:.0%} · Model confidence based on trained classifier probability output.")
            else:
                st.info("Enter transaction details and click Scan Transaction to see the result here.")

# ---------------------------------------------------------------------------
# Page: Batch Prediction
# ---------------------------------------------------------------------------
elif page.startswith("📊"):
    render_hero()
    st.markdown("<div class='eyebrow'>Scan · Batch</div>", unsafe_allow_html=True)
    st.markdown("#### Score a batch of transactions")
    st.caption(
        "Upload a CSV with columns `Time, V1, ..., V28, Amount` "
        "(and optionally `Class` for ground-truth evaluation)."
    )

    with st.container(border=True):
        uploaded = st.file_uploader("Upload transactions CSV", type=["csv"], label_visibility="collapsed")

    if uploaded is not None:
        df = pd.read_csv(uploaded)
        required = ["Time", "Amount"] + V_COLS
        missing = [c for c in required if c not in df.columns]

        if missing:
            st.error(f"CSV is missing required columns: {missing}")
        else:
            has_labels = "Class" in df.columns
            preds, probs = predict(df[required], threshold)
            results = df.copy()
            results["Fraud_Probability"] = probs
            results["Prediction"] = np.where(preds == 1, "Fraud", "Legit")

            n_fraud = int((preds == 1).sum())
            c1, c2, c3 = st.columns(3)
            c1.metric("TOTAL TRANSACTIONS", f"{len(df):,}")
            c2.metric("FLAGGED AS FRAUD", f"{n_fraud:,}")
            c3.metric("FRAUD RATE", f"{n_fraud / len(df):.2%}")

            st.markdown("<div class='eyebrow' style='margin-top:18px;'>Results</div>", unsafe_allow_html=True)
            with st.container(border=True):
                styled = results.sort_values("Fraud_Probability", ascending=False).style.background_gradient(
                    subset=["Fraud_Probability"], cmap="Reds"
                )
                st.dataframe(styled, use_container_width=True, height=380)

                csv_out = results.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "⬇️  Download results as CSV",
                    data=csv_out,
                    file_name="fraud_predictions.csv",
                    mime="text/csv",
                )

            if has_labels:
                st.markdown("<div class='eyebrow' style='margin-top:18px;'>Evaluation vs Ground Truth</div>", unsafe_allow_html=True)
                from sklearn.metrics import (
                    precision_score, recall_score, f1_score, ConfusionMatrixDisplay,
                )
                y_true = df["Class"].values

                with st.container(border=True):
                    p1, p2, p3 = st.columns(3)
                    p1.metric("PRECISION", f"{precision_score(y_true, preds):.3f}")
                    p2.metric("RECALL", f"{recall_score(y_true, preds):.3f}")
                    p3.metric("F1 SCORE", f"{f1_score(y_true, preds):.3f}")

                    fig, ax = plt.subplots(figsize=(4, 4))
                    fig.patch.set_facecolor("#121826")
                    ax.set_facecolor("#121826")
                    ConfusionMatrixDisplay.from_predictions(
                        y_true, preds, display_labels=["Legit", "Fraud"], cmap="Blues", ax=ax, colorbar=False
                    )
                    ax.tick_params(colors="#E5E7EB")
                    ax.xaxis.label.set_color("#E5E7EB")
                    ax.yaxis.label.set_color("#E5E7EB")
                    ax.title.set_color("#E5E7EB")
                    st.pyplot(fig, use_container_width=False)

# ---------------------------------------------------------------------------
# Page: About
# ---------------------------------------------------------------------------
else:
    render_hero()
    st.markdown("<div class='eyebrow'>System Info</div>", unsafe_allow_html=True)
    st.markdown("#### About Fraud Radar")

    with st.container(border=True):
        st.markdown(f"""
This app screens credit card transactions for fraud using a model trained on the
Kaggle Credit Card Fraud Detection dataset (284,807 transactions, 492 confirmed frauds).

**Pipeline**
- `RobustScaler` applied to `Time` and `Amount` (the only non-PCA features)
- Class imbalance handled with SMOTE, applied only to training data
- Model trained and evaluated with a stratified, untouched test set
- Evaluated on Precision, Recall, F1, and ROC-AUC — accuracy alone is
  misleading on a dataset this imbalanced

**How to use**
- **Single Transaction** — manually enter transaction details for an instant scan
- **Batch Prediction** — upload a CSV of transactions and get predictions
  for all of them at once, plus evaluation metrics if you include the
  true `Class` labels

Adjust the decision threshold in the sidebar to trade off between
catching more fraud (lower threshold) and reducing false alarms (higher threshold).

---

**Built by [{AUTHOR_NAME}]({GITHUB_URL})**
[GitHub]({GITHUB_URL}) · [LinkedIn]({LINKEDIN_URL}) · [Kaggle]({KAGGLE_URL})
        """)

render_footer()
