"""
Credit Card Fraud Detection — Streamlit App
Loads the artifacts produced by Credit_Card_Fraud_Detection_Colab.ipynb:
    - fraud_detection_model.pkl
    - scaler_amount.pkl
    - scaler_time.pkl
    - feature_columns.pkl
"""

import numpy as np
import pandas as pd
import streamlit as st
import joblib
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide",
)

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
    """Take a raw dataframe with Time, V1-V28, Amount and return it in the
    exact scaled/ordered format the model was trained on."""
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


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
st.sidebar.title("💳 Fraud Detection")
page = st.sidebar.radio("Navigate", ["Single Transaction", "Batch Prediction (CSV)", "About"])

threshold = st.sidebar.slider(
    "Fraud decision threshold",
    min_value=0.05, max_value=0.95, value=0.50, step=0.05,
    help="Lower = catches more fraud but more false alarms. Higher = fewer false alarms but may miss fraud.",
)

if not ARTIFACTS_OK:
    st.error(
        "Model artifacts not found. Make sure `fraud_detection_model.pkl`, "
        "`scaler_amount.pkl`, `scaler_time.pkl`, and `feature_columns.pkl` "
        "are in the same folder as this app (produced by the Colab notebook)."
    )
    st.stop()

# ---------------------------------------------------------------------------
# Page: Single Transaction
# ---------------------------------------------------------------------------
if page == "Single Transaction":
    st.title("Check a Single Transaction")
    st.caption(
        "Enter transaction details below. `V1`–`V28` are anonymized PCA "
        "components from the original dataset — if you don't have real "
        "values, leave them at 0."
    )

    col1, col2 = st.columns(2)
    with col1:
        time_val = st.number_input("Time (seconds since first transaction)", value=0.0, step=1.0)
    with col2:
        amount_val = st.number_input("Amount ($)", value=0.0, min_value=0.0, step=1.0)

    with st.expander("Advanced: V1–V28 PCA features (optional)"):
        v_values = {}
        cols = st.columns(4)
        for i, v in enumerate(V_COLS):
            with cols[i % 4]:
                v_values[v] = st.number_input(v, value=0.0, format="%.4f", key=v)

    if st.button("🔍 Check Transaction", type="primary"):
        row = {"Time": time_val, "Amount": amount_val, **v_values}
        df_input = pd.DataFrame([row])
        preds, probs = predict(df_input, threshold)
        pred, prob = preds[0], probs[0]

        st.divider()
        c1, c2 = st.columns([1, 2])
        with c1:
            if pred == 1:
                st.error("🚨 **FRAUD DETECTED**")
            else:
                st.success("✅ **Legitimate Transaction**")
            st.metric("Fraud Probability", f"{prob:.2%}")
        with c2:
            st.progress(min(max(prob, 0.0), 1.0))
            st.caption(f"Decision threshold: {threshold:.2f}")

# ---------------------------------------------------------------------------
# Page: Batch Prediction
# ---------------------------------------------------------------------------
elif page == "Batch Prediction (CSV)":
    st.title("Batch Prediction from CSV")
    st.caption(
        "Upload a CSV with columns `Time, V1, V2, ..., V28, Amount` "
        "(and optionally `Class` for evaluation)."
    )

    uploaded = st.file_uploader("Upload transactions CSV", type=["csv"])

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
            c1.metric("Total Transactions", len(df))
            c2.metric("Flagged as Fraud", n_fraud)
            c3.metric("Fraud Rate", f"{n_fraud / len(df):.2%}")

            st.subheader("Results")
            st.dataframe(
                results.sort_values("Fraud_Probability", ascending=False),
                use_container_width=True,
            )

            csv_out = results.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇️ Download results as CSV",
                data=csv_out,
                file_name="fraud_predictions.csv",
                mime="text/csv",
            )

            if has_labels:
                st.subheader("Evaluation Against Ground Truth")
                from sklearn.metrics import (
                    confusion_matrix, precision_score, recall_score, f1_score, ConfusionMatrixDisplay,
                )

                y_true = df["Class"].values
                p1, p2, p3 = st.columns(3)
                p1.metric("Precision", f"{precision_score(y_true, preds):.3f}")
                p2.metric("Recall", f"{recall_score(y_true, preds):.3f}")
                p3.metric("F1 Score", f"{f1_score(y_true, preds):.3f}")

                fig, ax = plt.subplots(figsize=(4, 4))
                ConfusionMatrixDisplay.from_predictions(
                    y_true, preds, display_labels=["Legit", "Fraud"], cmap="Blues", ax=ax
                )
                st.pyplot(fig)

# ---------------------------------------------------------------------------
# Page: About
# ---------------------------------------------------------------------------
else:
    st.title("About This Project")
    st.markdown(
        """
This app detects potentially fraudulent credit card transactions using a
model trained on the Kaggle **Credit Card Fraud Detection** dataset
(284,807 transactions, 492 confirmed frauds).

**Pipeline**
- `RobustScaler` applied to `Time` and `Amount` (the only non-PCA features)
- Class imbalance handled with **SMOTE**, applied only to training data
- Model trained and evaluated with a stratified, untouched test set
- Evaluated on Precision, Recall, F1, and ROC-AUC — accuracy alone is
  misleading on a dataset this imbalanced

**How to use**
- **Single Transaction**: manually enter transaction details
- **Batch Prediction**: upload a CSV of transactions and get predictions
  for all of them at once, plus evaluation metrics if you include the
  true `Class` labels

Adjust the **fraud decision threshold** in the sidebar to trade off
between catching more fraud (lower threshold) and reducing false alarms
(higher threshold).
        """
    )
