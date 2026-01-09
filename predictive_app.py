# predictive_app.py
import time

import joblib
import pandas as pd
import streamlit as st

from log_utils import log_prediction

st.set_page_config(page_title="Revenue Prediction App with Monitoring",
                   layout="centered")

st.title("Revenue Prediction App with Live Monitoring")

@st.cache_resource
def load_models():
    old_model = joblib.load("revenue_model_v1.pkl")  # trained on ["units_sold"]
    new_model = joblib.load("revenue_model_v2.pkl")  # trained on ["units_sold", "region", "product"]
    return old_model, new_model

old_model, new_model = load_models()

# ---------- Initialise session state ----------
if "pred_ready" not in st.session_state:
    st.session_state["pred_ready"] = False
if "old_pred" not in st.session_state:
    st.session_state["old_pred"] = None
if "new_pred" not in st.session_state:
    st.session_state["new_pred"] = None
if "latency_ms" not in st.session_state:
    st.session_state["latency_ms"] = None
if "input_summary" not in st.session_state:
    st.session_state["input_summary"] = ""

# ---------- INPUT SECTION ----------
st.sidebar.header("Input Parameters")

units = st.sidebar.slider("Units Sold", min_value=1, max_value=200, value=50)
region = st.sidebar.selectbox("Region", ["North", "South", "East", "West"])
product = st.sidebar.selectbox("Product", ["Widget", "Gadget", "Tool", "Device"])

# Canonical input dataframe
input_df = pd.DataFrame({
    "units_sold": [units],
    "region": [region],
    "product": [product],
})

st.subheader("Input Summary")
st.write(input_df)

# ---------- BUTTON 1: RUN PREDICTION ----------
if st.button("Run Prediction"):
    start_time = time.time()

    # v1: baseline – only uses units_sold
    input_v1 = input_df[["units_sold"]]
    old_pred = old_model.predict(input_v1)[0]

    # v2: improved – uses all three features
    input_v2 = input_df[["units_sold", "region", "product"]]
    new_pred = new_model.predict(input_v2)[0]

    latency_ms = (time.time() - start_time) * 1000.0

    # Store in session_state so they survive reruns
    st.session_state["old_pred"] = float(old_pred)
    st.session_state["new_pred"] = float(new_pred)
    st.session_state["latency_ms"] = float(latency_ms)
    st.session_state["input_summary"] = f"units={units}, region={region}, product={product}"
    st.session_state["pred_ready"] = True

# ---------- SHOW PREDICTIONS IF READY ----------
if st.session_state["pred_ready"]:
    st.subheader("Predictions")
    st.write(f"Old Model (v1 - units only): **${st.session_state['old_pred']:,.2f}**")
    st.write(f"New Model (v2 - units + region + product): **${st.session_state['new_pred']:,.2f}**")
    st.write(f"Latency: {st.session_state['latency_ms']:.1f} ms")
else:
    st.info("Click **Run Prediction** to see model outputs before giving feedback.")

# ---------- FEEDBACK SECTION ----------
st.subheader("Your Feedback on These Predictions")

feedback_score = st.slider(
    "How useful were these predictions? (1 = Poor, 5 = Excellent)",
    min_value=1,
    max_value=5,
    value=4,
    key="feedback_score",
)
feedback_text = st.text_area("Comments (optional)", key="feedback_text")

# ---------- BUTTON 2: SUBMIT FEEDBACK ----------
if st.button("Submit Feedback"):
    if not st.session_state["pred_ready"]:
        st.warning("Please run the prediction first, then submit your feedback.")
    else:
        # Log both models using saved predictions and input summary
        log_prediction(
            model_version="v1_old",
            model_type="baseline",
            input_summary=st.session_state["input_summary"],
            prediction=st.session_state["old_pred"],
            latency_ms=st.session_state["latency_ms"],
            feedback_score=feedback_score,
            feedback_text=feedback_text,
        )

        log_prediction(
            model_version="v2_new",
            model_type="improved",
            input_summary=st.session_state["input_summary"],
            prediction=st.session_state["new_pred"],
            latency_ms=st.session_state["latency_ms"],
            feedback_score=feedback_score,
            feedback_text=feedback_text,
        )

        st.success(
            "Feedback and predictions have been saved to monitoring_logs.csv. "
            "You can now view them in the monitoring dashboard."
        )
