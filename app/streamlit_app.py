"""Streamlit front-end for the fraud detection model.

Run with:  streamlit run app/streamlit_app.py

Tabs:
- Single Prediction: score one transaction (sample a real one or edit values)
- Batch Prediction: upload a CSV of transactions, score them all, download
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.inference import RAW_INPUT_COLUMNS, FraudDetector  # noqa: E402
from src.utils import load_config, resolve_path  # noqa: E402

st.set_page_config(page_title="Fraud Detection", page_icon="💳", layout="wide")

RISK_COLORS = {"LOW": "#2e7d32", "MEDIUM": "#f9a825", "HIGH": "#c62828"}


@st.cache_resource
def get_detector() -> FraudDetector:
    """Load the trained model once per server process."""
    return FraudDetector.load()


@st.cache_data
def get_sample_pool() -> pd.DataFrame | None:
    """Load a pool of real transactions for the demo sampler, if available."""
    config = load_config()
    raw = resolve_path(config["paths"]["raw_data"])
    if not raw.exists():
        return None
    df = pd.read_csv(raw)
    # keep all frauds + 2000 genuine so both buttons always have material
    pool = pd.concat([
        df[df["Class"] == 1],
        df[df["Class"] == 0].sample(2000, random_state=7),
    ])
    return pool.reset_index(drop=True)


def render_result(result: dict) -> None:
    """Show probability, class, and risk level for one scored transaction."""
    proba = result["fraud_probability"]
    risk = result["risk_level"]
    verdict = "🚨 FRAUD" if result["prediction"] == 1 else "✅ GENUINE"

    c1, c2, c3 = st.columns(3)
    c1.metric("Fraud probability", f"{proba:.2%}")
    c2.metric("Decision", verdict,
              help=f"Decision threshold = {result['threshold']:.3f}")
    c3.markdown(
        f"<div style='text-align:center;padding:0.6em;border-radius:8px;"
        f"background:{RISK_COLORS[risk]};color:white;font-size:1.4em;"
        f"font-weight:bold'>{risk} RISK</div>", unsafe_allow_html=True)
    st.progress(min(max(proba, 0.0), 1.0))


def single_prediction_tab(detector: FraudDetector) -> None:
    st.subheader("Score a single transaction")
    pool = get_sample_pool()

    if "txn" not in st.session_state:
        st.session_state.txn = {c: 0.0 for c in RAW_INPUT_COLUMNS}

    if pool is not None:
        b1, b2, _ = st.columns([1, 1, 2])
        if b1.button("🎲 Load random GENUINE transaction"):
            row = pool[pool["Class"] == 0].sample(1).iloc[0]
            st.session_state.txn = {c: float(row[c]) for c in RAW_INPUT_COLUMNS}
        if b2.button("🎯 Load random FRAUD transaction"):
            row = pool[pool["Class"] == 1].sample(1).iloc[0]
            st.session_state.txn = {c: float(row[c]) for c in RAW_INPUT_COLUMNS}
    else:
        st.info("Raw dataset not found - enter values manually.")

    with st.expander("Transaction values (editable)", expanded=False):
        cols = st.columns(4)
        for i, name in enumerate(RAW_INPUT_COLUMNS):
            st.session_state.txn[name] = cols[i % 4].number_input(
                name, value=float(st.session_state.txn[name]), format="%.6f",
                key=f"in_{name}")

    if st.button("🔍 Score transaction", type="primary"):
        try:
            result = detector.predict_one(st.session_state.txn)
        except ValueError as exc:
            st.error(f"Invalid input: {exc}")
            return
        render_result(result)


def batch_prediction_tab(detector: FraudDetector) -> None:
    st.subheader("Score a batch of transactions")
    st.caption("Upload a CSV with columns: Time, V1-V28, Amount "
               "(a Class column, if present, is ignored for scoring).")
    upload = st.file_uploader("Transactions CSV", type="csv")
    if upload is None:
        return
    try:
        df = pd.read_csv(upload)
        scored = detector.predict_batch(df)
    except ValueError as exc:
        st.error(f"Could not score file: {exc}")
        return

    n_flag = int(scored["prediction"].sum())
    c1, c2, c3 = st.columns(3)
    c1.metric("Transactions", f"{len(scored):,}")
    c2.metric("Flagged as fraud", f"{n_flag:,}")
    c3.metric("Flag rate", f"{n_flag / len(scored):.2%}")

    st.bar_chart(scored["risk_level"].value_counts())

    st.markdown("**Probability distribution** (log-scaled counts)")
    hist = np.histogram(scored["fraud_probability"], bins=20, range=(0, 1))[0]
    st.bar_chart(pd.DataFrame({"count": hist},
                              index=[f"{i / 20:.2f}" for i in range(20)]))

    st.dataframe(
        scored.sort_values("fraud_probability", ascending=False).head(200),
        use_container_width=True)
    st.download_button("⬇️ Download scored CSV",
                       scored.to_csv(index=False).encode(),
                       file_name="scored_transactions.csv", mime="text/csv")


def main() -> None:
    st.title("💳 Credit Card Fraud Detection")
    try:
        detector = get_detector()
    except FileNotFoundError as exc:
        st.error(str(exc))
        st.stop()
        return

    meta = detector.metadata
    with st.sidebar:
        st.header("Model card")
        st.write(f"**Model:** {meta.get('model')}")
        st.write(f"**Imbalance strategy:** {meta.get('strategy')}")
        st.write(f"**Decision threshold:** {detector.threshold:.3f} "
                 "(business-cost optimized)")
        tm = meta.get("test_metrics_tuned", {})
        if tm:
            st.divider()
            st.header("Held-out test metrics")
            st.write(f"Precision: **{tm['precision']:.3f}**")
            st.write(f"Recall: **{tm['recall']:.3f}**")
            st.write(f"F1: **{tm['f1']:.3f}**")
            st.write(f"PR-AUC: **{tm['pr_auc']:.3f}**")
            st.write(f"ROC-AUC: **{tm['roc_auc']:.3f}**")
        bc = meta.get("business_cost", {})
        if bc:
            st.divider()
            st.header("Business impact")
            st.write(f"Fraud loss without model: **${bc['no_model_cost']:,.0f}**")
            st.write(f"Cost with model: **${bc['tuned_cost']:,.0f}**")
            st.write(f"Savings: **{bc['savings_vs_no_model_pct']}%**")

    tab1, tab2 = st.tabs(["🔍 Single Prediction", "📦 Batch Prediction"])
    with tab1:
        single_prediction_tab(detector)
    with tab2:
        batch_prediction_tab(detector)


if __name__ == "__main__":
    main()
