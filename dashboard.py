"""
Streamlit dashboard: live-stream transactions through the fraud model
with real-time KPIs, alerts, and charts. Or score a batch CSV upload.

Run:
    streamlit run dashboard.py
"""
import joblib
import pandas as pd
import streamlit as st
import time

from dataload import load_data
from train import preprocess

st.set_page_config(page_title="Fraud Detection Dashboard", layout="wide", page_icon="💳")

model = joblib.load("models/xgb_model.joblib")
threshold = joblib.load("models/best_threshold.joblib")
feature_columns = joblib.load("models/feature_columns.joblib")

st.markdown("""
<style>
.big-metric { font-size: 2.2rem !important; }
.fraud-alert {
    background-color: #ff4b4b22; border-left: 5px solid #ff4b4b;
    padding: 0.75rem 1rem; border-radius: 6px; margin-bottom: 0.5rem;
}
</style>
""", unsafe_allow_html=True)

st.title("💳 Real-Time Fraud Detection")
st.caption("XGBoost · trained on SMOTE-resampled transaction data · threshold = "
           f"{threshold:.4f}")

tab1, tab2 = st.tabs(["🔴 Live Stream Simulation", "📁 Batch CSV Scoring"])

with tab1:
    col_a, col_b, col_c = st.columns([2, 1, 1])
    with col_a:
        data_path = st.text_input(
            "kagglehub dataset path", value="",
            placeholder=r"C:\Users\ASUS\.cache\kagglehub\datasets\mlg-ulb\creditcardfraud\versions\3",
        )
    with col_b:
        n_txns = st.slider("Transactions to stream", 5, 2000, 50)
    with col_c:
        fast_mode = st.checkbox("Fast mode (no delay)", value=n_txns > 100)

    start = st.button("▶ Start streaming", type="primary")

    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpi_scanned = kpi1.empty()
    kpi_flagged = kpi2.empty()
    kpi_caught = kpi3.empty()
    kpi_rate = kpi4.empty()

    alert_box = st.container()
    chart_box = st.empty()
    table_box = st.empty()

    if start and data_path:
        df = load_data(data_path)
        X, y, _ = preprocess(df)
        sample = X.sample(n_txns, random_state=None)
        true_labels = y.loc[sample.index]

        results = []
        n_flagged = 0
        n_caught = 0
        delay = 0.0 if fast_mode else 0.15

        for idx, row in sample.iterrows():
            prob = float(model.predict_proba(row.to_frame().T[feature_columns])[:, 1][0])
            actual = int(true_labels.loc[idx])
            flagged = prob >= threshold
            n_flagged += int(flagged)
            n_caught += int(flagged and actual == 1)

            results.append({
                "txn_id": idx,
                "fraud_probability": round(prob, 4),
                "flagged": "🚩" if flagged else "",
                "actual_label": "FRAUD" if actual == 1 else "normal",
            })

            n_scanned = len(results)
            kpi_scanned.metric("Transactions scanned", n_scanned)
            kpi_flagged.metric("Flagged as fraud", n_flagged)
            kpi_caught.metric("True frauds caught", n_caught)
            precision_live = (n_caught / n_flagged * 100) if n_flagged else 0.0
            kpi_rate.metric("Live precision", f"{precision_live:.0f}%")

            if flagged and actual == 1:
                with alert_box:
                    st.markdown(
                        f'<div class="fraud-alert">🚨 <b>Fraud caught</b> — txn_id '
                        f'{idx}, probability {prob:.4f}</div>',
                        unsafe_allow_html=True,
                    )
            elif flagged and actual == 0:
                with alert_box:
                    st.markdown(
                        f'<div class="fraud-alert" style="border-color:#ffa500;'
                        f'background-color:#ffa50022;">⚠️ Flagged (false positive) '
                        f'— txn_id {idx}, probability {prob:.4f}</div>',
                        unsafe_allow_html=True,
                    )

            res_df = pd.DataFrame(results)
            chart_box.bar_chart(res_df.set_index("txn_id")["fraud_probability"])

            def highlight(row):
                if row["actual_label"] == "FRAUD":
                    return ["background-color: #ff4b4b33"] * len(row)
                if row["flagged"] == "🚩":
                    return ["background-color: #ffa50033"] * len(row)
                return [""] * len(row)

            table_box.dataframe(
                res_df.iloc[::-1].style.apply(highlight, axis=1),
                width="stretch",
                height=350,
            )

            if delay:
                time.sleep(delay)

        st.success(
            f"Done. Scanned {len(results)} transactions · "
            f"{n_flagged} flagged · {n_caught} confirmed fraud caught."
        )

with tab2:
    uploaded = st.file_uploader("Upload CSV with matching feature columns", type="csv")
    if uploaded:
        batch = pd.read_csv(uploaded)
        missing = set(feature_columns) - set(batch.columns)
        if missing:
            st.error(f"Missing columns: {missing}")
        else:
            probs = model.predict_proba(batch[feature_columns])[:, 1]
            batch["fraud_probability"] = probs
            batch["flagged"] = probs >= threshold

            c1, c2 = st.columns(2)
            c1.metric("Total transactions", len(batch))
            c2.metric("Flagged transactions", int(batch["flagged"].sum()))

            st.bar_chart(batch["fraud_probability"])
            st.dataframe(batch, width="stretch")