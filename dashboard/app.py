import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
import joblib
import plotly.express as px
import plotly.graph_objects as go
st.set_page_config(
    page_title="UPI Fraud Detection Dashboard",
    page_icon="🔐",
    layout="wide"
)

@st.cache_resource
def load_models():
    xgb = joblib.load("models/xgb_model.pkl")
    iso = joblib.load("models/iso_forest.pkl")
    return xgb, iso

@st.cache_data
def load_sample_data(n=2000):
    import glob
    csv_files = glob.glob("data/*.csv")
    if not csv_files:
        st.error("No CSV found in data/. Download PaySim dataset.")
        return pd.DataFrame()
    from src.preprocess import load_and_engineer, get_feature_cols
    df = load_and_engineer(csv_files[0])
    return df.sample(n, random_state=42)

# ── Sidebar ────────────────────────────────────────────────────────────────
st.sidebar.title("🔐 UPI Fraud Shield")
st.sidebar.markdown("PaySim · Multi-Layer Detection")
threshold = st.sidebar.slider("Risk Threshold", 0.1, 0.9, 0.5, 0.05)
mode = st.sidebar.radio("View", ["Dashboard", "Explain Transaction", "Graph Analytics"])

# ── Load ───────────────────────────────────────────────────────────────────
df = load_sample_data()
if df.empty:
    st.stop()

try:
    xgb_model, iso_forest = load_models()
    from src.preprocess import get_feature_cols
    from src.behavioral_engine import get_behavioral_scores
    from src.risk_fusion import fuse_risk_scores

    FEATURES = get_feature_cols()
    X = df[FEATURES].fillna(0)
    xgb_scores, iso_scores = get_behavioral_scores(xgb_model, iso_forest, X)
    risk = fuse_risk_scores(xgb_scores, iso_scores, xgb_scores * 0.85, iso_scores * 0.5)
    df["risk_score"] = risk
    df["prediction"] = (risk > threshold).astype(int)
    models_loaded = True
except Exception as e:
    st.warning(f"Models not trained yet. Run train_pipeline.py first. ({e})")
    df["risk_score"] = np.random.uniform(0, 1, len(df))
    df["prediction"] = (df["risk_score"] > threshold).astype(int)
    models_loaded = False

# ── Dashboard View ─────────────────────────────────────────────────────────
if mode == "Dashboard":
    st.title("🔐 UPI Fraud Monitoring Dashboard")

    # KPI row
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Transactions", f"{len(df):,}")
    c2.metric("Flagged Fraud", f"{df['prediction'].sum():,}",
              delta=f"{df['prediction'].mean()*100:.1f}%")
    c3.metric("Avg Risk Score", f"{df['risk_score'].mean():.3f}")
    c4.metric("True Fraud Rate", f"{df['isFraud'].mean()*100:.2f}%")

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Risk Score Distribution")
        fig = px.histogram(df, x="risk_score", color="isFraud",
                           nbins=50, barmode="overlay",
                           color_discrete_map={0: "#2196F3", 1: "#F44336"},
                           labels={"isFraud": "Fraud"})
        fig.add_vline(x=threshold, line_dash="dash", line_color="orange",
                      annotation_text=f"Threshold: {threshold}")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Transaction Amount vs Risk")
        fig2 = px.scatter(df.sample(500), x="amount", y="risk_score",
                          color="isFraud",
                          color_discrete_map={0: "#2196F3", 1: "#F44336"},
                          opacity=0.6)
        st.plotly_chart(fig2, use_container_width=True)

    st.subheader("🚨 Top Flagged Transactions")
    flagged = df[df["prediction"] == 1].sort_values("risk_score", ascending=False)
    display_cols = ["step", "type_enc", "amount", "risk_score", "isFraud"]
    st.dataframe(
        flagged[display_cols].head(20).style
        .background_gradient(subset=["risk_score"], cmap="Reds")
        .format({"risk_score": "{:.3f}", "amount": "{:,.0f}"}),
        use_container_width=True
    )

    # Fraud by transaction type
    st.subheader("Fraud Rate by Transaction Type")
    type_stats = df.groupby("type_enc")["isFraud"].mean().reset_index()
    type_stats.columns = ["type", "fraud_rate"]
    fig3 = px.bar(type_stats, x="type", y="fraud_rate",
                  color="fraud_rate", color_continuous_scale="Reds")
    st.plotly_chart(fig3, use_container_width=True)

# ── Explain Transaction ────────────────────────────────────────────────────
elif mode == "Explain Transaction":
    st.title("🔍 Transaction Explainability (SHAP)")

    idx = st.number_input("Select transaction index (0–1999)", 0, len(df)-1, 0)
    row = df.iloc[[idx]]

    if models_loaded:
        from src.preprocess import get_feature_cols
        FEATURES = get_feature_cols()
        X_row = row[FEATURES].fillna(0)

        explainer = shap.TreeExplainer(xgb_model)
        sv = explainer.shap_values(X_row)

        risk_val = df.iloc[idx]["risk_score"]
        true_label = df.iloc[idx]["isFraud"]

        col1, col2, col3 = st.columns(3)
        col1.metric("Risk Score", f"{risk_val:.3f}")
        col2.metric("Prediction", "🚨 FRAUD" if risk_val > threshold else "✅ LEGIT")
        col3.metric("True Label", "FRAUD" if true_label else "LEGIT")

        st.subheader("Feature Contributions")
        contrib = pd.DataFrame({
            "Feature": FEATURES,
            "SHAP Value": sv[0],
            "Feature Value": X_row.values[0]
        }).sort_values("SHAP Value", key=abs, ascending=False).head(10)

        colors = ["#F44336" if v > 0 else "#2196F3" for v in contrib["SHAP Value"]]
        fig = go.Figure(go.Bar(
            x=contrib["SHAP Value"], y=contrib["Feature"],
            orientation="h", marker_color=colors
        ))
        fig.update_layout(title="Top SHAP Contributors (Red = Increases Fraud Risk)")
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(contrib)
    else:
        st.warning("Train models first with train_pipeline.py")

# ── Graph Analytics ────────────────────────────────────────────────────────
elif mode == "Graph Analytics":
    st.title("🕸️ Transaction Graph Analytics")

    try:
        import networkx as nx

        raw_path = [f for f in __import__("glob").glob("data/*.csv")][0]

        graph_n = min(len(df), 2000)

        raw = pd.read_csv(
            raw_path,
            usecols=["nameOrig", "nameDest"],
            nrows=graph_n
        )

        raw["isFraud"] = df["isFraud"].values[:graph_n]

        G = nx.DiGraph()

        for _, row in raw.iterrows():
            G.add_edge(
                row["nameOrig"],
                row["nameDest"],
                fraud=row["isFraud"]
            )

        fraud_nodes = set(
            raw[raw["isFraud"] == 1]["nameOrig"].tolist() +
            raw[raw["isFraud"] == 1]["nameDest"].tolist()
        )

        c1, c2, c3 = st.columns(3)

        c1.metric("Total Nodes", f"{G.number_of_nodes():,}")
        c2.metric("Total Edges", f"{G.number_of_edges():,}")
        c3.metric("Fraud-Involved Nodes", f"{len(fraud_nodes):,}")

        # Top suspicious receivers
        in_deg = pd.DataFrame(
            [(n, d) for n, d in G.in_degree()],
            columns=["node", "in_degree"]
        ).sort_values("in_degree", ascending=False).head(15)

        st.subheader("Top High In-Degree Receivers (Mule Candidates)")

        fig = px.bar(
            in_deg,
            x="node",
            y="in_degree",
            color="in_degree",
            color_continuous_scale="Reds"
        )

        fig.update_xaxes(tickangle=45)

        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"Graph analytics error: {e}")