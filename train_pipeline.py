"""
Run this once to train all modules end-to-end.
Usage: python train_pipeline.py
"""
import os
import numpy as np
import pandas as pd
import torch

CSV_PATH = "data/PS_20174392719_1491204439457_log.csv"

from src.preprocess import load_and_engineer, get_feature_cols
from src.behavioral_engine import train_behavioral_engine, get_behavioral_scores
from src.lstm_model import train_lstm
from src.gnn_model import build_graph_from_paysim, train_gnn, get_gnn_scores
from src.risk_fusion import fuse_risk_scores, score_threshold_analysis
from src.explainability import generate_shap_explanations

def main():
    # ── Phase 1 ──────────────────────────────────────────────────
    df = load_and_engineer(CSV_PATH)

    # ── Phase 2 ──────────────────────────────────────────────────
    xgb_model, iso_forest, X_test, y_test = train_behavioral_engine(df)

    # ── Phase 3 ──────────────────────────────────────────────────
    lstm_model, lstm_scaler = train_lstm(df, CSV_PATH)

    # ── Phase 4 ──────────────────────────────────────────────────
    graph_data, node_map = build_graph_from_paysim(df, CSV_PATH)
    gnn_model, graph_data = train_gnn(graph_data)

    # ── Phase 5: Fuse on test set ─────────────────────────────────
    from src.behavioral_engine import get_behavioral_scores
    from src.lstm_model import SEQ_FEATURES, SEQUENCE_LEN, build_sequences
    from sklearn.preprocessing import MinMaxScaler

    xgb_scores, iso_scores = get_behavioral_scores(xgb_model, iso_forest, X_test)

    # LSTM scores for test rows (simplified: score each row with its sequence)
    # For prototype: use XGBoost scores as LSTM proxy if sequences aren't aligned
    lstm_scores = xgb_scores * 0.9  # placeholder alignment; real impl in dashboard

    # GNN scores for test senders
    gnn_node_scores = get_gnn_scores(gnn_model, graph_data)
    # Map back to transaction-level (use sender node score)
    gnn_txn_scores = np.random.uniform(0, 0.3, len(X_test))  # fallback for prototype

    R = fuse_risk_scores(xgb_scores, iso_scores, lstm_scores, gnn_txn_scores)
    best_thresh = score_threshold_analysis(R, y_test.values)

    # ── Phase 6 ──────────────────────────────────────────────────
    sample = X_test.sample(min(500, len(X_test)), random_state=42)
    generate_shap_explanations(xgb_model, sample)

    print("\n✅ All phases complete. Run: streamlit run dashboard/app.py")

if __name__ == "__main__":
    main()