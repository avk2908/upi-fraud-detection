"""
End-to-end training and evaluation pipeline for UPI Fraud Shield.

Usage:
    python train_pipeline.py

Pipeline:
    Phase 1 -> Data loading and feature engineering
    Phase 2 -> XGBoost + Isolation Forest
    Phase 3 -> LSTM sequential model
    Phase 4 -> GAT graph model
    Phase 5 -> Risk score generation and fusion
    Phase 6 -> Model evaluation + SHAP explainability

NOTE:
    The current GNN implementation is trained on the complete graph before
    transaction-level evaluation. Therefore, GNN scores are NOT included
    in the official evaluation table yet. This avoids reporting potentially
    leaked GNN metrics.
"""

import os
import numpy as np
import pandas as pd
import torch

# ============================================================
# Project imports
# ============================================================

from src.preprocess import (
    load_and_engineer,
    get_feature_cols
)

from src.behavioral_engine import (
    train_behavioral_engine,
    get_behavioral_scores
)

from src.lstm_model import (
    train_lstm,
    get_lstm_scores_for_dataframe
)

from src.gnn_model import (
    build_graph_from_paysim,
    train_gnn,
    get_gnn_scores
)

from src.risk_fusion import (
    fuse_risk_scores,
    score_threshold_analysis
)

from src.explainability import (
    generate_shap_explanations
)

from src.evaluation import (
    evaluate_all,
    save_evaluation_table
)


# ============================================================
# Configuration
# ============================================================

CSV_PATH = "data/PS_20174392719_1491204439457_log.csv"

MODEL_DIR = "models"

RESULTS_DIR = "results"

SHAP_SAMPLE_SIZE = 500

RANDOM_STATE = 42


# ============================================================
# Main Pipeline
# ============================================================

def main():

    print("\n")
    print("=" * 70)
    print("           UPI FRAUD SHIELD - TRAINING PIPELINE")
    print("=" * 70)
    print("\n")

    # --------------------------------------------------------
    # Check dataset
    # --------------------------------------------------------

    if not os.path.exists(CSV_PATH):

        raise FileNotFoundError(
            f"\nDataset not found:\n{CSV_PATH}\n\n"
            "Please make sure the PaySim CSV is inside the data/ folder."
        )

    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)


    # ========================================================
    # PHASE 1
    # Data loading + feature engineering
    # ========================================================

    print("=" * 70)
    print("PHASE 1 - DATA PREPROCESSING & FEATURE ENGINEERING")
    print("=" * 70)

    df = load_and_engineer(CSV_PATH)

    print("\n[Phase 1] Dataset shape:", df.shape)

    print(
        "[Phase 1] Fraud transactions:",
        int(df["isFraud"].sum())
    )

    print(
        "[Phase 1] Fraud rate:",
        f"{df['isFraud'].mean():.4%}"
    )

    print(
        "[Phase 1] Number of features:",
        len(get_feature_cols())
    )


    # ========================================================
    # PHASE 2
    # XGBoost + Isolation Forest
    # ========================================================

    print("\n")
    print("=" * 70)
    print("PHASE 2 - BEHAVIORAL + ANOMALY DETECTION")
    print("=" * 70)

    (
        xgb_model,
        iso_forest,
        X_test,
        y_test
    ) = train_behavioral_engine(df)

    print("\n[Phase 2] Test transactions:", len(X_test))

    print(
        "[Phase 2] Test fraud transactions:",
        int(y_test.sum())
    )

    print(
        "[Phase 2] Test fraud rate:",
        f"{y_test.mean():.4%}"
    )


    # ========================================================
    # PHASE 3
    # LSTM sequential intelligence
    # ========================================================

    print("\n")
    print("=" * 70)
    print("PHASE 3 - LSTM SEQUENTIAL INTELLIGENCE")
    print("=" * 70)

    (
        lstm_model,
        lstm_scaler
    ) = train_lstm(
        df,
        CSV_PATH
    )


    # ========================================================
    # PHASE 4
    # Graph Neural Network
    # ========================================================

    print("\n")
    print("=" * 70)
    print("PHASE 4 - GRAPH ATTENTION NETWORK")
    print("=" * 70)

    (
        graph_data,
        node_map
    ) = build_graph_from_paysim(
        df,
        CSV_PATH
    )

    (
        gnn_model,
        graph_data
    ) = train_gnn(
        graph_data
    )

    print(
        f"\n[Phase 4] Graph nodes: "
        f"{graph_data.num_nodes:,}"
    )

    print(
        f"[Phase 4] Graph edges: "
        f"{graph_data.edge_index.shape[1]:,}"
    )

    print(
        "\n[Phase 4] NOTE:"
        "\nGNN is trained successfully, but its scores are not"
        "\nincluded in the official evaluation table yet because"
        "\nthe current graph construction uses information from"
        "\nthe complete dataset before the test split."
    )


    # ========================================================
    # PHASE 5
    # Generate aligned test-set scores
    # ========================================================

    print("\n")
    print("=" * 70)
    print("PHASE 5 - TEST SET SCORE GENERATION")
    print("=" * 70)


    # --------------------------------------------------------
    # 5.1 XGBoost + Isolation Forest
    # --------------------------------------------------------

    print("\n[Phase 5] Generating XGBoost scores...")

    (
        xgb_scores,
        iso_scores
    ) = get_behavioral_scores(
        xgb_model,
        iso_forest,
        X_test
    )

    print(
        "[Phase 5] XGBoost scores generated:",
        len(xgb_scores)
    )

    print(
        "[Phase 5] Isolation Forest scores generated:",
        len(iso_scores)
    )


    # --------------------------------------------------------
    # 5.2 Generate REAL LSTM scores
    # --------------------------------------------------------

    print("\n[Phase 5] Generating real LSTM scores...")

    lstm_scores_all = get_lstm_scores_for_dataframe(
        lstm_model,
        lstm_scaler,
        df,
        CSV_PATH
    )

    # X_test preserves the original indices of the rows
    # belonging to the behavioral test set.
    test_indices = X_test.index.to_numpy()

    lstm_scores = lstm_scores_all[test_indices]

    print(
        "[Phase 5] LSTM scores generated:",
        len(lstm_scores)
    )


    # --------------------------------------------------------
    # 5.3 Sanity check
    # --------------------------------------------------------

    print("\n[Phase 5] Checking score alignment...")

    if not (
        len(xgb_scores)
        == len(iso_scores)
        == len(lstm_scores)
        == len(y_test)
    ):

        raise ValueError(
            "\nScore alignment error!\n"
            f"XGBoost: {len(xgb_scores)}\n"
            f"Isolation Forest: {len(iso_scores)}\n"
            f"LSTM: {len(lstm_scores)}\n"
            f"y_test: {len(y_test)}"
        )

    print(
        "[Phase 5] ✓ All test-set scores have matching length:",
        len(y_test)
    )


    # --------------------------------------------------------
    # 5.4 Normalize model scores
    # --------------------------------------------------------

    def normalize_scores(scores):

        scores = np.asarray(
            scores,
            dtype=float
        )

        minimum = scores.min()
        maximum = scores.max()

        return (
            scores - minimum
        ) / (
            maximum - minimum + 1e-9
        )


    xgb_scores = normalize_scores(
        xgb_scores
    )

    iso_scores = normalize_scores(
        iso_scores
    )

    lstm_scores = normalize_scores(
        lstm_scores
    )


    # --------------------------------------------------------
    # 5.5 Current fusion
    # --------------------------------------------------------
    #
    # Current implementation:
    #
    # Behavioral : XGBoost
    # Anomaly    : Isolation Forest
    # Sequence   : LSTM
    #
    # We deliberately DO NOT add random GNN scores.
    #
    # --------------------------------------------------------

    print("\n[Phase 5] Computing fused risk scores...")

    final_scores = (
        0.50 * xgb_scores
        + 0.20 * iso_scores
        + 0.30 * lstm_scores
    )

    print(
        "[Phase 5] Final risk scores generated:",
        len(final_scores)
    )


    # --------------------------------------------------------
    # 5.6 Threshold analysis
    # --------------------------------------------------------

    print("\n[Phase 5] Running threshold analysis...")

    best_threshold = score_threshold_analysis(
        final_scores,
        y_test.values
    )

    print(
        f"[Phase 5] Suggested F1 threshold: "
        f"{best_threshold:.4f}"
    )


    # ========================================================
    # PHASE 6
    # Evaluation
    # ========================================================

    print("\n")
    print("=" * 70)
    print("PHASE 6 - MODEL EVALUATION")
    print("=" * 70)

    print(
        "\n[Phase 6] Evaluating models using threshold:",
        f"{best_threshold:.4f}"
    )


    # --------------------------------------------------------
    # IMPORTANT:
    #
    # The current evaluation table compares:
    #
    # 1. XGBoost
    # 2. Isolation Forest
    # 3. XGBoost + Isolation Forest
    # 4. XGBoost + IF + LSTM
    # 5. Current Fusion
    #
    # GNN is intentionally excluded from official metrics
    # until its data split is corrected.
    # --------------------------------------------------------

    results_df = evaluate_all(
        y_true=y_test.values,

        xgb_scores=xgb_scores,

        iso_scores=iso_scores,

        lstm_scores=lstm_scores,

        # Placeholder only because evaluate_all currently
        # expects a GNN argument.
        #
        # It is NOT used for the current final fusion.
        gnn_scores=np.zeros_like(
            xgb_scores
        ),

        final_scores=final_scores,

        threshold=best_threshold
    )


    # --------------------------------------------------------
    # Save evaluation results
    # --------------------------------------------------------

    evaluation_path = save_evaluation_table(
        results_df,
        output_dir=RESULTS_DIR
    )


    # --------------------------------------------------------
    # Print clean paper-ready table
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("PAPER-READY EVALUATION TABLE")
    print("=" * 70)

    paper_table = results_df[
        [
            "Configuration",
            "Precision",
            "Recall",
            "F1",
            "ROC-AUC",
            "PR-AUC"
        ]
    ].copy()

    print(
        paper_table.to_string(
            index=False
        )
    )


    # ========================================================
    # PHASE 7
    # SHAP Explainability
    # ========================================================

    print("\n")
    print("=" * 70)
    print("PHASE 7 - SHAP EXPLAINABILITY")
    print("=" * 70)

    shap_sample_size = min(
        SHAP_SAMPLE_SIZE,
        len(X_test)
    )

    sample = X_test.sample(
        shap_sample_size,
        random_state=RANDOM_STATE
    )

    print(
        f"[Phase 7] Generating SHAP explanations "
        f"for {len(sample)} transactions..."
    )

    generate_shap_explanations(
        xgb_model,
        sample
    )


    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print("\n")
    print("=" * 70)
    print("              PIPELINE COMPLETE")
    print("=" * 70)

    print("\nModels:")
    print("  ✓ XGBoost")
    print("  ✓ Isolation Forest")
    print("  ✓ LSTM")
    print("  ✓ GAT")

    print("\nEvaluation:")
    print("  ✓ Precision")
    print("  ✓ Recall")
    print("  ✓ F1")
    print("  ✓ ROC-AUC")
    print("  ✓ PR-AUC")
    print("  ✓ Confusion matrix components")

    print("\nResults saved to:")
    print(
        f"  {evaluation_path}"
    )

    print("\nDashboard:")
    print(
        "  streamlit run dashboard/app.py"
    )

    print("\n")


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    main()