import numpy as np
import pandas as pd

# Tunable weights (sum to 1.0)
W_BEHAVIORAL  = 0.40   # XGBoost score
W_ANOMALY     = 0.15   # Isolation Forest
W_SEQUENCE    = 0.25   # LSTM
W_RELATIONAL  = 0.20   # GNN / graph heuristic

def fuse_risk_scores(
    xgb_scores: np.ndarray,
    iso_scores: np.ndarray,
    lstm_scores: np.ndarray,
    gnn_scores: np.ndarray
) -> np.ndarray:
    """
    Weighted fusion of all module outputs → final risk score in [0, 1].
    R = w_b*B + w_a*A + w_s*S + w_r*G
    """
    def norm(arr):
        mn, mx = arr.min(), arr.max()
        return (arr - mn) / (mx - mn + 1e-9)

    B = norm(xgb_scores)
    A = norm(iso_scores)
    S = norm(lstm_scores)
    G = norm(gnn_scores)

    R = W_BEHAVIORAL * B + W_ANOMALY * A + W_SEQUENCE * S + W_RELATIONAL * G
    return R


def score_threshold_analysis(R: np.ndarray, y_true: np.ndarray):
    """Find optimal threshold by F1."""
    from sklearn.metrics import f1_score, precision_recall_curve
    precisions, recalls, thresholds = precision_recall_curve(y_true, R)
    f1s = 2 * precisions * recalls / (precisions + recalls + 1e-9)
    best_idx = np.argmax(f1s)
    best_thresh = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
    print(f"[Risk Fusion] Optimal threshold: {best_thresh:.3f} | F1: {f1s[best_idx]:.4f}")
    return best_thresh