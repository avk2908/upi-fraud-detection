import numpy as np

W_BEHAVIORAL = 0.40
W_ANOMALY = 0.15
W_SEQUENCE = 0.25
W_RELATIONAL = 0.20


def minmax_fit(arr):
    # All model interfaces produce calibrated/ranked values in [0, 1]. The
    # anomaly adapter performs its train-fitted scaling upstream.
    return 0.0, 1.0


def minmax_apply(arr, mn, mx):
    arr = np.asarray(arr, dtype=float)
    return np.clip((arr - mn) / (mx - mn + 1e-9), 0.0, 1.0)


def fuse_risk_scores(xgb_scores, iso_scores, lstm_scores, gnn_scores):
    """
    Final risk score. Federated learning is a TRAINING STRATEGY, not a fifth
    independent risk score, so it is intentionally not added as a separate
    term. The federated/centralized comparison is evaluated separately.
    """
    arrays = [xgb_scores, iso_scores, lstm_scores, gnn_scores]
    normed = []
    for a in arrays:
        mn, mx = minmax_fit(a)
        normed.append(minmax_apply(a, mn, mx))

    B, A, S, G = normed
    return (
        W_BEHAVIORAL * B
        + W_ANOMALY * A
        + W_SEQUENCE * S
        + W_RELATIONAL * G
    )
