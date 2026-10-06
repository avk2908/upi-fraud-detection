import numpy as np
import pandas as pd


def temporal_split(df, train_frac=0.70, val_frac=0.15):
    """
    Chronological split by PaySim 'step'. No future rows enter training.
    """
    order = np.lexsort((df["txn_id"].to_numpy(), df["step"].to_numpy()))
    train_end = int(len(df) * train_frac)
    val_end = int(len(df) * (train_frac + val_frac))
    train = df.iloc[order[:train_end]].copy()
    val = df.iloc[order[train_end:val_end]].copy()
    test = df.iloc[order[val_end:]].copy()
    return train, val, test


def unseen_attack_proxy(y_true, scores, high_risk_mask, threshold=0.5):
    """
    A dataset-only robustness proxy. PaySim does not provide attack-family
    labels, so this is NOT a true unseen-attack benchmark.

    It reports recall on a user-defined subset such as:
      - high-value transactions
      - new beneficiary transactions
      - extreme amount-deviation transactions

    Use this only as a stress test, not as proof of adaptive-attack
    robustness.
    """
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores)
    mask = np.asarray(high_risk_mask).astype(bool)

    if mask.sum() == 0:
        return {"subset_size": 0, "recall": np.nan}

    pred = (scores[mask] >= threshold).astype(int)
    truth = y_true[mask]
    positives = truth.sum()

    return {
        "subset_size": int(mask.sum()),
        "positive_count": int(positives),
        "recall": float((pred[truth == 1].sum() / positives)
                        if positives else np.nan)
    }


def robustness_report(y_true, scores, df, threshold=0.5):
    """
    Produces a compact stress-test report for:
    - high-value transactions
    - novel beneficiaries
    - extreme amount deviation

    These are proxies for distribution shift, not attacker simulations.
    """
    q99 = df["amount_deviation"].quantile(0.99)
    masks = {
        "high_value": df["amount"] >= df["amount"].quantile(0.99),
        "novel_beneficiary": df["receiver_novelty"] == 1,
        "extreme_amount_deviation": df["amount_deviation"] >= q99,
    }

    from sklearn.metrics import precision_score, recall_score, f1_score, average_precision_score
    rows = []
    for name, mask in masks.items():
        selected = mask.to_numpy(dtype=bool)
        truth = np.asarray(y_true).astype(int)[selected]
        pred_scores = np.asarray(scores)[selected]
        pred = (pred_scores >= threshold).astype(int)
        result = {
            "subset_size": int(selected.sum()),
            "positive_count": int(truth.sum()),
            "precision": float(precision_score(truth, pred, zero_division=0)) if len(truth) else np.nan,
            "recall": float(recall_score(truth, pred, zero_division=0)) if len(truth) else np.nan,
            "f1": float(f1_score(truth, pred, zero_division=0)) if len(truth) else np.nan,
            "pr_auc": float(average_precision_score(truth, pred_scores)) if len(np.unique(truth)) > 1 else np.nan,
        }
        result["Scenario"] = name
        rows.append(result)

    return pd.DataFrame(rows)
