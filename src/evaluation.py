import os
import time
import json
import numpy as np
import pandas as pd

from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score,
    average_precision_score, confusion_matrix, balanced_accuracy_score
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import xgboost as xgb


def metric_row(y_true, scores, threshold=0.5, name="model"):
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores).astype(float)
    pred = (scores >= threshold).astype(int)

    return {
        "Model": name,
        "Precision": precision_score(y_true, pred, zero_division=0),
        "Recall": recall_score(y_true, pred, zero_division=0),
        "F1": f1_score(y_true, pred, zero_division=0),
        "ROC-AUC": roc_auc_score(y_true, scores),
        "PR-AUC": average_precision_score(y_true, scores),
        "Balanced-Accuracy": balanced_accuracy_score(y_true, pred),
        "TN": int(confusion_matrix(y_true, pred, labels=[0, 1])[0, 0]),
        "FP": int(confusion_matrix(y_true, pred, labels=[0, 1])[0, 1]),
        "FN": int(confusion_matrix(y_true, pred, labels=[0, 1])[1, 0]),
        "TP": int(confusion_matrix(y_true, pred, labels=[0, 1])[1, 1]),
    }


def best_f1_threshold(y_true, scores):
    from sklearn.metrics import precision_recall_curve
    p, r, t = precision_recall_curve(y_true, scores)
    if len(t) == 0:
        return 0.5
    f1 = 2 * p[:-1] * r[:-1] / (p[:-1] + r[:-1] + 1e-12)
    return float(t[np.argmax(f1)])


def evaluate_scores(y_true, scores, threshold=None, name="model"):
    threshold = (
        best_f1_threshold(y_true, scores)
        if threshold is None else threshold
    )
    row = metric_row(y_true, scores, threshold, name)
    row["Threshold"] = threshold
    return row


def evaluate_strong_baselines(X_train, y_train, X_test, y_test, X_val=None, y_val=None):
    """
    Same train/test split for all baselines.
    SMOTE is intentionally NOT applied here so the comparison is clean.
    Class weighting is used for supervised baselines.
    """
    models = {
        "Logistic Regression": make_pipeline(
            StandardScaler(),
            LogisticRegression(
                max_iter=1000, class_weight="balanced", random_state=42
            )
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=12,
            class_weight="balanced_subsample",
            random_state=42,
            n_jobs=1,
        ),
        "XGBoost": xgb.XGBClassifier(
            n_estimators=80,
            max_depth=4,
            learning_rate=0.05,
            scale_pos_weight=(
                (y_train == 0).sum() / max((y_train == 1).sum(), 1)
            ),
            eval_metric="aucpr",
            random_state=42,
            n_jobs=1,
        ),
    }

    rows = []
    for name, model in models.items():
        start = time.perf_counter()
        model.fit(X_train, y_train)
        val_scores = model.predict_proba(X_val)[:, 1] if X_val is not None else None
        threshold = best_f1_threshold(y_val, val_scores) if val_scores is not None else 0.5
        scores = model.predict_proba(X_test)[:, 1]
        elapsed = time.perf_counter() - start

        row = evaluate_scores(y_test, scores, threshold=threshold, name=name)
        row["TrainSeconds"] = elapsed
        rows.append(row)

    return pd.DataFrame(rows)


def run_ablation(y_true, score_map, threshold=0.5):
    """
    score_map keys should include:
      full, no_anomaly, no_temporal, no_graph, centralized
    """
    rows = []
    labels = {
        "full": "Full: XGB + IF + LSTM(FedAvg) + HGNN",
        "no_anomaly": "Ablation: - Isolation Forest",
        "no_temporal": "Ablation: - LSTM",
        "no_graph": "Ablation: - HGNN",
        "centralized": "Ablation: - Federated Training",
    }
    for key, label in labels.items():
        if key in score_map:
            rows.append(
                evaluate_scores(
                    y_true, score_map[key], threshold=(threshold.get(key, 0.5) if isinstance(threshold, dict) else threshold), name=label
                )
            )
    return pd.DataFrame(rows)


def save_results(df, path="results"):
    os.makedirs(path, exist_ok=True)
    csv_path = os.path.join(path, "evaluation_results.csv")
    json_path = os.path.join(path, "evaluation_results.json")
    df.to_csv(csv_path, index=False)
    df.to_json(json_path, orient="records", indent=2)
    print(f"[Evaluation] Saved {csv_path}")
    return csv_path
