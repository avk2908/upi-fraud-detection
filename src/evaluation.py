import os
import numpy as np
import pandas as pd

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix
)


def evaluate_scores(
    y_true,
    scores,
    threshold=0.5,
    model_name="Model"
):
    """
    Evaluate a fraud score against ground-truth labels.

    Parameters
    ----------
    y_true : array-like
        True fraud labels (0/1)

    scores : array-like
        Continuous fraud/risk scores in [0,1]

    threshold : float
        Threshold used to convert score into fraud/not-fraud

    model_name : str
        Name shown in evaluation table
    """

    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores).astype(float)

    predictions = (scores >= threshold).astype(int)

    precision = precision_score(
        y_true,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_true,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_true,
        predictions,
        zero_division=0
    )

    try:
        roc_auc = roc_auc_score(y_true, scores)
    except ValueError:
        roc_auc = np.nan

    try:
        pr_auc = average_precision_score(y_true, scores)
    except ValueError:
        pr_auc = np.nan

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1]
    ).ravel()

    return {
        "Configuration": model_name,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "ROC-AUC": roc_auc,
        "PR-AUC": pr_auc,
        "TP": tp,
        "FP": fp,
        "TN": tn,
        "FN": fn,
        "Threshold": threshold
    }


def evaluate_all(
    y_true,
    xgb_scores,
    iso_scores,
    lstm_scores,
    gnn_scores,
    final_scores,
    threshold=0.5
):
    """
    Evaluate all individual and combined layers.
    """

    results = []

    # ---------------------------------------------------------
    # 1. XGBoost
    # ---------------------------------------------------------
    results.append(
        evaluate_scores(
            y_true,
            xgb_scores,
            threshold,
            "XGBoost"
        )
    )

    # ---------------------------------------------------------
    # 2. Isolation Forest
    # ---------------------------------------------------------
    results.append(
        evaluate_scores(
            y_true,
            iso_scores,
            threshold,
            "Isolation Forest"
        )
    )

    # ---------------------------------------------------------
    # 3. XGBoost + Isolation Forest
    # ---------------------------------------------------------
    behavioral_scores = (
        0.70 * xgb_scores +
        0.30 * iso_scores
    )

    results.append(
        evaluate_scores(
            y_true,
            behavioral_scores,
            threshold,
            "XGBoost + Isolation Forest"
        )
    )

    # ---------------------------------------------------------
    # 4. XGBoost + IF + LSTM
    # ---------------------------------------------------------
    sequential_scores = (
        0.55 * xgb_scores +
        0.20 * iso_scores +
        0.25 * lstm_scores
    )

    results.append(
        evaluate_scores(
            y_true,
            sequential_scores,
            threshold,
            "XGBoost + IF + LSTM"
        )
    )

    # ---------------------------------------------------------
    # 5. Full Fusion
    # ---------------------------------------------------------
    results.append(
        evaluate_scores(
            y_true,
            final_scores,
            threshold,
            "Full Fusion"
        )
    )

    results_df = pd.DataFrame(results)

    # Round only for presentation
    metric_cols = [
        "Precision",
        "Recall",
        "F1",
        "ROC-AUC",
        "PR-AUC"
    ]

    results_df[metric_cols] = results_df[metric_cols].round(4)

    return results_df


def save_evaluation_table(
    results_df,
    output_dir="results"
):
    """
    Save evaluation results in CSV and Excel-compatible format.
    """

    os.makedirs(output_dir, exist_ok=True)

    csv_path = os.path.join(
        output_dir,
        "evaluation_results.csv"
    )

    results_df.to_csv(
        csv_path,
        index=False
    )

    print("\n==============================================")
    print("FINAL EVALUATION RESULTS")
    print("==============================================")

    print(
        results_df[
            [
                "Configuration",
                "Precision",
                "Recall",
                "F1",
                "ROC-AUC",
                "PR-AUC"
            ]
        ].to_string(index=False)
    )

    print("\nSaved to:")
    print(csv_path)

    return csv_path