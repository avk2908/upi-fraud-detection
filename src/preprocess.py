import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import joblib
import os

def load_and_engineer(csv_path: str) -> pd.DataFrame:
    print("[Phase 1] Loading PaySim dataset...")
    df = pd.read_csv(csv_path, nrows=20000)

    # --- Basic cleaning ---
    df = df.drop(columns=["nameOrig", "nameDest", "isFlaggedFraud"], errors="ignore")
    df["type_enc"] = LabelEncoder().fit_transform(df["type"])

    # --- Balance error features (key fraud signal) ---
    df["orig_balance_error"] = (
        df["oldbalanceOrg"] - df["amount"] - df["newbalanceOrig"]
    ).abs()
    df["dest_balance_error"] = (
        df["oldbalanceDest"] + df["amount"] - df["newbalanceDest"]
    ).abs()
    df["zero_dest_after"] = (df["newbalanceDest"] == 0).astype(int)
    df["zero_orig_before"] = (df["oldbalanceOrg"] == 0).astype(int)

    # --- Reload with sender IDs for group features ---
    raw = pd.read_csv(csv_path, nrows=20000)
    df["nameOrig"] = raw["nameOrig"]
    df["nameDest"] = raw["nameDest"]

    print("[Phase 1] Engineering per-sender behavioral features...")
    df = df.sort_values(["nameOrig", "step"]).reset_index(drop=True)

    # Rolling velocity (last 3 txns per sender)
    df["sender_rolling_mean_amt"] = (
    df.groupby("nameOrig")["amount"]
    .transform("mean")
)
    df["amount_deviation"] = (
        df["amount"] - df["sender_rolling_mean_amt"]
    ).abs()

    # Transaction count per sender
    df["sender_txn_count"] = df.groupby("nameOrig")["step"].transform("count")

    # Inter-transaction gap
    df["prev_step"] = df.groupby("nameOrig")["step"].shift(1).fillna(0)
    df["txn_gap"] = df["step"] - df["prev_step"]

    # Receiver novelty: first time sender → receiver?
    df["pair"] = df["nameOrig"] + "_" + df["nameDest"]
    df["pair_count"] = df.groupby("pair")["step"].transform("count")
    df["receiver_novelty"] = (df["pair_count"] == 1).astype(int)

    # Beneficiary concentration (how often receiver appears globally)
    dest_freq = df["nameDest"].value_counts().to_dict()
    df["dest_frequency"] = df["nameDest"].map(dest_freq)

    # Drop string cols used only for engineering
    df = df.drop(columns=["nameOrig", "nameDest", "pair", "type", "prev_step"])

    print(f"[Phase 1] Done. Shape: {df.shape}, Fraud rate: {df['isFraud'].mean():.4f}")
    return df


def get_feature_cols():
    return [
        "step", "type_enc", "amount", "oldbalanceOrg", "newbalanceOrig",
        "oldbalanceDest", "newbalanceDest", "orig_balance_error",
        "dest_balance_error", "zero_dest_after", "zero_orig_before",
        "sender_rolling_mean_amt", "amount_deviation", "sender_txn_count",
        "txn_gap", "receiver_novelty", "dest_frequency", "pair_count"
    ]