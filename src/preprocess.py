import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


FEATURES = [
    "step", "type_enc", "amount", "oldbalanceOrg", "newbalanceOrig",
    "oldbalanceDest", "newbalanceDest", "orig_balance_error",
    "dest_balance_error", "zero_dest_after", "zero_orig_before",
    "sender_rolling_mean_amt", "amount_deviation", "sender_txn_count",
    "txn_gap", "receiver_novelty", "dest_frequency", "pair_count"
]


def load_and_engineer(csv_path: str, nrows: int = 20000) -> pd.DataFrame:
    """
    Loads PaySim and preserves txn_id so every downstream model can map
    its output back to the exact transaction row.
    """
    print("[Phase 1] Loading PaySim...")
    raw = pd.read_csv(csv_path, nrows=nrows)
    raw["txn_id"] = np.arange(len(raw), dtype=np.int64)

    df = raw.copy()

    df = df.drop(columns=["nameOrig", "nameDest", "isFlaggedFraud"], errors="ignore")
    df["type_enc"] = LabelEncoder().fit_transform(df["type"].astype(str))

    df["orig_balance_error"] = (
        df["oldbalanceOrg"] - df["amount"] - df["newbalanceOrig"]
    ).abs()
    df["dest_balance_error"] = (
        df["oldbalanceDest"] + df["amount"] - df["newbalanceDest"]
    ).abs()
    df["zero_dest_after"] = (df["newbalanceDest"] == 0).astype(int)
    df["zero_orig_before"] = (df["oldbalanceOrg"] == 0).astype(int)

    # Keep identifiers only for feature engineering / alignment.
    df["nameOrig"] = raw["nameOrig"].values
    df["nameDest"] = raw["nameDest"].values

    df = df.sort_values(["nameOrig", "step", "txn_id"]).reset_index(drop=True)

    # Strictly past-only sender statistics. Same-step transactions are not
    # allowed to contribute to the current transaction's history.
    df["sender_rolling_mean_amt"] = (
        df.groupby("nameOrig")["amount"].transform(lambda s: s.shift(1).expanding().mean())
    ).fillna(0.0)
    df["amount_deviation"] = (
        df["amount"] - df["sender_rolling_mean_amt"]
    ).abs()
    df["sender_txn_count"] = df.groupby("nameOrig").cumcount().astype(float)
    df["prev_step"] = df.groupby("nameOrig")["step"].shift(1)
    df["txn_gap"] = (df["step"] - df["prev_step"].fillna(df["step"])).clip(lower=0)

    df["pair"] = df["nameOrig"].astype(str) + "_" + df["nameDest"].astype(str)
    df["pair_count"] = df.groupby("pair").cumcount().astype(float)
    df["receiver_novelty"] = (df["pair_count"] == 0).astype(int)

    receiver_history = (
        df.sort_values(["step", "txn_id"])
        .groupby("nameDest", sort=False)
        .cumcount()
    )
    df["dest_frequency"] = pd.Series(receiver_history.to_numpy(), index=df.sort_values(["step", "txn_id"]).index).reindex(df.index).fillna(0).astype(float)

    print(
        f"[Phase 1] Done. rows={len(df):,}, features={len(FEATURES)}, "
        f"fraud_rate={df['isFraud'].mean():.6f}"
    )
    return df


def get_feature_cols():
    return FEATURES.copy()


def get_model_matrix(df: pd.DataFrame) -> pd.DataFrame:
    return df[FEATURES].replace([np.inf, -np.inf], np.nan).fillna(0)
