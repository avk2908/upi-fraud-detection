import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import MinMaxScaler
import joblib
import os

SEQUENCE_LEN = 5   # last 5 transactions per sender
SEQ_FEATURES = [
    "amount", "type_enc", "orig_balance_error", "dest_balance_error",
    "amount_deviation", "txn_gap"
]

def build_sequences(df: pd.DataFrame, raw_csv: str):
    """Build per-sender transaction sequences of length SEQUENCE_LEN."""
    raw = pd.read_csv(raw_csv, usecols=["nameOrig"], nrows=len(df))
    df = df.copy()
    df["nameOrig"] = raw["nameOrig"].values

    sequences, labels = [], []
    grouped = df.groupby("nameOrig")

    for _, grp in grouped:
        grp = grp.sort_values("step")
        feat = grp[SEQ_FEATURES].fillna(0).values
        lbl = grp["isFraud"].values

        # Pad short senders
        if len(feat) < SEQUENCE_LEN:
            pad = np.zeros((SEQUENCE_LEN - len(feat), len(SEQ_FEATURES)))
            feat = np.vstack([pad, feat])
            lbl_pad = np.zeros(SEQUENCE_LEN - len(lbl))
            lbl = np.concatenate([lbl_pad, lbl])

        for i in range(SEQUENCE_LEN, len(feat) + 1):
            sequences.append(feat[i - SEQUENCE_LEN:i])
            labels.append(lbl[i - 1])

    return np.array(sequences, dtype=np.float32), np.array(labels, dtype=np.float32)


def train_lstm(df: pd.DataFrame, raw_csv: str, model_dir: str = "models"):
    os.makedirs(model_dir, exist_ok=True)
    print("[Phase 3] Building LSTM sequences...")
    X_seq, y_seq = build_sequences(df, raw_csv)

    # Scale
    n, t, f = X_seq.shape
    X_flat = X_seq.reshape(-1, f)
    scaler = MinMaxScaler()
    X_flat = scaler.fit_transform(X_flat)
    X_seq = X_flat.reshape(n, t, f)
    joblib.dump(scaler, f"{model_dir}/lstm_scaler.pkl")

    # Split
    split = int(0.8 * n)
    X_tr, X_te = X_seq[:split], X_seq[split:]
    y_tr, y_te = y_seq[:split], y_seq[split:]

    # Class weight
    pos = y_tr.sum()
    neg = len(y_tr) - pos
    cw = {0: 1.0, 1: neg / (pos + 1e-9)}

    print(f"[Phase 3] Sequences: {n}, Fraud ratio: {y_seq.mean():.4f}")
    print("[Phase 3] Training LSTM...")

    model = Sequential([
        LSTM(64, input_shape=(SEQUENCE_LEN, f), return_sequences=True),
        Dropout(0.3),
        BatchNormalization(),
        LSTM(32),
        Dropout(0.3),
        Dense(16, activation="relu"),
        Dense(1, activation="sigmoid")
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="binary_crossentropy",
        metrics=["AUC"]
    )

    callbacks = [
        EarlyStopping(patience=5, restore_best_weights=True, monitor="val_AUC", mode="max"),
        ReduceLROnPlateau(patience=3, factor=0.5)
    ]
    model.fit(
        X_tr, y_tr,
        validation_data=(X_te, y_te),
        epochs=30,
        batch_size=512,
        class_weight=cw,
        callbacks=callbacks,
        verbose=1
    )
    model.save(f"{model_dir}/lstm_model.h5")
    print("[Phase 3] LSTM saved.")
    return model, scaler


def load_lstm(model_dir: str = "models"):
    model = load_model(f"{model_dir}/lstm_model.h5")
    scaler = joblib.load(f"{model_dir}/lstm_scaler.pkl")
    return model, scaler

def get_lstm_scores_for_dataframe(
    model,
    scaler,
    df: pd.DataFrame,
    raw_csv: str
):
    """
    Generate one LSTM fraud score for every transaction.

    The score for transaction i is based on the previous
    SEQUENCE_LEN transactions of the same sender.
    """

    raw = pd.read_csv(
        raw_csv,
        usecols=["nameOrig"],
        nrows=len(df)
    )

    work = df.copy()
    work["nameOrig"] = raw["nameOrig"].values

    # Preserve original transaction position
    work["_original_index"] = np.arange(len(work))

    # Sort by sender and time
    work = work.sort_values(
        ["nameOrig", "step"]
    )

    sequences = []
    indices = []

    for sender, grp in work.groupby("nameOrig"):

        grp = grp.sort_values("step")

        features = (
            grp[SEQ_FEATURES]
            .fillna(0)
            .values
        )

        original_indices = (
            grp["_original_index"]
            .values
        )

        for i in range(len(grp)):

            # Build history ending at current transaction
            start = max(0, i - SEQUENCE_LEN + 1)

            seq = features[start:i + 1]

            # Pad beginning if necessary
            if len(seq) < SEQUENCE_LEN:

                padding = np.zeros(
                    (
                        SEQUENCE_LEN - len(seq),
                        len(SEQ_FEATURES)
                    )
                )

                seq = np.vstack(
                    [padding, seq]
                )

            sequences.append(seq)
            indices.append(original_indices[i])

    X_seq = np.array(
        sequences,
        dtype=np.float32
    )

    indices = np.array(indices)

    # Scale using the SAME scaler used during training
    n, t, f = X_seq.shape

    X_flat = X_seq.reshape(-1, f)

    X_flat = scaler.transform(X_flat)

    X_seq = X_flat.reshape(
        n,
        t,
        f
    )

    # Predict
    scores = model.predict(
        X_seq,
        verbose=0
    ).flatten()

    # Put scores back into original transaction order
    ordered_scores = np.zeros(len(df))

    ordered_scores[indices] = scores

    return ordered_scores