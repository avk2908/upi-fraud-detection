import os
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
import time

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler

from src.preprocess import get_feature_cols, get_model_matrix


def train_behavioral_engine(
    df: pd.DataFrame,
    train_idx=None,
    test_idx=None,
    val_idx=None,
    model_dir: str = "models",
    seed: int = 42,
):
    """
    Train XGBoost + Isolation Forest on an explicitly supplied split.

    This is important for the paper: every model must use the same held-out
    test transactions.
    """
    os.makedirs(model_dir, exist_ok=True)
    features = get_feature_cols()
    X = get_model_matrix(df)
    y = df["isFraud"].astype(int)

    if train_idx is None or test_idx is None:
        raise ValueError(
            "Pass explicit train_idx and test_idx so the experiment uses "
            "one identical split across models."
        )

    X_train = X.iloc[train_idx]
    X_test = X.iloc[test_idx]
    y_train = y.iloc[train_idx]
    y_test = y.iloc[test_idx]

    # Use class weighting on the untouched training partition. Keeping sample
    # IDs intact is important for matched comparisons and auditable splits.

    print("[Phase 2] Training XGBoost...")
    scale_pos = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    model = xgb.XGBClassifier(
        n_estimators=80,
        max_depth=4,
        learning_rate=0.05,
        scale_pos_weight=scale_pos,
        eval_metric="aucpr",
        random_state=seed,
        n_jobs=1,
    )
    start = time.perf_counter()
    eval_set = None
    if val_idx is not None:
        eval_set = [(X.iloc[val_idx], y.iloc[val_idx])]
    model.fit(X_train, y_train, eval_set=eval_set, verbose=False)
    model.training_seconds_ = time.perf_counter() - start

    print("[Phase 2] Training Isolation Forest...")
    iso = IsolationForest(
        n_estimators=200,
        contamination=0.02,
        random_state=seed,
        n_jobs=1,
    )
    start = time.perf_counter()
    iso.fit(X_train)
    iso.training_seconds_ = time.perf_counter() - start

    anomaly_scaler = MinMaxScaler(clip=True).fit((-iso.score_samples(X_train)).reshape(-1, 1))

    joblib.dump(model, f"{model_dir}/xgb_model.pkl")
    joblib.dump(iso, f"{model_dir}/iso_forest.pkl")
    joblib.dump(anomaly_scaler, f"{model_dir}/iso_scaler.pkl")

    return model, iso, X_train, X_test, y_train, y_test


def load_behavioral_engine(model_dir="models"):
    return (
        joblib.load(f"{model_dir}/xgb_model.pkl"),
        joblib.load(f"{model_dir}/iso_forest.pkl"),
    )


def get_behavioral_scores(xgb_model, iso_forest, X):
    xgb_scores = xgb_model.predict_proba(X)[:, 1]
    raw = -iso_forest.score_samples(X)
    iso_scores = (raw - raw.min()) / (raw.max() - raw.min() + 1e-9)
    return xgb_scores, iso_scores
