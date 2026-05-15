import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, roc_auc_score, average_precision_score,
    confusion_matrix
)
from imblearn.over_sampling import SMOTE
import joblib
import os

from src.preprocess import get_feature_cols

def train_behavioral_engine(df: pd.DataFrame, model_dir: str = "models"):
    os.makedirs(model_dir, exist_ok=True)
    FEATURES = get_feature_cols()
    X = df[FEATURES].fillna(0)
    y = df["isFraud"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # --- Handle class imbalance with SMOTE ---
    print("[Phase 2] Applying SMOTE for class balance...")
    sm = SMOTE(random_state=42, k_neighbors=3)
    X_res, y_res = sm.fit_resample(X_train, y_train)

    # --- XGBoost ---
    print("[Phase 2] Training XGBoost...")
    scale_pos = (y_train == 0).sum() / (y_train == 1).sum()
    xgb_model = xgb.XGBClassifier(
        n_estimators=80,
        max_depth=4,
        learning_rate=0.05,
        scale_pos_weight=scale_pos,
        use_label_encoder=False,
        eval_metric="aucpr",
        random_state=42,
        n_jobs=-1
    )
    xgb_model.fit(
        X_res, y_res,
        eval_set=[(X_test, y_test)],
        verbose=50
    )

    # --- Isolation Forest (unsupervised anomaly layer) ---
    print("[Phase 2] Training Isolation Forest...")
    iso_forest = IsolationForest(
        n_estimators=200,
        contamination=0.02,
        random_state=42,
        n_jobs=-1
    )
    iso_forest.fit(X_res)

    # --- Evaluate ---
    xgb_proba = xgb_model.predict_proba(X_test)[:, 1]
    iso_scores = -iso_forest.score_samples(X_test)  # higher = more anomalous
    # Normalize iso scores to [0,1]
    iso_norm = (iso_scores - iso_scores.min()) / (iso_scores.max() - iso_scores.min() + 1e-9)

    print("\n[Phase 2] XGBoost Evaluation:")
    print(classification_report(y_test, (xgb_proba > 0.5).astype(int)))
    print(f"ROC-AUC: {roc_auc_score(y_test, xgb_proba):.4f}")
    print(f"PR-AUC:  {average_precision_score(y_test, xgb_proba):.4f}")

    # Save
    joblib.dump(xgb_model, f"{model_dir}/xgb_model.pkl")
    joblib.dump(iso_forest, f"{model_dir}/iso_forest.pkl")
    print("[Phase 2] Models saved.")

    return xgb_model, iso_forest, X_test, y_test


def load_behavioral_engine(model_dir: str = "models"):
    xgb_model = joblib.load(f"{model_dir}/xgb_model.pkl")
    iso_forest = joblib.load(f"{model_dir}/iso_forest.pkl")
    return xgb_model, iso_forest


def get_behavioral_scores(xgb_model, iso_forest, X: pd.DataFrame):
    xgb_proba = xgb_model.predict_proba(X)[:, 1]
    iso_raw = -iso_forest.score_samples(X)
    iso_norm = (iso_raw - iso_raw.min()) / (iso_raw.max() - iso_raw.min() + 1e-9)
    return xgb_proba, iso_norm