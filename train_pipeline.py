"""
Reproducible MVP training/evaluation pipeline.

Run from project root:
    python train_pipeline.py

Expected:
    data/PS_20174392719_1491204439457_log.csv

Outputs:
    models/
    results/evaluation_results.csv
    results/ablation_results.csv
    results/robustness_results.csv

IMPORTANT:
- This pipeline makes the paper claims match implemented experiments.
- Federated learning is simulated with 3 non-IID clients and FedAvg.
- PaySim does not contain device IDs or merchant IDs, so the HGNN uses
  account + transaction node types rather than inventing unavailable entities.
"""
import os
import random
import time
import numpy as np
import pandas as pd
import torch

from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score

from src.preprocess import load_and_engineer, get_feature_cols, get_model_matrix
from src.behavioral_engine import train_behavioral_engine, get_behavioral_scores
from src.federated_lstm import train_federated_lstm, train_centralized_lstm
from src.hetero_gnn_model import (
    build_hetero_graph, train_hetero_gnn, predict_transaction_scores
)
from src.risk_fusion import fuse_risk_scores
from src.evaluation import (
    evaluate_scores, evaluate_strong_baselines, run_ablation, save_results,
    best_f1_threshold
)
from src.robustness import temporal_split, robustness_report


CSV_PATH = "data/PS_20174392719_1491204439457_log.csv"
NROWS = 20000
SEED = 42
SEQ_LEN = 5


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def clean_json_values(value):
    if isinstance(value, dict):
        return {str(k): clean_json_values(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_json_values(v) for v in value]
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return None
    if isinstance(value, np.integer):
        return int(value)
    return value


def make_sequences(df, feature_names, seq_len=5):
    """
    Sequence is created per sender. Each sequence label belongs to its final
    transaction. Returns sequence tensors plus the final transaction IDs.
    """
    Xs, ys, ids = [], [], []

    work = df.sort_values(["nameOrig", "step", "txn_id"])
    for _, grp in work.groupby("nameOrig", sort=False):
        vals = grp[feature_names].fillna(0).to_numpy(dtype=np.float32)
        labels = grp["isFraud"].to_numpy(dtype=np.float32)
        txn_ids = grp["txn_id"].to_numpy(dtype=np.int64)

        for i in range(len(grp)):
            window = vals[max(0, i - seq_len + 1):i + 1]
            if len(window) < seq_len:
                # Deterministic cold-start: left-pad with this sender's first
                # observed event so every transaction has a fixed-length input.
                window = np.concatenate([np.repeat(window[:1], seq_len - len(window), axis=0), window], axis=0)
            Xs.append(window)
            ys.append(labels[i])
            ids.append(txn_ids[i])

    return (
        np.asarray(Xs, dtype=np.float32),
        np.asarray(ys, dtype=np.float32),
        np.asarray(ids, dtype=np.int64),
    )


def main():
    set_seed(SEED)
    os.makedirs("results", exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load + engineer
    # ------------------------------------------------------------------
    df = load_and_engineer(CSV_PATH, nrows=NROWS)
    feature_cols = get_feature_cols()

    # Keep a single fixed transaction-level split for all core comparisons.
    # Stratification keeps the extremely rare fraud class represented.
    train_idx, remainder_idx = train_test_split(
        np.arange(len(df)),
        test_size=0.30,
        random_state=SEED,
        stratify=df["isFraud"].values,
    )
    val_idx, test_idx = train_test_split(
        remainder_idx, test_size=0.50, random_state=SEED,
        stratify=df["isFraud"].iloc[remainder_idx].values,
    )
    train_idx, val_idx, test_idx = (np.sort(x) for x in (train_idx, val_idx, test_idx))
    pd.concat([
        pd.DataFrame({"txn_id": df.iloc[idx]["txn_id"], "split": split, "label": df.iloc[idx]["isFraud"]})
        for split, idx in (("train", train_idx), ("validation", val_idx), ("test", test_idx))
    ], ignore_index=True).to_csv("results/splits.csv", index=False)

    X_all = get_model_matrix(df)
    X_train = X_all.iloc[train_idx]
    X_test = X_all.iloc[test_idx]
    y_train = df["isFraud"].iloc[train_idx]
    y_test = df["isFraud"].iloc[test_idx]

    # ------------------------------------------------------------------
    # 2. Behavioral layer: XGBoost + Isolation Forest
    # ------------------------------------------------------------------
    xgb_model, iso_model, _, _, _, _ = train_behavioral_engine(
        df, train_idx=train_idx, test_idx=test_idx, val_idx=val_idx
    )

    xgb_infer_start = time.perf_counter()
    xgb_scores_val = xgb_model.predict_proba(X_all.iloc[val_idx])[:, 1]
    xgb_scores_test = xgb_model.predict_proba(X_test)[:, 1]
    xgb_inference_seconds = time.perf_counter() - xgb_infer_start
    def scale_anomaly(raw, lo, hi): return np.clip((raw - lo) / (hi - lo + 1e-9), 0, 1)
    iso_train_raw = -iso_model.score_samples(X_all.iloc[train_idx])
    iso_lo, iso_hi = float(iso_train_raw.min()), float(iso_train_raw.max())
    iso_infer_start = time.perf_counter()
    iso_scores_val = scale_anomaly(-iso_model.score_samples(X_all.iloc[val_idx]), iso_lo, iso_hi)
    iso_scores_test = scale_anomaly(-iso_model.score_samples(X_test), iso_lo, iso_hi)
    iso_inference_seconds = time.perf_counter() - iso_infer_start

    # ------------------------------------------------------------------
    # 3. LSTM sequences
    # ------------------------------------------------------------------
    seq_features = [
        "amount", "type_enc", "orig_balance_error", "dest_balance_error",
        "amount_deviation", "txn_gap"
    ]
    X_seq, y_seq, seq_ids = make_sequences(df, seq_features, SEQ_LEN)

    train_id_set = set(df.iloc[train_idx]["txn_id"].values.tolist())
    test_id_set = set(df.iloc[test_idx]["txn_id"].values.tolist())

    seq_train_mask = np.array([i in train_id_set for i in seq_ids], dtype=bool)
    val_id_set = set(df.iloc[val_idx]["txn_id"].values.tolist())
    seq_val_mask = np.array([i in val_id_set for i in seq_ids], dtype=bool)
    seq_test_mask = np.array([i in test_id_set for i in seq_ids], dtype=bool)

    Xs_train = X_seq[seq_train_mask]
    ys_train = y_seq[seq_train_mask]
    Xs_test = X_seq[seq_test_mask]
    Xs_val = X_seq[seq_val_mask]
    ids_val = seq_ids[seq_val_mask]
    from sklearn.preprocessing import StandardScaler
    seq_scaler = StandardScaler().fit(Xs_train.reshape(-1, Xs_train.shape[-1]))
    Xs_train = seq_scaler.transform(Xs_train.reshape(-1, Xs_train.shape[-1])).reshape(Xs_train.shape).astype(np.float32)
    Xs_val = seq_scaler.transform(Xs_val.reshape(-1, Xs_val.shape[-1])).reshape(Xs_val.shape).astype(np.float32)
    Xs_test = seq_scaler.transform(Xs_test.reshape(-1, Xs_test.shape[-1])).reshape(Xs_test.shape).astype(np.float32)
    import joblib
    joblib.dump(seq_scaler, "models/lstm_sequence_scaler.pkl")
    ids_test = seq_ids[seq_test_mask]

    # Federated LSTM.
    lstm_fed_start = time.perf_counter()
    fed_model, partitions, fed_history, total_upload = train_federated_lstm(
        Xs_train,
        ys_train,
        input_dim=Xs_train.shape[-1],
        n_clients=3,
        rounds=5,
        local_epochs=1,
        alpha=0.30,
        seed=SEED,
    )
    lstm_fed_seconds = time.perf_counter() - lstm_fed_start

    fed_model.eval()
    with torch.no_grad():
        lstm_scores_val_seq = torch.sigmoid(fed_model(torch.tensor(Xs_val, dtype=torch.float32))).cpu().numpy()
        lstm_scores_seq = torch.sigmoid(
            fed_model(torch.tensor(Xs_test, dtype=torch.float32))
        ).cpu().numpy()

    # Matched centralized LSTM for the federated-learning ablation.
    lstm_central_start = time.perf_counter()
    centralized_model = train_centralized_lstm(
        Xs_train, ys_train,
        input_dim=Xs_train.shape[-1],
        epochs=5,
        seed=SEED,
    )
    lstm_central_seconds = time.perf_counter() - lstm_central_start
    centralized_model.eval()
    with torch.no_grad():
        central_scores_val_seq = torch.sigmoid(centralized_model(torch.tensor(Xs_val, dtype=torch.float32))).cpu().numpy()
        centralized_scores_seq = torch.sigmoid(
            centralized_model(torch.tensor(Xs_test, dtype=torch.float32))
        ).cpu().numpy()

    # Map sequence scores back to transaction IDs.
    lstm_map = pd.Series(lstm_scores_seq, index=ids_test)
    lstm_val_map = pd.Series(lstm_scores_val_seq, index=ids_val)
    centralized_lstm_map = pd.Series(
        centralized_scores_seq, index=ids_test
    )
    centralized_val_map = pd.Series(central_scores_val_seq, index=ids_val)

    lstm_scores_test = (
        df.iloc[test_idx]["txn_id"].map(lstm_map).fillna(0.0).to_numpy()
    )
    lstm_scores_val = df.iloc[val_idx]["txn_id"].map(lstm_val_map).fillna(0.0).to_numpy()
    centralized_lstm_scores_test = (
        df.iloc[test_idx]["txn_id"]
        .map(centralized_lstm_map)
        .fillna(0.0)
        .to_numpy()
    )
    centralized_lstm_scores_val = df.iloc[val_idx]["txn_id"].map(centralized_val_map).fillna(0.0).to_numpy()
    torch.save(centralized_model.state_dict(), "models/lstm_centralized.pt")
    torch.save(fed_model.state_dict(), "models/lstm_federated.pt")

    # ------------------------------------------------------------------
    # 4. Heterogeneous GNN
    # ------------------------------------------------------------------
    graph, _ = build_hetero_graph(df, feature_cols)
    hgnn_train_start = time.perf_counter()
    hgnn_model, graph = train_hetero_gnn(
        graph,
        train_idx=train_idx,
        epochs=30,
        lr=1e-3,
        weight_decay=1e-4,
        seed=SEED,
    )
    hgnn_training_seconds = time.perf_counter() - hgnn_train_start
    hgnn_infer_start = time.perf_counter()
    gnn_all_scores = predict_transaction_scores(hgnn_model, graph)
    hgnn_inference_seconds = time.perf_counter() - hgnn_infer_start
    gnn_scores_test = gnn_all_scores[test_idx]
    gnn_scores_val = gnn_all_scores[val_idx]
    torch.save(hgnn_model.state_dict(), "models/hgnn_model.pt")

    # ------------------------------------------------------------------
    # 5. Full fused model
    # ------------------------------------------------------------------
    full_scores = fuse_risk_scores(
        xgb_scores_test,
        iso_scores_test,
        lstm_scores_test,
        gnn_scores_test,
    )
    val_fused_scores = fuse_risk_scores(xgb_scores_val, iso_scores_val, lstm_scores_val, gnn_scores_val)
    threshold = best_f1_threshold(df.iloc[val_idx]["isFraud"].values, val_fused_scores)
    with open("models/thresholds.json", "w", encoding="utf-8") as f:
        import json
        json.dump({"decision_threshold": threshold, "selection_split": "validation", "risk_levels": {"medium": 0.35, "high": 0.65, "critical": 0.90}}, f, indent=2)

    full_row = evaluate_scores(
        y_test.values, full_scores, threshold=threshold,
        name="Full: XGB + IF + LSTM(FedAvg) + HGNN"
    )

    # ------------------------------------------------------------------
    # 6. Strong single-model baselines on identical split
    # ------------------------------------------------------------------
    baseline_df = evaluate_strong_baselines(X_train, y_train, X_test, y_test, X_val=X_all.iloc[val_idx], y_val=df.iloc[val_idx]["isFraud"])
    y_val = df.iloc[val_idx]["isFraud"].values
    baseline_df = pd.concat([baseline_df, pd.DataFrame([evaluate_scores(y_test.values, full_scores, threshold=threshold, name="Full fused model")])], ignore_index=True)
    for label, val_component, test_component in (
        ("Isolation Forest only", iso_scores_val, iso_scores_test),
        ("LSTM only (FedAvg)", lstm_scores_val, lstm_scores_test),
        ("HGNN only", gnn_scores_val, gnn_scores_test),
    ):
        component_threshold = best_f1_threshold(y_val, val_component)
        baseline_df = pd.concat([baseline_df, pd.DataFrame([evaluate_scores(y_test.values, test_component, threshold=component_threshold, name=label)])], ignore_index=True)
    baseline_df.to_csv("results/strong_baselines.csv", index=False)

    # ------------------------------------------------------------------
    # 7. Ablation study
    # ------------------------------------------------------------------
    # Remove IF: replace anomaly contribution with zeros.
    no_anomaly = fuse_risk_scores(
        xgb_scores_test,
        np.zeros_like(iso_scores_test),
        lstm_scores_test,
        gnn_scores_test,
    )

    # Remove LSTM.
    no_temporal = fuse_risk_scores(
        xgb_scores_test,
        iso_scores_test,
        np.zeros_like(lstm_scores_test),
        gnn_scores_test,
    )

    # Remove HGNN.
    no_graph = fuse_risk_scores(
        xgb_scores_test,
        iso_scores_test,
        lstm_scores_test,
        np.zeros_like(gnn_scores_test),
    )

    # Federated-learning ablation: replace the federated LSTM score with the
    # matched centralized LSTM score while keeping every other module fixed.
    centralized_full = fuse_risk_scores(
        xgb_scores_test,
        iso_scores_test,
        centralized_lstm_scores_test,
        gnn_scores_test,
    )
    val_ablation_scores = {
        "full": val_fused_scores,
        "no_anomaly": fuse_risk_scores(xgb_scores_val, np.zeros_like(iso_scores_val), lstm_scores_val, gnn_scores_val),
        "no_temporal": fuse_risk_scores(xgb_scores_val, iso_scores_val, np.zeros_like(lstm_scores_val), gnn_scores_val),
        "no_graph": fuse_risk_scores(xgb_scores_val, iso_scores_val, lstm_scores_val, np.zeros_like(gnn_scores_val)),
        "centralized": fuse_risk_scores(xgb_scores_val, iso_scores_val, centralized_lstm_scores_val, gnn_scores_val),
    }

    ablation_scores = {
        "full": full_scores,
        "no_anomaly": no_anomaly,
        "no_temporal": no_temporal,
        "no_graph": no_graph,
        "centralized": centralized_full,
    }
    ablation_thresholds = {key: best_f1_threshold(y_val, valscore) for key, valscore in val_ablation_scores.items()}
    ablation_df = run_ablation(y_test.values, ablation_scores, ablation_thresholds)
    ablation_df.to_csv("results/ablation_results.csv", index=False)

    # ------------------------------------------------------------------
    # 8. Temporal robustness / concept-drift-oriented split
    # ------------------------------------------------------------------
    temporal_train, temporal_val, temporal_test = temporal_split(df)

    # The temporal split is reported as a separate evaluation track. It is
    # intentionally not mixed with the random-split numbers above.
    print(
        "\n[Temporal robustness split]",
        f"train={len(temporal_train):,}",
        f"val={len(temporal_val):,}",
        f"test={len(temporal_test):,}",
    )

    # Dataset-only stress tests on the common held-out test set.
    robustness_df = robustness_report(
        y_test.values,
        full_scores,
        df.iloc[test_idx].reset_index(drop=True),
        threshold=threshold,
    )
    robustness_df.to_csv("results/robustness_results.csv", index=False)
    # Secondary future-data track: separately fit XGBoost on earliest rows,
    # choose its operating threshold on the next 15%, then evaluate latest 15%.
    temporal_train, temporal_val, temporal_test = temporal_split(df, .70, .15)
    t_train = temporal_train.index.to_numpy()
    t_val = temporal_val.index.to_numpy()
    t_test = temporal_test.index.to_numpy()
    temporal_xgb, _, *_ = train_behavioral_engine(df, train_idx=t_train, test_idx=t_test, val_idx=t_val, model_dir="models/temporal")
    tv = temporal_xgb.predict_proba(X_all.iloc[t_val])[:, 1]
    tt = temporal_xgb.predict_proba(X_all.iloc[t_test])[:, 1]
    temporal_threshold = best_f1_threshold(df.iloc[t_val].isFraud.values, tv)
    temporal_row = evaluate_scores(df.iloc[t_test].isFraud.values, tt, threshold=temporal_threshold, name="Chronological XGBoost")
    temporal_row["train_rows"], temporal_row["validation_rows"], temporal_row["future_test_rows"] = len(t_train), len(t_val), len(t_test)
    pd.DataFrame([temporal_row]).to_csv("results/temporal_results.csv", index=False)
    fed_threshold = best_f1_threshold(y_val, lstm_scores_val)
    cen_threshold = best_f1_threshold(y_val, centralized_lstm_scores_val)
    pd.DataFrame([evaluate_scores(y_test.values, lstm_scores_test, threshold=fed_threshold, name="Federated LSTM"), evaluate_scores(y_test.values, centralized_lstm_scores_test, threshold=cen_threshold, name="Centralized LSTM")]).to_csv("results/federated_vs_centralized.csv", index=False)

    # ------------------------------------------------------------------
    # 9. Save core summary
    # ------------------------------------------------------------------
    summary = pd.DataFrame([full_row])
    save_results(summary)

    pd.DataFrame(fed_history).to_csv(
        "results/federated_rounds.csv", index=False
    )
    pd.DataFrame([
        {"model":"XGBoost","training_seconds":getattr(xgb_model,"training_seconds_",None),"inference_seconds":xgb_inference_seconds,"inference_ms_per_transaction":xgb_inference_seconds/(len(val_idx)+len(test_idx))*1000,"model_bytes":os.path.getsize("models/xgb_model.pkl"),"feature_dimension":len(feature_cols)},
        {"model":"Isolation Forest","training_seconds":getattr(iso_model,"training_seconds_",None),"inference_seconds":iso_inference_seconds,"inference_ms_per_transaction":iso_inference_seconds/(len(val_idx)+len(test_idx))*1000,"model_bytes":os.path.getsize("models/iso_forest.pkl"),"feature_dimension":len(feature_cols)},
        {"model":"Federated LSTM","training_seconds":lstm_fed_seconds,"inference_seconds":float(np.nan),"inference_ms_per_transaction":float(np.nan),"model_bytes":os.path.getsize("models/lstm_federated.pt"),"feature_dimension":6,"communication_upload_bytes":total_upload,"communication_download_bytes":sum(r["download_bytes"] for r in fed_history),"communication_total_bytes":total_upload+sum(r["download_bytes"] for r in fed_history)},
        {"model":"Centralized LSTM","training_seconds":lstm_central_seconds,"inference_seconds":float(np.nan),"inference_ms_per_transaction":float(np.nan),"model_bytes":os.path.getsize("models/lstm_centralized.pt"),"feature_dimension":6},
        {"model":"Heterogeneous GNN","training_seconds":hgnn_training_seconds,"inference_seconds":hgnn_inference_seconds,"inference_ms_per_transaction":hgnn_inference_seconds/len(df)*1000,"model_bytes":os.path.getsize("models/hgnn_model.pt"),"feature_dimension":len(feature_cols)}
    ]).to_csv("results/computational_overhead.csv", index=False)
    pd.DataFrame([{"split":name,"rows":len(idx),"legitimate":int((df.iloc[idx].isFraud==0).sum()),"fraud":int((df.iloc[idx].isFraud==1).sum()),"post_balance_legitimate":int((df.iloc[idx].isFraud==0).sum()),"post_balance_fraud":int((df.iloc[idx].isFraud==1).sum()),"imbalance_strategy":"class weights; no resampling"} for name,idx in (("train",train_idx),("validation",val_idx),("test",test_idx))]).to_csv("results/class_distribution.csv",index=False)
    pd.DataFrame([{k:full_row[k] for k in ("TN","FP","FN","TP")}]).to_csv("results/confusion_matrix.csv",index=False)
    import json
    with open("results/final_summary.json","w",encoding="utf-8") as f:
        summary_data = {
            "dataset":"PaySim synthetic mobile-money proxy", "rows":len(df),
            "split":{"train":len(train_idx),"validation":len(val_idx),"test":len(test_idx)},
            "threshold":threshold, "threshold_selected_on":"validation", "full_model":full_row,
            "strong_baselines":pd.read_csv("results/strong_baselines.csv").to_dict(orient="records"),
            "ablations":pd.read_csv("results/ablation_results.csv").to_dict(orient="records"),
            "temporal_evaluation":pd.read_csv("results/temporal_results.csv").to_dict(orient="records"),
            "robustness_stress_proxies":pd.read_csv("results/robustness_results.csv").to_dict(orient="records"),
            "federated_vs_centralized":pd.read_csv("results/federated_vs_centralized.csv").to_dict(orient="records"),
            "computational_overhead":pd.read_csv("results/computational_overhead.csv").to_dict(orient="records"),
            "federated_rounds":fed_history,
            "federated_communication":{"upload_bytes":int(total_upload),"download_bytes":int(sum(r["download_bytes"] for r in fed_history)),"total_bytes":int(total_upload+sum(r["download_bytes"] for r in fed_history))},
            "limitations":["transductive HGNN","FedAvg simulated on one machine","PaySim is not actual UPI data","chronological scores use an XGBoost-only secondary model"]
        }
        json.dump(clean_json_values(summary_data), f, indent=2, allow_nan=False)
    with open("models/metadata.json","w",encoding="utf-8") as f:
        json.dump({"model_version":"1.0.0","training_dataset":"PaySim synthetic mobile-money proxy","rows_used":len(df),"split":{"train":len(train_idx),"validation":len(val_idx),"test":len(test_idx)},"feature_version":"paysim_features_v1_18","sequence":{"length":SEQ_LEN,"features":6},"seed":SEED,"trained_at_utc":pd.Timestamp.now(tz="UTC").isoformat(),"xgboost":{"n_estimators":80,"max_depth":4,"learning_rate":0.05},"isolation_forest":{"n_estimators":200,"contamination":0.02},"federated":{"clients":3,"rounds":5,"local_epochs":1,"dirichlet_alpha":0.30},"hgnn":{"node_types":["account","transaction"],"epochs":30,"hidden":32,"heads":[2,1]}},f,indent=2)

    with open("results/experiment_config.txt", "w", encoding="utf-8") as f:
        f.write(
            "seed=42\n"
            "nrows=20000\n"
            "random_train=0.70\nrandom_validation=0.15\nrandom_test=0.15\n"
            "fed_clients=3\n"
            "fed_rounds=5\n"
            "fed_local_epochs=1\n"
            "fed_dirichlet_alpha=0.30\n"
            "lstm_sequence_length=5\n"
            "hgnn_hidden=32\n"
            "hgnn_heads=2\n"
            "hgnn_epochs=30\n"
            f"federated_upload_bytes={total_upload}\n"
        )

    print("\n====================================================")
    print("FINAL EVALUATION")
    print("====================================================")
    for k, v in full_row.items():
        if isinstance(v, float):
            print(f"{k}: {v:.4f}")
        else:
            print(f"{k}: {v}")

    print("\nFiles written under results/:")
    print("  strong_baselines.csv")
    print("  ablation_results.csv")
    print("  robustness_results.csv")
    print("  federated_rounds.csv")
    print("  evaluation_results.csv")
    print("  experiment_config.txt")


if __name__ == "__main__":
    main()
