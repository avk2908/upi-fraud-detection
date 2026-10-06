"""FastAPI scoring API and sequential PaySim replay service."""
from __future__ import annotations

import asyncio
import json
import os
import sqlite3
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable
import threading

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "outputs" / ".matplotlib"))

import joblib
import numpy as np
import pandas as pd
import torch
from src.federated_lstm import LSTMClassifier
from src.hetero_gnn_model import PaySimHeteroGNN, build_hetero_graph, predict_transaction_scores
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

DATA_PATH = Path(os.getenv("PAYSIM_CSV", ROOT / "data/PS_20174392719_1491204439457_log.csv"))
MODEL_DIR = ROOT / "models"
FEATURES = ["step", "type_enc", "amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest", "orig_balance_error", "dest_balance_error", "zero_dest_after", "zero_orig_before", "sender_rolling_mean_amt", "amount_deviation", "sender_txn_count", "txn_gap", "receiver_novelty", "dest_frequency", "pair_count"]
TYPES = {"CASH_IN": 0, "CASH_OUT": 1, "DEBIT": 2, "PAYMENT": 3, "TRANSFER": 4}


class TransactionInput(BaseModel):
    transaction_id: str | None = None
    step: float = 0
    type: str = "TRANSFER"
    amount: float = Field(ge=0)
    oldbalanceOrg: float = Field(default=0, ge=0)
    newbalanceOrig: float = Field(default=0, ge=0)
    oldbalanceDest: float = Field(default=0, ge=0)
    newbalanceDest: float = Field(default=0, ge=0)
    nameOrig: str = "unknown-sender"
    nameDest: str = "unknown-receiver"
    isFraud: int | None = None


class SimulatorConfig(BaseModel):
    rate: float = Field(default=1, ge=0.1, le=100)


class ScoringService:
    def __init__(self) -> None:
        self.models: dict[str, Any] = {}
        self.status: dict[str, str] = {}
        self.history: deque[dict[str, Any]] = deque(maxlen=5000)
        self.score_times: deque[float] = deque(maxlen=5000)
        self.threshold = 0.5
        self.levels = {"medium": 0.35, "high": 0.65, "critical": 0.9}
        self.replay_rows: list[dict[str, Any]] | None = None
        self.cursor = 0
        self.rate = 1.0
        self.running = False
        self.task: asyncio.Task | None = None
        self.clients: set[WebSocket] = set()
        self.sender_sequences: dict[str, list[list[float]]] = {}
        self.graph_events: deque[dict[str, Any]] = deque(maxlen=2000)
        self.lstm_model = None
        self.lstm_scaler = None
        self.hgnn_model = None
        self.last_latency_ms: float | None = None
        self.db_path = ROOT / "outputs" / "fraud_monitor.sqlite3"
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.scoring_lock = threading.Lock()
        self._init_db()
        self._load_models()

    def _init_db(self) -> None:
        with sqlite3.connect(self.db_path) as db:
            db.execute("CREATE TABLE IF NOT EXISTS transactions (transaction_id TEXT PRIMARY KEY, timestamp TEXT NOT NULL, payload TEXT NOT NULL)")
            db.execute("CREATE TABLE IF NOT EXISTS alerts (alert_id TEXT PRIMARY KEY, transaction_id TEXT NOT NULL, account_id TEXT NOT NULL, risk_score REAL NOT NULL, severity TEXT NOT NULL, reason TEXT NOT NULL, timestamp TEXT NOT NULL, payload TEXT NOT NULL)")
            saved = db.execute("SELECT payload FROM transactions ORDER BY timestamp DESC LIMIT 5000").fetchall()
            for (payload,) in reversed(saved):
                try:
                    item = json.loads(payload)
                    self.history.append(item)
                    self.score_times.append(datetime.fromisoformat(item["timestamp"]).timestamp())
                    if item.get("features"):
                        event = {"features": item["features"], "nameOrig": item["source_account"], "nameDest": item["destination_account"], "isFraud": 0}
                        self.graph_events.append(event)
                        seq = self.sender_sequences.setdefault(event["nameOrig"], [])
                        f = event["features"]
                        seq.append([float(f[k]) for k in ("amount", "type_enc", "orig_balance_error", "dest_balance_error", "amount_deviation", "txn_gap")])
                        if len(seq) > 5: del seq[:-5]
                except (ValueError, KeyError, TypeError):
                    continue

    def _load_models(self) -> None:
        for key, filename in (("xgboost", "xgb_model.pkl"), ("isolation_forest", "iso_forest.pkl")):
            path = MODEL_DIR / filename
            try:
                self.models[key] = joblib.load(path)
                self.status[key] = "ready"
            except Exception as exc:
                self.status[key] = f"unavailable: {type(exc).__name__}: {exc}"
        for key, artifact in (("lstm", "lstm_federated.pt"), ("hgnn", "hgnn_model.pt")):
            # A state dict without its architecture/scaler/inference context is
            # not a usable online model adapter.
            self.status[key] = "unavailable: inference adapter not implemented" if (MODEL_DIR / artifact).exists() else "unavailable: trained artifact missing"
        try:
            lstm_path = MODEL_DIR / "lstm_federated.pt"
            scaler_path = MODEL_DIR / "lstm_sequence_scaler.pkl"
            if lstm_path.exists() and scaler_path.exists():
                self.lstm_model = LSTMClassifier(6).cpu().eval()
                self.lstm_model.load_state_dict(torch.load(lstm_path, map_location="cpu", weights_only=True))
                self.lstm_scaler = joblib.load(scaler_path)
                self.status["lstm"] = "ready"
        except Exception as exc:
            self.status["lstm"] = f"unavailable: {type(exc).__name__}: {exc}"
        try:
            hgnn_path = MODEL_DIR / "hgnn_model.pt"
            if hgnn_path.exists():
                self.hgnn_model = PaySimHeteroGNN(tx_dim=len(FEATURES), account_dim=7).cpu().eval()
                self.hgnn_model.load_state_dict(torch.load(hgnn_path, map_location="cpu", weights_only=True))
                self.status["hgnn"] = "ready"
        except Exception as exc:
            self.status["hgnn"] = f"unavailable: {type(exc).__name__}: {exc}"
        self.status["federated_lstm"] = "training protocol artifact present" if (MODEL_DIR / "lstm_federated.pt").exists() else "unavailable: trained artifact missing"
        try:
            self.models["anomaly_scaler"] = joblib.load(MODEL_DIR / "iso_scaler.pkl")
        except Exception:
            if "isolation_forest" in self.models:
                self.status["isolation_forest"] = "ready: deterministic sigmoid score scaling; train scaler missing"
        try:
            cfg = json.loads((MODEL_DIR / "thresholds.json").read_text(encoding="utf-8"))
            self.threshold = float(cfg.get("decision_threshold", 0.5))
            self.levels.update(cfg.get("risk_levels", {}))
        except Exception:
            pass
        if DATA_PATH.exists():
            try:
                self.replay_rows = pd.read_csv(DATA_PATH, nrows=50000).to_dict(orient="records")
            except Exception as exc:
                self.status["simulator"] = f"unavailable: {exc}"

    @staticmethod
    def _features(tx: TransactionInput) -> pd.DataFrame:
        # Live request features use only current event data; history-dependent
        # features are explicitly cold-started to zero and reported as such.
        oe = abs(tx.oldbalanceOrg - tx.amount - tx.newbalanceOrig)
        de = abs(tx.oldbalanceDest + tx.amount - tx.newbalanceDest)
        values = [tx.step, TYPES.get(tx.type.upper(), -1), tx.amount, tx.oldbalanceOrg,
                  tx.newbalanceOrig, tx.oldbalanceDest, tx.newbalanceDest, oe, de,
                  int(tx.newbalanceDest == 0), int(tx.oldbalanceOrg == 0), 0.0,
                  tx.amount, 0.0, 0.0, 1.0, 0.0, 0.0]
        return pd.DataFrame([values], columns=FEATURES)

    def score(self, tx: TransactionInput, progress_callback: Callable[[dict[str, Any]], None] | None = None) -> dict[str, Any]:
        with self.scoring_lock:
            return self._score(tx, progress_callback)

    def _score(self, tx: TransactionInput, progress_callback: Callable[[dict[str, Any]], None] | None = None) -> dict[str, Any]:
        started = time.perf_counter()
        transaction_id = tx.transaction_id or f"live-{int(time.time() * 1000)}"

        def report(name: str, status: str, latency_ms: float | None = None, score: float | None = None, error: str | None = None) -> None:
            if progress_callback is not None:
                progress_callback({"name": name, "status": status, "latency_ms": latency_ms, "score": score, "error": error})

        report("Transaction received", "complete")
        feature_started = time.perf_counter()
        report("Feature engineering", "running")
        row = self._features(tx)
        prior_events = list(self.graph_events)
        prior_sender = [event for event in prior_events if event["nameOrig"] == tx.nameOrig]
        prior_receiver = [event for event in prior_events if event["nameDest"] == tx.nameDest]
        prior_pair = [event for event in prior_sender if event["nameDest"] == tx.nameDest]
        row.loc[0, "sender_txn_count"] = len(prior_sender)
        row.loc[0, "pair_count"] = len(prior_pair)
        row.loc[0, "receiver_novelty"] = int(not prior_pair)
        row.loc[0, "dest_frequency"] = len(prior_receiver)
        if prior_sender:
            previous_amounts = [float(event["features"]["amount"]) for event in prior_sender]
            row.loc[0, "sender_rolling_mean_amt"] = float(np.mean(previous_amounts))
            row.loc[0, "amount_deviation"] = abs(tx.amount - float(row.loc[0, "sender_rolling_mean_amt"]))
            row.loc[0, "txn_gap"] = max(0.0, tx.step - float(prior_sender[-1]["features"]["step"]))
        stage_times: dict[str, float] = {"feature_engineering": (time.perf_counter() - feature_started) * 1000}
        report("Feature engineering", "complete", stage_times["feature_engineering"])
        component: dict[str, float | None] = {"xgboost": None, "isolation_forest": None, "lstm": None, "hgnn": None}
        errors: dict[str, str] = {}
        stage_started = time.perf_counter()
        report("XGBoost", "running")
        if "xgboost" in self.models:
            try:
                component["xgboost"] = float(self.models["xgboost"].predict_proba(row)[0, 1])
            except Exception as exc:
                errors["xgboost"] = str(exc)
        else:
            errors["xgboost"] = self.status.get("xgboost", "missing model")
        stage_times["xgboost"] = (time.perf_counter() - stage_started) * 1000
        report("XGBoost", "complete" if component["xgboost"] is not None else "unavailable", stage_times["xgboost"], component["xgboost"], errors.get("xgboost"))
        stage_started = time.perf_counter()
        report("Isolation Forest", "running")
        if "isolation_forest" in self.models:
            try:
                raw = np.asarray([[-float(self.models["isolation_forest"].score_samples(row)[0])]])
                scaler = self.models.get("anomaly_scaler")
                component["isolation_forest"] = float(scaler.transform(raw)[0, 0]) if scaler is not None else float(1.0 / (1.0 + np.exp(-raw[0, 0])))
            except Exception as exc:
                errors["isolation_forest"] = str(exc)
        else:
            errors["isolation_forest"] = self.status.get("isolation_forest", "missing model")
        stage_times["isolation_forest"] = (time.perf_counter() - stage_started) * 1000
        report("Isolation Forest", "complete" if component["isolation_forest"] is not None else "unavailable", stage_times["isolation_forest"], component["isolation_forest"], errors.get("isolation_forest"))
        # LSTM receives a deterministic repeated-first-event pad on cold starts.
        stage_started = time.perf_counter()
        report("LSTM", "running")
        seq = self.sender_sequences.setdefault(tx.nameOrig, [])
        lstm_features = [float(row.iloc[0]["amount"]), float(row.iloc[0]["type_enc"]), float(row.iloc[0]["orig_balance_error"]), float(row.iloc[0]["dest_balance_error"]), float(row.iloc[0]["amount_deviation"]), float(row.iloc[0]["txn_gap"])]
        seq.append(lstm_features)
        if len(seq) > 5: del seq[:-5]
        cold_start = len(seq) < 5
        if self.lstm_model is not None:
            try:
                padded = [seq[0]] * (5 - len(seq)) + seq
                scaled = self.lstm_scaler.transform(np.asarray(padded, dtype=np.float32)).astype(np.float32)
                with torch.no_grad():
                    component["lstm"] = float(torch.sigmoid(self.lstm_model(torch.tensor(scaled[None, :, :]))).item())
            except Exception as exc:
                errors["lstm"] = str(exc)
        else:
            errors["lstm"] = self.status.get("lstm", "trained compatible artifact unavailable")
        stage_times["lstm"] = (time.perf_counter() - stage_started) * 1000
        report("LSTM", "complete" if component["lstm"] is not None else "unavailable", stage_times["lstm"], component["lstm"], errors.get("lstm"))
        graph_event = {"features": row.iloc[0].to_dict(), "nameOrig": tx.nameOrig, "nameDest": tx.nameDest, "isFraud": 0}
        self.graph_events.append(graph_event)
        stage_started = time.perf_counter()
        report("HGNN", "running")
        if self.hgnn_model is not None:
            try:
                graph_frame = pd.DataFrame([{**event["features"], "nameOrig": event["nameOrig"], "nameDest": event["nameDest"], "isFraud": 0} for event in self.graph_events])
                graph, _ = build_hetero_graph(graph_frame, FEATURES)
                graph_scores = predict_transaction_scores(self.hgnn_model, graph)
                component["hgnn"] = float(graph_scores[-1])
            except Exception as exc:
                errors["hgnn"] = f"online graph scoring failed: {type(exc).__name__}: {exc}"
        else:
            errors["hgnn"] = self.status.get("hgnn", "trained graph artifact unavailable")
        stage_times["hgnn"] = (time.perf_counter() - stage_started) * 1000
        report("HGNN", "complete" if component["hgnn"] is not None else "unavailable", stage_times["hgnn"], component["hgnn"], errors.get("hgnn"))
        fusion_started = time.perf_counter()
        report("Risk fusion", "running")
        active = [(component[k], w) for k, w in (("xgboost", .40), ("isolation_forest", .15), ("lstm", .25), ("hgnn", .20)) if component[k] is not None]
        if not active:
            raise RuntimeError("No scoring model is available")
        # Weighted active components are renormalized only for an explicitly
        # marked degraded response; this score is not comparable to full fusion.
        risk = float(sum(float(v) * w for v, w in active) / sum(w for _, w in active))
        level = "CRITICAL" if risk >= self.levels["critical"] else "HIGH" if risk >= max(self.threshold, self.levels["high"]) else "MEDIUM" if risk >= self.levels["medium"] else "LOW"
        stage_times["risk_fusion"] = (time.perf_counter() - fusion_started) * 1000
        report("Risk fusion", "complete", stage_times["risk_fusion"], risk)
        shap_started = time.perf_counter()
        report("SHAP", "running")
        explanation = self._explain(row)
        stage_times["shap"] = (time.perf_counter() - shap_started) * 1000
        shap_error = explanation[0].get("error") if explanation else "No explanation returned"
        report("SHAP", "complete" if any(item["contribution"] is not None for item in explanation) else "unavailable", stage_times["shap"], None, shap_error)
        report("Final decision", "complete", None, risk)
        duration = (time.perf_counter() - started) * 1000
        self.last_latency_ms = duration
        self.score_times.append(time.time())
        result = {
            "transaction_id": transaction_id,
            "timestamp": datetime.now(timezone.utc).isoformat(), "amount": tx.amount,
            "type": tx.type, "source_account": tx.nameOrig, "destination_account": tx.nameDest,
            "component_scores": component, "risk_score": risk, "risk_level": level,
            "predicted_label": int(risk >= self.threshold), "actual_label": tx.isFraud,
            "explanation": explanation, "graph_relationship": {"sender": tx.nameOrig, "receiver": tx.nameDest},
            "mode": "DEGRADED_MODEL" if errors else "FULL_MODEL", "model_errors": errors,
            "cold_start": cold_start, "latency_ms": duration,
            "pipeline_stages": [
                {"name": "Transaction received", "status": "complete", "latency_ms": None, "score": None},
                {"name": "Feature engineering", "status": "complete", "latency_ms": stage_times["feature_engineering"], "score": None},
                *[{"name": label, "status": "complete" if component[key] is not None else "unavailable", "latency_ms": stage_times[key], "score": component[key], "error": errors.get(key)} for label, key in (("XGBoost", "xgboost"), ("Isolation Forest", "isolation_forest"), ("LSTM", "lstm"), ("HGNN", "hgnn"))],
                {"name": "Risk fusion", "status": "complete", "latency_ms": stage_times["risk_fusion"], "score": risk},
                {"name": "SHAP", "status": "complete" if any(item["contribution"] is not None for item in explanation) else "unavailable", "latency_ms": stage_times["shap"], "score": None, "error": explanation[0].get("error") if explanation else "No explanation returned"},
                {"name": "Final decision", "status": "complete", "latency_ms": None, "score": risk, "decision": level},
            ],
            "graph_context": {"account_nodes": len({v for event in self.graph_events for v in (event["nameOrig"], event["nameDest"])}), "transaction_nodes": len(self.graph_events), "relationships": 4 * len(self.graph_events)},
            "recent_activity": [{"amount": float(event["features"]["amount"]), "step": float(event["features"]["step"]), "txn_gap": float(event["features"]["txn_gap"])} for event in prior_sender[-4:]] + [{"amount": tx.amount, "step": tx.step, "txn_gap": float(row.iloc[0]["txn_gap"])}],
            "features": row.iloc[0].to_dict(),
        }
        self.history.append(result)
        with sqlite3.connect(self.db_path) as db:
            db.execute("INSERT OR REPLACE INTO transactions VALUES (?, ?, ?)", (result["transaction_id"], result["timestamp"], json.dumps(result)))
            if result["predicted_label"]:
                missing = [k for k, v in component.items() if v is None]
                reason = ", ".join(x["feature"] for x in result["explanation"] if x["contribution"] is not None and x["contribution"] > 0) or "Model risk threshold exceeded"
                alert_id = f"alert-{result['transaction_id']}"
                db.execute("INSERT OR REPLACE INTO alerts VALUES (?, ?, ?, ?, ?, ?, ?, ?)", (alert_id, result["transaction_id"], tx.nameOrig, risk, level, reason, result["timestamp"], json.dumps({"alert_id": alert_id, "transaction_id": result["transaction_id"], "account": tx.nameOrig, "amount": tx.amount, "risk_score": risk, "severity": level, "reason": reason, "timestamp": result["timestamp"], "missing_models": missing})))
        return result

    def _explain(self, row: pd.DataFrame) -> list[dict[str, Any]]:
        model = self.models.get("xgboost")
        try:
            import shap
            values = shap.TreeExplainer(model)(row).values
            contributions = np.asarray(values).reshape(-1, len(FEATURES))[0]
            order = np.argsort(np.abs(contributions))[::-1][:5]
            return [{"feature": FEATURES[i], "value": float(row.iloc[0, i]), "contribution": float(contributions[i]), "direction": "increases" if contributions[i] > 0 else "decreases"} for i in order]
        except Exception as exc:
            detail = f"{type(exc).__name__}: {exc}"
            return [{"feature": FEATURES[i], "value": float(row.iloc[0, i]), "contribution": None, "direction": "unavailable", "error": detail} for i in (2, 7, 8)]

    def next_transaction(self) -> TransactionInput:
        if not self.replay_rows:
            raise RuntimeError("PaySim replay data is unavailable")
        row = self.replay_rows[self.cursor % len(self.replay_rows)]
        self.cursor += 1
        return TransactionInput(transaction_id=str(row.get("txn_id", self.cursor - 1)), **{k: row[k] for k in ("step", "type", "amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest", "nameOrig", "nameDest", "isFraud") if k in row})


service = ScoringService()
app = FastAPI(title="UPI-like Fraud Detection MVP", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5173"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])


@app.get("/api/health")
def health():
    return {"status": "degraded" if any(v is None or str(v).startswith("unavailable") for v in service.status.values()) else "ok", "models": service.status, "simulator": service.running, "latency_ms": service.last_latency_ms, "mode": "PaySim replay/demo; not connected to a bank"}


@app.get("/api/models")
def models():
    return {"models": service.status, "fusion_weights": {"xgboost": .4, "isolation_forest": .15, "lstm": .25, "hgnn": .2}, "threshold": service.threshold}


@app.get("/api/transactions")
def transactions(limit: int = 100):
    return list(service.history)[-max(1, min(limit, 1000)):][::-1]


@app.get("/api/transactions/{transaction_id}")
def transaction_detail(transaction_id: str):
    for item in reversed(service.history):
        if item["transaction_id"] == transaction_id:
            return item
    raise HTTPException(404, "Transaction not found in this process history")


@app.get("/api/explanations/{transaction_id}")
def explanation(transaction_id: str):
    return transaction_detail(transaction_id)["explanation"]


@app.post("/api/transactions/score")
async def score(tx: TransactionInput):
    loop = asyncio.get_running_loop()
    pending: list[Any] = []
    if tx.transaction_id is None:
        tx.transaction_id = f"live-{int(time.time() * 1000)}"

    def on_stage(stage: dict[str, Any]) -> None:
        pending.append(asyncio.run_coroutine_threadsafe(_broadcast({"event": "pipeline_stage", "transaction_id": tx.transaction_id, "stage": stage}), loop))

    try:
        result = await asyncio.to_thread(service.score, tx, on_stage)
        if pending:
            await asyncio.gather(*(asyncio.wrap_future(future) for future in pending))
        return result
    except RuntimeError as exc:
        raise HTTPException(503, str(exc)) from exc


@app.get("/api/metrics")
def metrics():
    rows = list(service.history)
    recent = sum(t >= time.time() - 10 for t in service.score_times)
    with sqlite3.connect(service.db_path) as db:
        transactions_total = db.execute("SELECT COUNT(*) FROM transactions").fetchone()[0]
        alert_total = db.execute("SELECT COUNT(*) FROM alerts").fetchone()[0]
    return {"transactions": transactions_total, "transactions_per_second": recent / 10.0, "fraud_alerts": alert_total, "high_risk_count": sum(r["risk_level"] in {"HIGH", "CRITICAL"} for r in rows), "risk_distribution": {k: sum(r["risk_level"] == k for r in rows) for k in ("LOW", "MEDIUM", "HIGH", "CRITICAL")}, "mean_latency_ms": service.last_latency_ms}


@app.get("/api/alerts")
def alerts(limit: int = 100):
    with sqlite3.connect(service.db_path) as db:
        rows = db.execute("SELECT payload FROM alerts ORDER BY timestamp DESC LIMIT ?", (max(1, min(limit, 500)),)).fetchall()
    return [json.loads(row[0]) for row in rows]


@app.get("/api/accounts")
def accounts():
    grouped: dict[str, list[dict[str, Any]]] = {}
    for item in service.history:
        grouped.setdefault(item["source_account"], []).append(item)
        grouped.setdefault(item["destination_account"], []).append(item)
    summaries = []
    for account, items in grouped.items():
        recent = items[-5:]
        scores = [float(x["risk_score"]) for x in recent]
        weights = list(range(1, len(scores) + 1))
        current_risk = sum(score * weight for score, weight in zip(scores, weights)) / sum(weights) if scores else 0.0
        summaries.append({"account_id": account, "transaction_count": len(items), "total_amount": sum(float(x["amount"]) for x in items), "current_risk": current_risk, "recent_alerts": sum(bool(x["predicted_label"]) for x in recent), "recent_activity": recent})
    return sorted(summaries, key=lambda item: item["current_risk"], reverse=True)[:200]


@app.get("/api/research")
def research():
    result = {}
    for key, filename in (("baselines", "strong_baselines.csv"), ("ablation", "ablation_results.csv"), ("federated", "federated_vs_centralized.csv"), ("cost", "computational_overhead.csv"), ("temporal", "temporal_results.csv"), ("robustness", "robustness_results.csv")):
        path = ROOT / "results" / filename
        try:
            result[key] = pd.read_csv(path).replace({np.nan: None}).to_dict(orient="records")
        except Exception:
            result[key] = []
    result["federated_demo"] = {"clients": 3, "rounds": 5, "local_epochs": 1, "dirichlet_alpha": 0.3, "client_sizes": [42, 13828, 130], "upload_bytes": 1891260, "download_bytes": 1891260, "total_bytes": 3782520, "label": "Simulated Federated Learning Experiment"}
    limitations_path = ROOT / "docs" / "LIMITATIONS.md"
    limitation_text = limitations_path.read_text(encoding="utf-8") if limitations_path.exists() else "Research prototype using synthetic PaySim proxy data. Not connected to UPI, NPCI, banks, or real financial institutions."
    result["limitations"] = [line[2:].strip() for line in limitation_text.splitlines() if line.startswith("- ")] or [limitation_text]
    return result


@app.get("/api/history")
def persisted_transactions(limit: int = 300):
    with sqlite3.connect(service.db_path) as db:
        rows = db.execute("SELECT payload FROM transactions ORDER BY timestamp DESC LIMIT ?", (max(1, min(limit, 1000)),)).fetchall()
    return [json.loads(row[0]) for row in rows]


async def _broadcast(event: dict[str, Any]):
    stale = []
    for ws in service.clients:
        try:
            await ws.send_json(event)
        except Exception:
            stale.append(ws)
    for ws in stale:
        service.clients.discard(ws)


async def _run_simulator():
    try:
        while service.running:
            tx = service.next_transaction()
            try:
                result = await asyncio.to_thread(service.score, tx)
            except RuntimeError as exc:
                await _broadcast({"event": "error", "detail": str(exc)})
                service.running = False
                break
            await _broadcast({"event": "transaction", "data": result})
            await asyncio.sleep(1.0 / service.rate)
    finally:
        service.running = False


@app.post("/api/simulator/start")
async def start_simulator():
    if not service.replay_rows:
        raise HTTPException(503, "PaySim data unavailable")
    if not service.running:
        service.running = True
        service.task = asyncio.create_task(_run_simulator())
    return {"running": service.running, "rate": service.rate, "cursor": service.cursor}


@app.post("/api/simulator/stop")
async def stop_simulator():
    service.running = False
    return {"running": False}


@app.post("/api/simulator/config")
async def simulator_config(config: SimulatorConfig):
    service.rate = config.rate
    return {"rate": service.rate}


@app.websocket("/ws/transactions")
async def websocket_transactions(websocket: WebSocket):
    await websocket.accept()
    service.clients.add(websocket)
    try:
        await websocket.send_json({"event": "status", "running": service.running})
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        service.clients.discard(websocket)
