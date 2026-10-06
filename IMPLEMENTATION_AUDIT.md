# Implementation audit

## Prior state

- The original dashboard fabricated random risk values when models failed and derived an LSTM-like component from XGBoost/Isolation Forest scores. The old Streamlit application has been removed from the supported project.
- The first supplied upgrade added a real LSTM training routine, a heterogeneous account/transaction GNN, and local FedAvg simulation, but left an 80/20 pipeline and selected the threshold on test data.
- Prior model files and result tables were not produced by the revised experiment and must not be cited for it.

## Current verified implementation

- PaySim preprocessor: 18 behavioral features, transaction IDs retained, sender/receiver histories use past events only.
- Primary evaluation: 20,000 PaySim rows, seed 42, stratified 70/15/15 transaction split. `results/splits.csv` records IDs/labels. Threshold selected from validation; test is used only for final metrics.
- Behavioral models: XGBoost and Isolation Forest fit on training data. XGBoost uses class weighting; Isolation Forest uses the training-only fit and score scaler.
- Temporal model: actual 5-by-6 LSTM sequences, train-only scaler, class-weighted loss, deterministic repeat-first-event cold start. Centralized and three-client FedAvg checkpoints were generated.
- Graph model: two-node-type account/transaction HeteroGNN with four PaySim relations was trained for 30 epochs. Evaluation is transductive because held-out graph structure is visible.
- FedAvg is a training strategy and is compared against centralized LSTM; it is not a risk score. This local simulation does not prove production privacy.
- Superseded TensorFlow LSTM and homogeneous GAT source modules were removed. Legacy `.h5`/homogeneous GNN artifacts may remain in `models/` for reference but are not loaded or used by the revised pipeline.
- Full fusion is `0.40 B + 0.15 A + 0.25 S + 0.20 G`. Risk thresholds are selected on validation.
- FastAPI REST/WebSocket service and React/Vite frontend are implemented. The backend has XGBoost, Isolation Forest, LSTM, and rolling-window HGNN inference adapters. It marks the system degraded if a compatible model or graph score is unavailable.
- Training ran successfully on the configured first 20,000 rows. Fresh outputs are under `results/`; the chronological track retrains XGBoost on exactly 70/15/15 row-ordered partitions.
- PaySim is synthetic mobile-money data used as a UPI-like proxy. It has no real UPI IDs, devices, merchants, or real identities. Stress tests are dataset-derived proxies, not adaptive attack evidence.

## Verification

- Python suite: 11 passed.
- FastAPI health and scoring smoke checks returned HTTP 200; scoring reports degraded status for unavailable HGNN.
- Vite production build succeeded. WebSocket/simulator flow is covered by an automated TestClient test.

See `docs/EXPERIMENT_PROTOCOL.md`, `docs/REALTIME_APPLICATION.md`, and `docs/LIMITATIONS.md`.
