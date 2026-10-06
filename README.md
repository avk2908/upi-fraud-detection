# Explainable Federated Graph-Enhanced Fraud Detection

Research MVP for fraud analysis on PaySim, a **synthetic mobile-money dataset used as a UPI-like proxy**. It is not real UPI data and does not contain real UPI IDs, devices, merchants, or customer identities. The web app replays PaySim rows; it is not connected to a bank or payment rail.

## Architecture

- `src/`: preprocessing, XGBoost and Isolation Forest, LSTM/FedAvg, heterogeneous account/transaction GNN, risk fusion and evaluation.
- `backend/`: FastAPI REST/WebSocket API, model health, scoring, and sequential PaySim replay.
- `frontend/`: React, TypeScript, and Vite fraud-intelligence interface with six workspace pages.
- `scripts/`: paper consistency scan and generated research tables.
- `docs/`: experiment protocol, paper guidance, audit and limitations.

The behavioral matrix has 18 features. The temporal model consumes length-5 sequences with six features. The graph has account nodes (7 features), transaction nodes (18 features), and sends/sent_by/receives/received_by relations. The fusion equation is `0.40 B + 0.15 A + 0.25 S + 0.20 G`; FedAvg trains the LSTM and is not a score.

## Setup

Use Python 3.10 or 3.11 for compatibility with the pinned ML stack.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Place the PaySim CSV at `data/PS_20174392719_1491204439457_log.csv` or set `PAYSIM_CSV`. The repository's configured dataset path uses the first 20,000 rows for the research MVP. To train, run `python train_pipeline.py`; outputs are written to `models/` and `results/`. The primary split is seed-42 stratified 70/15/15 and is persisted in `results/splits.csv`. Validation selects the decision threshold; test is final evaluation only.

## Full-stack demo

Start the backend:

```powershell
python -m uvicorn backend.app:app --reload
```

Start the frontend in another terminal:

```powershell
cd frontend
npm install
npm run dev
```

Open the Vite URL (normally `http://localhost:5173`). The API is available at `http://localhost:8000/docs`; the UI receives scored replay events through `/ws/transactions`. The Live Monitor supports replay at 0.5x, 1x, 2x, and 5x. The Payment Simulator submits six controlled scenarios to the actual inference pipeline. Transactions and threshold alerts persist in `outputs/fraud_monitor.sqlite3`. The interface displays unavailable model components and degraded scoring explicitly; it never substitutes random scores.

To check the frontend types and production bundle:

```powershell
cd frontend
npm run typecheck
npm run build
```

## Evaluation and reports

```powershell
pytest
python scripts/generate_paper_tables.py
python scripts/check_paper_consistency.py
```

The training pipeline writes split, evaluation, baseline, ablation, stress, federated, and overhead CSVs under `results/`. The reporting scripts generate `results/paper_tables.md`, `results/paper_tables.csv`, and `results/paper_consistency_report.txt`. Existing results and model files are not valid for revised code until the pipeline is rerun.

## Limitations

The graph experiment is transductive. The FedAvg setup is a local protocol simulation, not a privacy guarantee. Stress tests are dataset-derived proxies, not adaptive attacks. SHAP is decision-support, not proof of regulatory compliance. See [the experiment protocol](docs/EXPERIMENT_PROTOCOL.md), [application guide](docs/REALTIME_APPLICATION.md), and [limitations](docs/LIMITATIONS.md).
