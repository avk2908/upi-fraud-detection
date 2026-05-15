# 🔐 UPI Fraud Shield

## Predictive Cognitive Fraud Prevention for UPI-Style Payment Ecosystems

An advanced multi-layer fraud detection system built using Machine Learning, Deep Learning, Graph Neural Networks, and Explainable AI to detect suspicious UPI-style financial transactions in real time.

This project combines behavioral analytics, anomaly detection, sequential intelligence, graph-based fraud ring detection, and explainable AI into a unified fraud prevention framework.

---

# 🚀 Project Overview

Digital payment systems such as UPI are experiencing rapid growth, but with increasing transaction volume comes increasingly sophisticated fraud.

Traditional rule-based systems struggle to detect:

* evolving fraud patterns
* coordinated fraud rings
* behavioral anomalies
* sequential fraud escalation
* previously unseen attack patterns

This project addresses those challenges using a hybrid AI architecture that integrates:

* XGBoost for behavioral fraud classification
* Isolation Forest for anomaly detection
* LSTM for sequential transaction intelligence
* Graph Attention Networks (GNN/GAT) for fraud ring detection
* SHAP for explainable AI
* Streamlit for interactive fraud monitoring dashboards

---

# 🧠 Key Features

✅ Multi-layer fraud intelligence pipeline
✅ Real-time transaction risk scoring
✅ Hybrid supervised + unsupervised learning
✅ Sequential fraud pattern detection using LSTM
✅ Graph Neural Network for fraud ring detection
✅ Explainable AI using SHAP
✅ Interactive Streamlit dashboard
✅ Risk threshold tuning and analyst simulation
✅ Fraud analytics and visualization

---

# 🏗️ System Architecture

```text
                 ┌────────────────────┐
                 │   PaySim Dataset   │
                 └─────────┬──────────┘
                           │
                 ┌─────────▼──────────┐
                 │ Preprocessing &    │
                 │ Feature Engineering│
                 └─────────┬──────────┘
                           │
      ┌────────────────────┼────────────────────┐
      │                    │                    │
┌─────▼─────┐      ┌───────▼───────┐    ┌──────▼──────┐
│ XGBoost   │      │ Isolation     │    │ LSTM Model  │
│ Behavioral│      │ Forest        │    │ Sequential  │
│ Engine    │      │ Anomaly Layer │    │ Intelligence│
└─────┬─────┘      └───────┬───────┘    └──────┬──────┘
      │                    │                    │
      └────────────────────┼────────────────────┘
                           │
                 ┌─────────▼──────────┐
                 │ Graph Neural       │
                 │ Network (GAT)      │
                 └─────────┬──────────┘
                           │
                 ┌─────────▼──────────┐
                 │ Risk Fusion Engine │
                 └─────────┬──────────┘
                           │
                 ┌─────────▼──────────┐
                 │ SHAP Explainability│
                 └─────────┬──────────┘
                           │
                 ┌─────────▼──────────┐
                 │ Streamlit Dashboard│
                 └────────────────────┘
```

---

# 📂 Project Structure

```text
upi_fraud_detection/
│
├── data/
│   └── PaySim Dataset CSV
│
├── src/
│   ├── preprocess.py
│   ├── behavioral_engine.py
│   ├── lstm_model.py
│   ├── gnn_model.py
│   ├── risk_fusion.py
│   └── explainability.py
│
├── dashboard/
│   └── app.py
│
├── models/
├── outputs/
├── train_pipeline.py
├── requirements.txt
└── README.md
```

---

# ⚙️ Tech Stack

| Category             | Technologies                |
| -------------------- | --------------------------- |
| Programming Language | Python                      |
| Data Processing      | Pandas, NumPy               |
| Machine Learning     | Scikit-learn, XGBoost       |
| Deep Learning        | TensorFlow, Keras           |
| Graph Learning       | PyTorch Geometric           |
| Explainable AI       | SHAP                        |
| Visualization        | Plotly, Matplotlib, Seaborn |
| Dashboard            | Streamlit                   |
| Graph Analytics      | NetworkX                    |

---

# 📊 Dataset

### PaySim Dataset

This project uses the PaySim synthetic financial transaction dataset from Kaggle.

Dataset Link:
[https://www.kaggle.com/datasets/ealaxi/paysim1](https://www.kaggle.com/datasets/ealaxi/paysim1)

The dataset simulates mobile money transactions and includes:

* transaction type
* sender/receiver balances
* transaction amounts
* fraud labels
* account behavior patterns

---

# 🔍 Fraud Detection Pipeline

## Phase 1 — Preprocessing & Feature Engineering

Engineered fraud-sensitive features such as:

* balance mismatch signals
* transaction velocity
* amount deviation
* transaction gaps
* receiver novelty
* beneficiary concentration

---

## Phase 2 — Behavioral Fraud Engine

### XGBoost

Learns transaction-level fraud behavior patterns.

### Isolation Forest

Detects anomalous or previously unseen suspicious behavior.

---

## Phase 3 — Sequential Intelligence (LSTM)

Captures temporal transaction patterns and sequential fraud escalation.

Example:

```text
Txn1 → Txn2 → Txn3 → Suspicious Escalation
```

---

## Phase 4 — Graph Neural Network (GAT)

Models transaction ecosystems as graphs:

* Nodes → accounts/users
* Edges → transactions

Detects:

* fraud rings
* mule accounts
* coordinated attacks
* suspicious transaction hubs

This is the core research contribution of the project.

---

## Phase 5 — Risk Fusion Engine

Combines all module outputs into a unified fraud risk score.

```text
Final Risk = Behavioral + Anomaly + Sequential + Relational
```

---

## Phase 6 — Explainable AI

SHAP explanations provide feature-level transparency for every fraud prediction.

This helps analysts understand:

* why a transaction was flagged
* which features contributed most
* how risk was determined

---

# 📈 Dashboard Features

The Streamlit dashboard provides:

* fraud monitoring KPIs
* risk score visualization
* transaction-level explainability
* graph analytics
* threshold tuning
* suspicious transaction exploration

---

# 🖥️ Installation

## 1. Clone Repository

```bash
git clone https://github.com/yourusername/upi-fraud-detection.git
cd upi-fraud-detection
```

---

## 2. Create Virtual Environment

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### Linux / macOS

```bash
python3 -m venv venv
source venv/bin/activate
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 4. Download Dataset

Download the PaySim dataset and place the CSV inside:

```text
data/
```

---

# ▶️ Running The Project

## Train Models

```bash
python train_pipeline.py
```

---

## Launch Dashboard

```bash
streamlit run dashboard/app.py
```

Dashboard opens at:

```text
http://localhost:8501
```

---

# 📸 Sample Dashboard Modules

* Fraud Monitoring Dashboard
* Risk Score Distribution
* Transaction Explainability
* Graph Analytics
* Mule Account Detection

---

---

# 🔮 Future Scope

* Real-time streaming fraud detection
* Kafka-based event pipelines
* Live API integration
* Reinforcement learning for adaptive fraud response
* Federated learning for privacy-preserving fraud detection
* Advanced graph embedding techniques
* Cloud deployment and scalable microservices

---

# 📚 References

* XGBoost Documentation
* TensorFlow Documentation
* PyTorch Geometric Documentation
* SHAP Explainability
* PaySim Dataset Research Paper

---

# 👨‍💻 Author

Amithava Varma
AI Engineering Student
Focused on AI Systems, Machine Learning, Fraud Intelligence, and Real-World AI Applications.

---
