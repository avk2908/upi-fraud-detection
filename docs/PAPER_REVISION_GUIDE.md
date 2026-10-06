# Paper revision guide

Report PaySim as a synthetic mobile-money proxy for UPI-like payments, never as real UPI evidence. Describe the graph as two node types (account and transaction). Report FedAvg as the LSTM training protocol and compare it with centralized LSTM. Do not add FedAvg as a fifth risk score.

Use only freshly generated result files from the current training revision. Select thresholds using validation data. The offline graph evaluation is transductive; the online demo builds a rolling graph from recent replay events, which differs from the training graph and needs separate calibration. The API marks scoring degraded if any required model artifact or adapter is unavailable. Dataset-derived high-value, novel-beneficiary, and amount-deviation subsets are stress proxies, not proof of adaptive attack resilience. SHAP supports analyst review but does not prove regulatory compliance.
