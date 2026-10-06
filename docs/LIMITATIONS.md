# Limitations

- PaySim is synthetic mobile-money data, not real UPI data and contains no real UPI IDs, device IDs, merchant IDs, or real customer identities.
- The graph model is transductive and may observe held-out graph structure. A chronological inductive graph evaluation remains necessary.
- FedAvg is a single-machine simulation, not evidence of privacy. Deployment needs secure aggregation, authenticated clients, update validation, poisoning defenses, possible differential privacy, secure transport, key management, and audit logging.
- Stress subsets are dataset-derived proxies, not adaptive-attacker experiments.
- SHAP is decision-support and audit support; it does not establish regulatory compliance.
- The web demo builds an online rolling graph from recent replay events. This graph context differs from the full transductive training graph and needs separate online calibration. Unavailable components are exposed in degraded status, never replaced with random predictions.
