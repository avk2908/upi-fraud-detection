# Experiment protocol

PaySim is synthetic mobile-money data used here as a UPI-like ecosystem proxy. The configured training run loads the first 20,000 rows unless `NROWS` is changed. The primary protocol is a seed-42 stratified transaction split: 70% train, 15% validation, 15% test. `results/splits.csv` is the source of truth for transaction IDs. Validation labels select thresholds; test labels are reserved for the final report.

Feature matrix: 18 columns. LSTM sequence: five events by six features. History features are computed from prior sender/receiver events only. Before five sender events exist, the sequence is left-padded by repeating that sender's first observed event; inference flags this cold-start condition. GNN node types are account (7 features) and transaction (18 features), connected by sends/sent_by and receives/received_by relations. The current graph uses the available transaction graph and is transductive; future-edge visibility is a limitation.

XGBoost uses 80 trees, depth 4, learning rate .05, AUC-PR evaluation and training-set class weighting. Isolation Forest uses 200 estimators, 0.02 contamination and seed 42. LSTM uses hidden sizes 64/32, dropout .30, dense size 16 and weighted BCE. FedAvg uses 3 simulated clients, Dirichlet alpha .30, 5 rounds, 1 local epoch, sample-count weighting. These are local simulations; they do not establish production privacy.

For the 20,000-row run the split class counts are train 13,943 legitimate / 57 fraud, validation 2,988 / 12, and test 2,988 / 12. The project uses weighting rather than oversampling, so the post-balancing row counts are unchanged; no validation/test row is resampled.

Always regenerate result files after code changes. Existing CSVs and model files must not be treated as results from a newer architecture without rerunning training.
