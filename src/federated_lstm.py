"""
Small, reproducible FedAvg simulation for the PaySim MVP.

Design:
- 3 simulated institutional clients
- non-IID label distribution using Dirichlet partitioning
- 5 communication rounds
- 1 local epoch per round
- weighted FedAvg by local sample count

Important:
This simulates the federated protocol on one machine. It does NOT prove
production privacy by itself.
"""
import copy
import time
import numpy as np
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader


class LSTMClassifier(nn.Module):
    def __init__(self, input_dim, hidden1=64, hidden2=32, dropout=0.30):
        super().__init__()
        self.lstm1 = nn.LSTM(input_dim, hidden1, batch_first=True)
        self.lstm2 = nn.LSTM(hidden1, hidden2, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden2, 16)
        self.fc2 = nn.Linear(16, 1)

    def forward(self, x):
        x, _ = self.lstm1(x)
        x = self.dropout(x)
        x, _ = self.lstm2(x)
        x = self.dropout(x[:, -1, :])
        x = torch.relu(self.fc1(x))
        return self.fc2(x).squeeze(-1)


def make_non_iid_partitions(y, n_clients=3, alpha=0.30, seed=42):
    """
    Dirichlet partitioning. Lower alpha => stronger heterogeneity.
    Guarantees every client gets at least one example.
    """
    rng = np.random.default_rng(seed)
    y = np.asarray(y).astype(int)
    idx_by_class = [np.where(y == c)[0] for c in np.unique(y)]

    client_indices = [[] for _ in range(n_clients)]
    for idx in idx_by_class:
        rng.shuffle(idx)
        proportions = rng.dirichlet(np.full(n_clients, alpha))
        cuts = (np.cumsum(proportions) * len(idx)).astype(int)[:-1]
        splits = np.split(idx, cuts)
        for k, part in enumerate(splits):
            client_indices[k].extend(part.tolist())

    # If a pathological split leaves a client empty, redistribute examples.
    for k in range(n_clients):
        if not client_indices[k]:
            donor = max(range(n_clients), key=lambda j: len(client_indices[j]))
            client_indices[k].append(client_indices[donor].pop())

    return [np.asarray(sorted(v), dtype=np.int64) for v in client_indices]


def _train_local(model, X, y, indices, device, epochs=1, batch_size=256, lr=1e-3):
    local = copy.deepcopy(model).to(device)
    local.train()

    ds = TensorDataset(
        torch.tensor(X[indices], dtype=torch.float32),
        torch.tensor(y[indices], dtype=torch.float32),
    )
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)

    pos = max(float(y[indices].sum()), 1.0)
    neg = max(float(len(indices) - y[indices].sum()), 1.0)
    pos_weight = torch.tensor([neg / pos], device=device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(local.parameters(), lr=lr)

    for _ in range(epochs):
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(local(xb), yb)
            loss.backward()
            optimizer.step()

    return local.state_dict(), len(indices)


def fedavg(state_dicts, weights):
    total = float(sum(weights))
    out = copy.deepcopy(state_dicts[0])
    for key in out:
        out[key] = sum(
            state_dicts[i][key].float() * (weights[i] / total)
            for i in range(len(state_dicts))
        )
    return out


def train_federated_lstm(
    X_train, y_train,
    input_dim,
    n_clients=3,
    rounds=5,
    local_epochs=1,
    alpha=0.30,
    seed=42,
    device=None,
):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    partitions = make_non_iid_partitions(
        y_train, n_clients=n_clients, alpha=alpha, seed=seed
    )

    global_model = LSTMClassifier(input_dim).to(device)
    history = []
    total_bytes = 0

    for rnd in range(1, rounds + 1):
        start = time.perf_counter()
        states, sizes, local_seconds = [], [], []

        for idx in partitions:
            local_start = time.perf_counter()
            state, n = _train_local(
                global_model, X_train, y_train, idx, device,
                epochs=local_epochs
            )
            states.append(state)
            sizes.append(n)
            local_seconds.append(time.perf_counter() - local_start)

        aggregation_start = time.perf_counter()
        global_model.load_state_dict(fedavg(states, sizes))
        aggregation_seconds = time.perf_counter() - aggregation_start
        elapsed = time.perf_counter() - start

        round_bytes = sum(
            tensor.numel() * tensor.element_size()
            for state in states
            for tensor in state.values()
            if torch.is_tensor(tensor)
        )
        # Approximate client->server update payload only.
        total_bytes += round_bytes
        model_bytes = sum(t.numel() * t.element_size() for t in global_model.state_dict().values())

        history.append({
            "round": rnd,
            "seconds": elapsed,
            "client_sizes": [int(v) for v in sizes],
            "client_training_seconds": local_seconds,
            "aggregation_seconds": aggregation_seconds,
            "upload_bytes": int(round_bytes),
            "download_bytes": int(model_bytes * len(partitions)),
            "model_update_bytes": int(model_bytes),
        })

    return global_model, partitions, history, total_bytes


def train_centralized_lstm(
    X_train, y_train, input_dim, epochs=5, batch_size=256, seed=42, device=None
):
    """Matched centralized LSTM used only for the FL ablation."""
    torch.manual_seed(seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTMClassifier(input_dim).to(device)

    ds = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32),
    )
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)

    pos = max(float(y_train.sum()), 1.0)
    neg = max(float(len(y_train) - y_train.sum()), 1.0)
    pos_weight = torch.tensor([neg / pos], device=device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    model.train()
    for _ in range(epochs):
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()

    return model
