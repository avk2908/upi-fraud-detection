import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GATConv, global_mean_pool
from torch_geometric.transforms import ToUndirected
import os

class FraudGAT(torch.nn.Module):
    """Graph Attention Network for fraud ring detection."""
    def __init__(self, in_channels: int, hidden: int = 64, heads: int = 4):
        super().__init__()
        self.gat1 = GATConv(in_channels, hidden, heads=heads, dropout=0.3)
        self.gat2 = GATConv(hidden * heads, hidden, heads=1, concat=False, dropout=0.3)
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(hidden, 32),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(32, 1)
        )

    def forward(self, x, edge_index, edge_attr=None):
        x = F.dropout(x, p=0.3, training=self.training)
        x = F.elu(self.gat1(x, edge_index))
        x = F.dropout(x, p=0.3, training=self.training)
        x = self.gat2(x, edge_index)
        return self.classifier(x).squeeze(-1)


def build_graph_from_paysim(df: pd.DataFrame, raw_csv: str) -> Data:
    """
    Build a transaction graph:
    - Nodes: unique senders + receivers
    - Edges: transactions (directed sender → receiver)
    - Node features: aggregated behavioral stats
    - Edge features: amount, type, fraud label
    """
    print("[Phase 4] Building transaction graph...")
    raw = pd.read_csv(
    raw_csv,
    usecols=["nameOrig", "nameDest"],
    nrows=len(df)
)
    df = df.copy()
    df["nameOrig"] = raw["nameOrig"].values
    df["nameDest"] = raw["nameDest"].values

    # Node index mapping
    all_nodes = pd.concat([df["nameOrig"], df["nameDest"]]).unique()
    node_map = {n: i for i, n in enumerate(all_nodes)}
    N = len(all_nodes)
    print(f"[Phase 4] Nodes: {N}, Edges: {len(df)}")

    # Node features: aggregate transaction stats per entity
    agg = {}
    for node in all_nodes:
        sent = df[df["nameOrig"] == node]
        recv = df[df["nameDest"] == node]
        agg[node] = [
            sent["amount"].mean() if len(sent) else 0,
            sent["amount"].std() if len(sent) > 1 else 0,
            len(sent),                          # out-degree
            len(recv),                          # in-degree
            sent["isFraud"].mean() if len(sent) else 0,  # fraud rate as sender
            recv["isFraud"].mean() if len(recv) else 0,  # fraud rate as receiver
        ]

    node_feat = torch.tensor(
        [agg[n] for n in all_nodes], dtype=torch.float
    )

    # Edge index
    src = [node_map[n] for n in df["nameOrig"]]
    dst = [node_map[n] for n in df["nameDest"]]
    edge_index = torch.tensor([src, dst], dtype=torch.long)

    # Node fraud labels (node is "fraud" if any edge from it is fraud)
    node_fraud = torch.zeros(N, dtype=torch.float)
    for i, row in df.iterrows():
        if row["isFraud"] == 1:
            node_fraud[node_map[row["nameOrig"]]] = 1

    data = Data(x=node_feat, edge_index=edge_index, y=node_fraud)
    print("[Phase 4] Graph built.")
    return data, node_map


def train_gnn(data: Data, model_dir: str = "models", epochs: int = 50):
    os.makedirs(model_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Phase 4] Training GAT on {device}...")

    model = FraudGAT(in_channels=data.x.shape[1]).to(device)
    data = data.to(device)

    # Train/test mask (80/20 on nodes)
    N = data.num_nodes
    perm = torch.randperm(N)
    train_mask = torch.zeros(N, dtype=torch.bool)
    train_mask[perm[:int(0.8 * N)]] = True
    test_mask = ~train_mask

    # Class weight for imbalance
    fraud_ratio = data.y.mean().item()
    pos_weight = torch.tensor([(1 - fraud_ratio) / (fraud_ratio + 1e-9)]).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_loss = float("inf")
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = criterion(out[train_mask], data.y[train_mask])
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            model.eval()
            with torch.no_grad():
                val_out = torch.sigmoid(model(data.x, data.edge_index))
                val_pred = (val_out[test_mask] > 0.5).float()
                acc = (val_pred == data.y[test_mask]).float().mean()
            print(f"  Epoch {epoch:3d} | Loss: {loss.item():.4f} | Test Acc: {acc:.4f}")

        if loss.item() < best_loss:
            best_loss = loss.item()
            torch.save(model.state_dict(), f"{model_dir}/gnn_model.pt")

    print("[Phase 4] GNN saved.")
    return model, data


def get_gnn_scores(model, data, device="cpu"):
    model.eval()
    data = data.to(device)
    with torch.no_grad():
        scores = torch.sigmoid(model(data.x, data.edge_index)).cpu().numpy()
    return scores  # per-node fraud probability