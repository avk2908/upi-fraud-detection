"""
A real heterogeneous GNN for the entities available in PaySim.

PaySim does NOT contain device identifiers or merchant identifiers.
Therefore this implementation does not invent those fields.

Node types:
- account
- transaction

Relations:
- account --sends--> transaction
- transaction --sent_by--> account
- account --receives--> transaction
- transaction --received_by--> account

The transaction node is the prediction target.
"""
import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric import set_home_dir
from torch_geometric.data import HeteroData
from torch_geometric.nn import HeteroConv, GATConv, Linear

# PyG 2.5 generates message-passing templates under its cache home. Keep that
# cache in the project workspace rather than a potentially unwritable user dir.
set_home_dir(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models", ".pyg_cache")))


def build_hetero_graph(df, feature_cols):
    data = HeteroData()

    accounts = pd.Index(
        pd.concat([df["nameOrig"], df["nameDest"]]).astype(str).unique()
    )
    account_map = {v: i for i, v in enumerate(accounts)}

    # Account features: 7 dimensions, constructed without fraud labels.
    sent = df.groupby("nameOrig").agg(
        sent_count=("amount", "count"),
        sent_amount_mean=("amount", "mean"),
        sent_amount_sum=("amount", "sum"),
        unique_receivers=("nameDest", "nunique"),
    )
    recv = df.groupby("nameDest").agg(
        recv_count=("amount", "count"),
        recv_amount_mean=("amount", "mean"),
        unique_senders=("nameOrig", "nunique"),
    )

    account_feat = pd.DataFrame(index=accounts)
    account_feat = account_feat.join(sent).join(recv)
    account_feat = account_feat.fillna(0.0)

    data["account"].x = torch.tensor(
        account_feat.values, dtype=torch.float32
    )

    # Transaction node features.
    tx_feat = df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
    data["transaction"].x = torch.tensor(
        tx_feat.values, dtype=torch.float32
    )
    data["transaction"].y = torch.tensor(
        df["isFraud"].values, dtype=torch.float32
    )

    src_account = torch.tensor(
        [account_map[str(x)] for x in df["nameOrig"]], dtype=torch.long
    )
    dst_account = torch.tensor(
        [account_map[str(x)] for x in df["nameDest"]], dtype=torch.long
    )
    tx_nodes = torch.arange(len(df), dtype=torch.long)

    data["account", "sends", "transaction"].edge_index = torch.stack(
        [src_account, tx_nodes], dim=0
    )
    data["transaction", "sent_by", "account"].edge_index = torch.stack(
        [tx_nodes, src_account], dim=0
    )
    data["account", "receives", "transaction"].edge_index = torch.stack(
        [dst_account, tx_nodes], dim=0
    )
    data["transaction", "received_by", "account"].edge_index = torch.stack(
        [tx_nodes, dst_account], dim=0
    )

    return data, account_map


class PaySimHeteroGNN(nn.Module):
    def __init__(self, tx_dim, account_dim, hidden=32, heads=2):
        super().__init__()

        self.convs = nn.ModuleList([
            HeteroConv({
                ("account", "sends", "transaction"):
                    GATConv((account_dim, tx_dim), hidden, heads=heads, add_self_loops=False),
                ("account", "receives", "transaction"):
                    GATConv((account_dim, tx_dim), hidden, heads=heads, add_self_loops=False),
                ("transaction", "sent_by", "account"):
                    GATConv((tx_dim, account_dim), hidden, heads=heads, add_self_loops=False),
                ("transaction", "received_by", "account"):
                    GATConv((tx_dim, account_dim), hidden, heads=heads, add_self_loops=False),
            }, aggr="sum"),
            HeteroConv({
                ("account", "sends", "transaction"):
                    GATConv((hidden * heads, hidden * heads), hidden, heads=1, add_self_loops=False),
                ("account", "receives", "transaction"):
                    GATConv((hidden * heads, hidden * heads), hidden, heads=1, add_self_loops=False),
                ("transaction", "sent_by", "account"):
                    GATConv((hidden * heads, hidden * heads), hidden, heads=1, add_self_loops=False),
                ("transaction", "received_by", "account"):
                    GATConv((hidden * heads, hidden * heads), hidden, heads=1, add_self_loops=False),
            }, aggr="sum"),
        ])
        self.out = Linear(hidden, 1)

    def forward(self, x_dict, edge_index_dict):
        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {
                k: F.elu(v) for k, v in x_dict.items()
            }
        return self.out(x_dict["transaction"]).squeeze(-1)


def train_hetero_gnn(
    data,
    train_idx,
    epochs=30,
    lr=1e-3,
    weight_decay=1e-4,
    seed=42,
):
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = data.to(device)

    model = PaySimHeteroGNN(
        tx_dim=data["transaction"].x.shape[1],
        account_dim=data["account"].x.shape[1],
    ).to(device)

    y = data["transaction"].y
    pos = y[train_idx].sum().item()
    neg = len(train_idx) - pos
    pos_weight = torch.tensor([neg / max(pos, 1.0)], device=device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )

    train_idx = torch.tensor(train_idx, dtype=torch.long, device=device)

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        logits = model(data.x_dict, data.edge_index_dict)
        loss = criterion(logits[train_idx], y[train_idx])
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print(f"[HGNN] epoch={epoch:02d} loss={loss.item():.4f}")

    return model, data


def predict_transaction_scores(model, data):
    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        logits = model(data.x_dict, data.edge_index_dict)
        return torch.sigmoid(logits).detach().cpu().numpy()
