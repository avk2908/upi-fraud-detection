from src.hetero_gnn_model import PaySimHeteroGNN

def test_hgnn_architecture_constructs():
    model=PaySimHeteroGNN(tx_dim=18,account_dim=7)
    assert len(model.convs)==2
