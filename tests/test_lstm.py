import torch
from src.federated_lstm import LSTMClassifier

def test_lstm_real_deterministic_shape():
    torch.manual_seed(42); model=LSTMClassifier(6).eval(); x=torch.zeros(2,5,6)
    with torch.no_grad(): a=model(x); b=model(x)
    assert a.shape == (2,)
    assert torch.equal(a,b)
