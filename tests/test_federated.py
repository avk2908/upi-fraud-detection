import numpy as np
from src.federated_lstm import make_non_iid_partitions

def test_federated_partition_is_complete_and_repeatable():
    y=np.array([0]*40+[1]*12)
    a=make_non_iid_partitions(y,seed=42); b=make_non_iid_partitions(y,seed=42)
    assert len(a)==3 and sorted(np.concatenate(a).tolist())==list(range(len(y)))
    assert all(np.array_equal(x,z) for x,z in zip(a,b))
