import pandas as pd
from src.preprocess import get_feature_cols, get_model_matrix

def test_feature_dimensions_and_order():
    assert len(get_feature_cols()) == 18
    frame = pd.DataFrame([{c: 0 for c in get_feature_cols()}])
    assert get_model_matrix(frame).shape == (1, 18)
