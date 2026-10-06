from src.preprocess import get_feature_cols
from xgboost import XGBClassifier

def test_behavioral_input_and_model_parameters():
    assert len(get_feature_cols()) == 18
    model = XGBClassifier(n_estimators=80, max_depth=4, learning_rate=0.05, random_state=42)
    assert model.get_params()["n_estimators"] == 80
