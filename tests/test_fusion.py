import numpy as np
from src.risk_fusion import fuse_risk_scores

def test_fusion_uses_four_components_and_fixed_scales():
    b=np.array([1.0]); a=np.array([0.0]); s=np.array([0.5]); g=np.array([0.25])
    np.testing.assert_allclose(fuse_risk_scores(b,a,s,g), [0.40+0.125+0.05])
