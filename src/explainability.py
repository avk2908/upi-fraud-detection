import shap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

def generate_shap_explanations(xgb_model, X_sample: pd.DataFrame, output_dir: str = "outputs"):
    os.makedirs(output_dir, exist_ok=True)
    print("[Phase 6] Computing SHAP values...")

    explainer = shap.TreeExplainer(xgb_model)
    shap_values = explainer.shap_values(X_sample)

    # Summary plot
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_sample, show=False, max_display=15)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/shap_summary.png", dpi=150)
    plt.close()
    print(f"[Phase 6] SHAP summary saved → {output_dir}/shap_summary.png")

    # Waterfall for top fraud case
    fraud_mask = xgb_model.predict_proba(X_sample)[:, 1] > 0.8
    if fraud_mask.any():
        idx = np.where(fraud_mask)[0][0]
        shap.plots._waterfall.waterfall_legacy(
            explainer.expected_value,
            shap_values[idx],
            feature_names=X_sample.columns.tolist(),
            show=False
        )
        plt.tight_layout()
        plt.savefig(f"{output_dir}/shap_waterfall_example.png", dpi=150)
        plt.close()
        print(f"[Phase 6] SHAP waterfall saved.")

    return shap_values