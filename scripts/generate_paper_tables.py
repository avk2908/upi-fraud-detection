from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"

def read(name):
    path = RESULTS / name
    return pd.read_csv(path) if path.exists() else pd.DataFrame()

def markdown_table(frame):
    values = [[str(v) for v in row] for row in frame.fillna("").itertuples(index=False, name=None)]
    headers = [str(c) for c in frame.columns]
    return "| " + " | ".join(headers) + " |\n| " + " | ".join(["---"] * len(headers)) + " |\n" + "\n".join("| " + " | ".join(row) + " |" for row in values)

def main():
    configs = pd.DataFrame([
        ["Behavioral input", 18, "XGBoost; Isolation Forest"],
        ["LSTM sequence", "5 × 6", "amount, type, balance errors, amount deviation, transaction gap"],
        ["HGNN", "account: 7; transaction: 18", "PaySim account and transaction entities"],
    ], columns=["Configuration", "Dimensions", "Details"])
    groups = [("Table 1. Feature dimensions and configuration", configs), ("Table 2. Strong baselines", read("strong_baselines.csv")), ("Table 3. Ablation study", read("ablation_results.csv")), ("Table 4. Centralized vs federated LSTM", read("federated_vs_centralized.csv")), ("Table 5. Temporal evaluation", read("temporal_results.csv")), ("Table 6. Robustness stress tests", read("robustness_results.csv")), ("Table 7. Computational and communication overhead", read("computational_overhead.csv"))]
    md=[]; csv_frames=[]
    for title, frame in groups:
        md += [f"## {title}\n", "_No experiment output available yet._\n" if frame.empty else markdown_table(frame) + "\n"]
        if not frame.empty:
            frame=frame.copy(); frame.insert(0,"table",title); csv_frames.append(frame)
    (RESULTS/"paper_tables.md").write_text("\n".join(md),encoding="utf-8")
    pd.concat(csv_frames,ignore_index=True,sort=False).to_csv(RESULTS/"paper_tables.csv",index=False) if csv_frames else pd.DataFrame(columns=["table"]).to_csv(RESULTS/"paper_tables.csv",index=False)

if __name__ == "__main__": main()
