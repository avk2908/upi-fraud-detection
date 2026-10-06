from pathlib import Path
import os

ROOT = Path(__file__).resolve().parents[1]
SKIP = {".git", "venv", "node_modules", "__pycache__", ".pytest_cache", "data", "models", "outputs", "results"}
TERMS = ["F1 = 0.86", "F1 = 0.88", "C1-C5", "four node types", "federated risk score", "random GNN", "proxy LSTM", "Streamlit dashboard", "device IDs", "merchant IDs", "UPI data"]

def main():
    hits = []
    for base, dirs, files in os.walk(ROOT):
        dirs[:] = [d for d in dirs if d not in SKIP]
        for filename in files:
            path = Path(base) / filename
            if path.suffix.lower() not in {".md", ".py", ".txt", ".csv", ".json"} or path.resolve() == Path(__file__).resolve():
                continue
            try: lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
            except OSError: continue
            for number, line in enumerate(lines, 1):
                for term in TERMS:
                    folded = line.casefold()
                    term_lower = term.casefold()
                    if term_lower not in folded: continue
                    if term_lower in {"device ids", "merchant ids", "upi data"} and any(phrase in folded for phrase in ("not real", "does not contain", "no real", "not actual")):
                        continue
                    hits.append(f"{path.relative_to(ROOT)}:{number}: {term}: {line.strip()}")
    output = "Paper consistency scan (candidate statements require human review)\n\n" + ("\n".join(hits) if hits else "No candidate wording found.") + "\n"
    target = ROOT / "results/paper_consistency_report.txt"
    target.parent.mkdir(exist_ok=True)
    target.write_text(output, encoding="utf-8")
    print(output)

if __name__ == "__main__": main()
