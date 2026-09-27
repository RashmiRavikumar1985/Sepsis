import os
import sys
import json
import subprocess
import time
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def run_script(script_path, description):
    print(f"[{time.strftime('%H:%M:%S')}] RUNNING: {description}")
    try:
        start_time = time.time()
        result = subprocess.run([sys.executable, script_path], check=True, capture_output=True, text=True)
        end_time = time.time()
        print(f"[{time.strftime('%H:%M:%S')}] SUCCESS ({end_time - start_time:.1f}s)")
        return {"status": "SUCCESS", "runtime": end_time - start_time, "error": None}
    except subprocess.CalledProcessError as e:
        end_time = time.time()
        print(f"[{time.strftime('%H:%M:%S')}] FAILED ({end_time - start_time:.1f}s)")
        print(e.stderr)
        return {"status": "FAILED", "runtime": end_time - start_time, "error": e.stderr}

def compare_models(root):
    """Load metrics from their actual artifact locations and produce a comparison report.

    Artifact locations:
      GRU-D baseline : root/checkpoints/training_log_full.json  (key: test_results)
      Transformer baseline: root/artifacts/baseline_metrics.json
      Transformer tuned   : root/artifacts/tuned_metrics.json
      GAT                 : root/experiments/results/gat/metrics.json
    """
    print("=== Generating Model Comparison ===")
    comp_dir = os.path.join(root, "experiments", "results", "comparison")
    os.makedirs(comp_dir, exist_ok=True)

    # Map model display name -> (path, optional sub-key)
    sources = [
        ("GRU-D",
         os.path.join(root, "checkpoints", "training_log_full.json"),
         "test_results"),
        ("Temporal Transformer (baseline)",
         os.path.join(root, "artifacts", "baseline_metrics.json"),
         None),
        ("Temporal Transformer (tuned)",
         os.path.join(root, "artifacts", "tuned_metrics.json"),
         None),
        ("GAT (Graph Baseline)",
         os.path.join(root, "experiments", "results", "gat", "metrics.json"),
         None),
    ]

    metrics_list = []
    for model_name, metrics_file, sub_key in sources:
        if os.path.exists(metrics_file):
            with open(metrics_file, 'r') as f:
                raw = json.load(f)
            mets = raw[sub_key] if sub_key and sub_key in raw else raw
            mets["Model"] = model_name
            metrics_list.append(mets)
        else:
            print(f"Missing metrics for {model_name}: {metrics_file}")

    if not metrics_list:
        print("No metrics found to compare.")
        return

    df = pd.DataFrame(metrics_list)
    df = df.set_index("Model")
    df.to_csv(os.path.join(comp_dir, "model_comparison.csv"))

    with open(os.path.join(comp_dir, "model_comparison.md"), 'w') as f:
        f.write("# Experimental Model Comparison\n\n")
        f.write(df.to_markdown())

    # Plot AUPRC comparison
    auprc_col = next((c for c in df.columns if "auprc" in c.lower()), None)
    if auprc_col:
        plt.figure(figsize=(10, 6))
        sns.barplot(x=df.index, y=df[auprc_col])
        plt.title("AUPRC Comparison")
        plt.ylabel("AUPRC")
        plt.tight_layout()
        plt.savefig(os.path.join(comp_dir, "auprc_comparison.png"))
        plt.close()

    print(f"Model comparison saved to {comp_dir}")

def main():
    # __file__ = f:\Sepsis\sepsis.1\sepsis\experiments\run_all.py
    # dirname once  = f:\Sepsis\sepsis.1\sepsis\experiments
    # dirname twice = f:\Sepsis\sepsis.1\sepsis   (sepsis_root)
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    results_dir = os.path.join(root, "experiments", "results")

    stages = [
        # (os.path.join(root, "experiments", "eda", "run_eda.py"), "EDA Pipeline"),
        (os.path.join(root, "experiments", "transformer", "train.py"), "Transformer Training"),
        (os.path.join(root, "experiments", "transformer", "evaluate.py"), "Transformer Evaluation"),
        (os.path.join(root, "experiments", "gat", "graph_builder.py"), "GAT Graph Construction"),
        (os.path.join(root, "experiments", "gat", "train.py"), "GAT Training"),
        (os.path.join(root, "experiments", "gat", "evaluate.py"), "GAT Evaluation"),
        (os.path.join(root, "experiments", "baseline", "evaluation.py"), "Baseline Evaluation"),
    ]

    summary = {}
    for script_path, desc in stages:
        if os.path.exists(script_path):
            summary[desc] = run_script(script_path, desc)
        else:
            summary[desc] = {"status": "NOT FOUND", "runtime": 0, "error": f"File not found: {script_path}"}

    compare_models(root)

    summary_path = os.path.join(results_dir, "run_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=4)
    print(f"\nExperiment run summary saved to {summary_path}")

if __name__ == "__main__":
    main()
