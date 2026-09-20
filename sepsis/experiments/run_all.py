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
    print("=== Generating Model Comparison ===")
    comp_dir = os.path.join(root, "experiments", "results", "comparison")
    os.makedirs(comp_dir, exist_ok=True)
    
    models = ["baseline", "transformer", "gat"]
    model_names = ["GRU-D", "Temporal Transformer", "GAT (Graph Baseline)"]
    metrics_list = []
    
    for i, m in enumerate(models):
        metrics_file = os.path.join(root, "experiments", "results", m, "metrics.json")
        if os.path.exists(metrics_file):
            with open(metrics_file, 'r') as f:
                mets = json.load(f)
                mets["Model"] = model_names[i]
                metrics_list.append(mets)
        else:
            print(f"Missing metrics for {m}")
            
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
    if "Test AUPRC" in df.columns:
        plt.figure(figsize=(10, 6))
        sns.barplot(x=df.index, y=df["Test AUPRC"])
        plt.title("AUPRC Comparison")
        plt.ylabel("AUPRC")
        plt.tight_layout()
        plt.savefig(os.path.join(comp_dir, "auprc_comparison.png"))
        plt.close()
        
    print(f"Model comparison saved to {comp_dir}")

def main():
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
