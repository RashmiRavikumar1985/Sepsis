import os
import sys
import json
import shutil

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

def evaluate_baseline():
    print("=== Extracting Baseline Evaluation (GRU-D) ===")
    
    # We will pull the results from the existing checkpoints/training_log_full.json
    log_path = os.path.join(project_root, "checkpoints", "training_log_full.json")
    results_dir = os.path.join(project_root, "experiments", "results", "baseline")
    os.makedirs(results_dir, exist_ok=True)
    
    if not os.path.exists(log_path):
        print(f"Error: {log_path} not found. Cannot extract baseline metrics.")
        return
        
    with open(log_path, 'r') as f:
        log_data = json.load(f)
        
    if "test_results" not in log_data:
        print("Error: test_results not found in baseline log.")
        return
        
    test_results = log_data["test_results"]
    metrics = {
        "Test AUPRC": test_results.get("test_auprc", 0.0),
        "Test AUROC": test_results.get("test_auroc", 0.0),
        "Test Precision": 0.0, # Not recorded in old run
        "Test Recall": 0.0,    # Not recorded in old run
        "Test F1": 0.0         # Not recorded in old run
    }
    
    with open(os.path.join(results_dir, "metrics.json"), 'w') as f:
        json.dump(metrics, f, indent=4)
        
    print(f"Test AUPRC: {metrics['Test AUPRC']:.4f} | Test AUROC: {metrics['Test AUROC']:.4f}")
    
if __name__ == "__main__":
    evaluate_baseline()
