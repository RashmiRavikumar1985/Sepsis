import os
import sys
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import (average_precision_score, roc_auc_score, precision_score, 
                             recall_score, f1_score, roc_curve, precision_recall_curve)
import matplotlib.pyplot as plt
import pandas as pd

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from src.dataset_grud import PhysioNetDatasetGRUD, collate_fn
from experiments.gat.model import GATBaseline

def evaluate_gat():
    print("=== Evaluating GAT Baseline ===")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    config_path = os.path.join(project_root, "experiments", "transformer", "config.json")
    with open(config_path, 'r') as f:
        config = json.load(f)
        
    splits_path = os.path.join(project_root, "artifacts", "splits.json")
    with open(splits_path, 'r') as f:
        splits = json.load(f)
        
    data_dirs = [
        os.path.join(project_root, "physionet2019", "training", "training_setA"),
        os.path.join(project_root, "physionet2019", "training", "training_setB")
    ]
    preprocess_cfg = os.path.join(project_root, "artifacts", "preprocessing_config.json")
    
    test_ids = splits['test']
    
    test_ds = PhysioNetDatasetGRUD(data_dirs, test_ids, preprocess_cfg, config['max_sequence_length'])
    test_loader = DataLoader(test_ds, batch_size=8, shuffle=False, collate_fn=collate_fn)
    
    # Load Graph edges
    graph_path = os.path.join(project_root, "experiments", "gat", "edges.csv")
    edges_df = pd.read_csv(graph_path)
    sources = edges_df['source'].values
    targets = edges_df['target'].values
    weights = edges_df['weight'].values
    
    edge_index = torch.tensor([sources, targets], dtype=torch.long).to(device)
    edge_weight = torch.tensor(weights, dtype=torch.float).to(device)

    model = GATBaseline(
        num_nodes=config['input_dim'],
        input_dim_per_node=3,
        static_size=config['static_dim'],
        hidden_dim=32,
        out_dim=64,
        heads=2,
        dropout=config['dropout']
    ).to(device)
    
    results_dir = os.path.join(project_root, "experiments", "results", "gat")
    model_path = os.path.join(results_dir, "best_model.pt")
    
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        print("Warning: best_model.pt not found. Evaluating randomly initialized model.")
        
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for values, mask, delta, static_features, labels, valid_mask in test_loader:
            values, mask = values.to(device), mask.to(device)
            delta, static_features = delta.to(device), static_features.to(device)
            labels, valid_mask = labels.to(device), valid_mask.to(device)
            
            logits = model(values, mask, delta, static_features, edge_index, edge_weight)
            probs = torch.sigmoid(logits)
            valid_idx = valid_mask.bool()
            
            all_preds.extend(probs[valid_idx].cpu().numpy().tolist())
            all_labels.extend(labels[valid_idx].cpu().numpy().tolist())
            
    auprc = average_precision_score(all_labels, all_preds)
    auroc = roc_auc_score(all_labels, all_preds)
    
    bin_preds = [1 if p >= 0.5 else 0 for p in all_preds]
    precision = precision_score(all_labels, bin_preds, zero_division=0)
    recall = recall_score(all_labels, bin_preds, zero_division=0)
    f1 = f1_score(all_labels, bin_preds, zero_division=0)
    
    metrics = {
        "Test AUPRC": auprc,
        "Test AUROC": auroc,
        "Test Precision": precision,
        "Test Recall": recall,
        "Test F1": f1
    }
    
    with open(os.path.join(results_dir, "metrics.json"), 'w') as f:
        json.dump(metrics, f, indent=4)
        
    print(f"Test AUPRC: {auprc:.4f} | Test AUROC: {auroc:.4f}")
    
    plot_dir = os.path.join(os.path.dirname(__file__), "plots")
    
    precision_vals, recall_vals, _ = precision_recall_curve(all_labels, all_preds)
    plt.figure(figsize=(8,6))
    plt.plot(recall_vals, precision_vals, label=f'AUPRC = {auprc:.4f}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve (GAT)')
    plt.legend()
    plt.savefig(os.path.join(plot_dir, "pr_curve.png"))
    plt.close()
    
    fpr, tpr, _ = roc_curve(all_labels, all_preds)
    plt.figure(figsize=(8,6))
    plt.plot(fpr, tpr, label=f'AUROC = {auroc:.4f}')
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve (GAT)')
    plt.legend()
    plt.savefig(os.path.join(plot_dir, "roc_curve.png"))
    plt.close()
    
if __name__ == "__main__":
    evaluate_gat()
