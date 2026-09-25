"""
evaluate_temporal.py  -  GAT-2 (Temporal GAT) Evaluation Script
================================================================

Evaluates the best saved TemporalGAT model on the test split.
Preserves identical evaluation methodology as GAT-1 and Transformer.

Saves:
  experiments/results/gat_temporal/metrics.json
  experiments/gat/plots_temporal/pr_curve.png
  experiments/gat/plots_temporal/roc_curve.png
"""

import os
import sys
import json
import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import (
    average_precision_score, roc_auc_score, precision_score,
    recall_score, f1_score, roc_curve, precision_recall_curve
)

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_MPL = True
except Exception:
    HAS_MPL = False

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from src.dataset_grud import PhysioNetDatasetGRUD, collate_fn
from experiments.gat.temporal_graph_builder import (
    load_clinical_edges, build_batched_temporal_edge_index,
)
from experiments.gat.temporal_model import TemporalGAT
from experiments.gat.train_temporal import find_data_dirs, GAT2_CONFIG


def evaluate_temporal_gat():
    print("=" * 65)
    print("  GAT-2: Temporal GAT — Evaluation on Test Split")
    print("=" * 65)

    sepsis_root   = project_root
    prep_cfg_path = os.path.join(sepsis_root, "artifacts", "preprocessing_config.json")
    splits_path   = os.path.join(sepsis_root, "artifacts", "splits.json")
    edges_csv     = os.path.join(sepsis_root, "experiments", "gat", "edges.csv")
    results_dir   = os.path.join(sepsis_root, "experiments", "results", "gat_temporal")
    plots_dir     = os.path.join(sepsis_root, "experiments", "gat", "plots_temporal")
    model_path    = os.path.join(results_dir, "best_model.pt")
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(plots_dir,   exist_ok=True)

    with open(prep_cfg_path) as f:
        prep_cfg = json.load(f)
    with open(splits_path) as f:
        splits = json.load(f)

    F = len(prep_cfg["dynamic_features"])
    S = len(prep_cfg["static_features"])
    assert F == 35 and S == 5

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Device: {device}")

    data_dirs = find_data_dirs(os.path.dirname(sepsis_root), sepsis_root)

    test_ds = PhysioNetDatasetGRUD(data_dirs, splits["test"], prep_cfg_path, 336)
    test_loader = DataLoader(test_ds, batch_size=8, shuffle=False,
                             collate_fn=collate_fn, num_workers=0)

    clinical_src, clinical_dst = load_clinical_edges(edges_csv)

    cfg = GAT2_CONFIG
    model = TemporalGAT(
        num_features=F, static_size=S,
        hidden_dim=cfg["hidden_dim"], out_dim=cfg["out_dim"],
        num_heads=cfg["num_heads"], num_gat_layers=cfg["num_gat_layers"],
        dropout=0.0,  # no dropout at eval
    ).to(device)

    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Loaded model: {model_path}")
    else:
        print(f"WARNING: {model_path} not found — evaluating random weights")

    model.eval()
    all_preds  = []
    all_labels = []

    with torch.no_grad():
        for values, mask, delta, static_feat, labels, valid_mask in test_loader:
            values      = values.to(device)
            mask        = mask.to(device)
            delta       = delta.to(device)
            static_feat = static_feat.to(device)
            labels      = labels.to(device)
            valid_mask  = valid_mask.to(device)

            T_pad = values.size(1)
            valid_lens = valid_mask.sum(dim=1).cpu().tolist()
            valid_lens = [int(v) for v in valid_lens]
            edge_index, _, _ = build_batched_temporal_edge_index(
                valid_lens, clinical_src, clinical_dst, T_pad, cfg["add_self_loops"]
            )
            edge_index = edge_index.to(device)

            logits = model(values, mask, delta, static_feat, edge_index, valid_mask)
            probs  = torch.sigmoid(logits)

            all_preds.append(probs[valid_mask].cpu())
            all_labels.append(labels[valid_mask].cpu())

    all_preds  = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()

    auprc  = average_precision_score(all_labels, all_preds)
    auroc  = roc_auc_score(all_labels, all_preds)
    bin_p  = (all_preds >= 0.5).astype(int)
    prec   = precision_score(all_labels, bin_p, zero_division=0)
    rec    = recall_score(all_labels, bin_p, zero_division=0)
    f1     = f1_score(all_labels, bin_p, zero_division=0)

    metrics = {
        "model": "GAT-2 (TemporalGAT)",
        "Test AUPRC": float(auprc),
        "Test AUROC": float(auroc),
        "Test Precision": float(prec),
        "Test Recall": float(rec),
        "Test F1": float(f1),
    }
    with open(os.path.join(results_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)

    print(f"\nTest AUPRC : {auprc:.4f}")
    print(f"Test AUROC : {auroc:.4f}")
    print(f"Test F1    : {f1:.4f}")
    print(f"Saved: {results_dir}/metrics.json")

    if HAS_MPL:
        prec_vals, rec_vals, _ = precision_recall_curve(all_labels, all_preds)
        plt.figure(figsize=(8, 6))
        plt.plot(rec_vals, prec_vals, label=f"AUPRC={auprc:.4f}")
        plt.xlabel("Recall"); plt.ylabel("Precision")
        plt.title("GAT-2 Temporal GAT — PR Curve")
        plt.legend(); plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "pr_curve.png")); plt.close()

        fpr, tpr, _ = roc_curve(all_labels, all_preds)
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f"AUROC={auroc:.4f}")
        plt.plot([0, 1], [0, 1], "--", color="gray")
        plt.xlabel("FPR"); plt.ylabel("TPR")
        plt.title("GAT-2 Temporal GAT — ROC Curve")
        plt.legend(); plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "roc_curve.png")); plt.close()

        print(f"Plots saved: {plots_dir}/")

    return metrics


if __name__ == "__main__":
    evaluate_temporal_gat()
