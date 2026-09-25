"""
evaluate_temporal.py  -  GAT-2 (Temporal GAT) Evaluation Script
================================================================

Evaluates a saved TemporalGAT checkpoint on the test split.

Usage examples:
  # Default (evaluates gat_temporal/best_model.pt at 72h):
  python experiments/gat/evaluate_temporal.py

  # Evaluate the 120h run:
  python experiments/gat/evaluate_temporal.py \
      --model-path experiments/results/gat2_120h/best_model.pt \
      --seq-len 120 \
      --results-dir experiments/results/gat2_120h

Saves:
  <results-dir>/metrics.json
  experiments/gat/plots_<run_name>/pr_curve.png
  experiments/gat/plots_<run_name>/roc_curve.png
"""

import argparse
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
from experiments.gat.temporal_graph_builder import load_clinical_edges
from experiments.gat.temporal_model import TemporalGAT
from experiments.gat.train_temporal import find_data_dirs, GAT2_CONFIG


def evaluate_temporal_gat(model_path=None, seq_len=None, results_dir=None, plots_dir=None, split="test"):
    print("=" * 65)
    print(f"  GAT-2: Temporal GAT — Evaluation on {split.upper()} Split")
    print("=" * 65)

    sepsis_root   = project_root
    prep_cfg_path = os.path.join(sepsis_root, "artifacts", "preprocessing_config.json")
    splits_path   = os.path.join(sepsis_root, "artifacts", "splits.json")
    edges_csv     = os.path.join(sepsis_root, "experiments", "gat", "edges.csv")

    # ── Resolve paths: CLI args take priority over defaults ──
    default_results = os.path.join(sepsis_root, "experiments", "results", "gat_temporal")
    results_dir = results_dir or default_results
    model_path  = model_path  or os.path.join(results_dir, "best_model.pt")

    # Derive plots subdir from results_dir basename (e.g. gat2_120h -> plots_gat2_120h)
    run_name  = os.path.basename(os.path.normpath(results_dir))
    plots_dir = plots_dir or os.path.join(
        sepsis_root, "experiments", "gat", f"plots_{run_name}"
    )

    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(plots_dir,   exist_ok=True)

    print(f"Model path   : {model_path}")
    print(f"Results dir  : {results_dir}")
    print(f"Plots dir    : {plots_dir}")

    # ── Auto-detect seq_len from run_config.json if not supplied ──
    run_cfg_path = os.path.join(results_dir, "run_config.json")
    if seq_len is None and os.path.exists(run_cfg_path):
        with open(run_cfg_path) as f:
            run_cfg = json.load(f)
        seq_len = run_cfg.get("max_seq_len", GAT2_CONFIG["max_seq_len"])
        print(f"seq_len      : {seq_len}h  (from run_config.json)")
    elif seq_len is None:
        seq_len = GAT2_CONFIG["max_seq_len"]
        print(f"seq_len      : {seq_len}h  (default from GAT2_CONFIG)")
    else:
        print(f"seq_len      : {seq_len}h  (from --seq-len arg)")

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
    print(f"Device       : {device}")

    data_dirs = find_data_dirs(os.path.dirname(sepsis_root), sepsis_root)

    # Use the correct seq_len for the chosen split dataset
    eval_ds = PhysioNetDatasetGRUD(data_dirs, splits[split], prep_cfg_path, seq_len)
    eval_loader = DataLoader(eval_ds, batch_size=8, shuffle=False,
                             collate_fn=collate_fn, num_workers=0)

    clinical_src, clinical_dst = load_clinical_edges(edges_csv)

    cfg = GAT2_CONFIG
    model = TemporalGAT(
        num_features=F, static_size=S,
        hidden_dim=cfg["hidden_dim"], out_dim=cfg["out_dim"],
        num_heads=cfg["num_heads"], num_gat_layers=cfg["num_gat_layers"],
        dropout=0.0,  # no dropout at eval time
    ).to(device)

    # Bake adjacency into model (required before load_state_dict)
    model.set_adjacency(clinical_src, clinical_dst, cfg["add_self_loops"])

    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        print(f"Loaded model : {model_path}")
    else:
        raise FileNotFoundError(
            f"Checkpoint not found: {model_path}\n"
            f"Did you mean to pass --model-path <path>?"
        )

    model.eval()
    all_preds  = []
    all_labels = []

    with torch.no_grad():
        for values, mask, delta, static_feat, labels, valid_mask in eval_loader:
            values      = values.to(device)
            mask        = mask.to(device)
            delta       = delta.to(device)
            static_feat = static_feat.to(device)
            labels      = labels.to(device)
            valid_mask  = valid_mask.to(device)

            # TemporalGAT uses baked adjacency — no edge_index needed at runtime
            logits = model(values, mask, delta, static_feat, valid_mask=valid_mask)
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

    prefix = split.capitalize()
    metrics = {
        "model": f"GAT-2 (TemporalGAT, seq_len={seq_len}h)",
        "run_name": run_name,
        "split": split,
        "seq_len": seq_len,
        f"{prefix} AUPRC": float(auprc),
        f"{prefix} AUROC": float(auroc),
        f"{prefix} Precision": float(prec),
        f"{prefix} Recall": float(rec),
        f"{prefix} F1": float(f1),
    }
    metrics_file = f"{split}_metrics.json" if split != "test" else "metrics.json"
    metrics_path = os.path.join(results_dir, metrics_file)
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)

    print(f"\n{prefix} AUPRC : {auprc:.4f}")
    print(f"{prefix} AUROC : {auroc:.4f}")
    print(f"{prefix} F1    : {f1:.4f}")
    print(f"Saved: {metrics_path}")

    if HAS_MPL:
        prec_vals, rec_vals, _ = precision_recall_curve(all_labels, all_preds)
        plt.figure(figsize=(8, 6))
        plt.plot(rec_vals, prec_vals, label=f"AUPRC={auprc:.4f}")
        plt.xlabel("Recall"); plt.ylabel("Precision")
        plt.title(f"GAT-2 ({run_name}) — PR Curve")
        plt.legend(); plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "pr_curve.png")); plt.close()

        fpr, tpr, _ = roc_curve(all_labels, all_preds)
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f"AUROC={auroc:.4f}")
        plt.plot([0, 1], [0, 1], "--", color="gray")
        plt.xlabel("FPR"); plt.ylabel("TPR")
        plt.title(f"GAT-2 ({run_name}) — ROC Curve")
        plt.legend(); plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "roc_curve.png")); plt.close()

        print(f"Plots saved: {plots_dir}/")

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate a saved TemporalGAT checkpoint on the test split."
    )
    parser.add_argument(
        "--model-path", type=str, default=None,
        help="Path to best_model.pt. Default: <results-dir>/best_model.pt"
    )
    parser.add_argument(
        "--seq-len", type=int, default=None,
        help="max_seq_len used during training (e.g. 72, 120). "
             "Auto-detected from run_config.json in results-dir if omitted."
    )
    parser.add_argument(
        "--results-dir", type=str, default=None,
        help="Directory to save metrics.json. "
             "Default: experiments/results/gat_temporal/"
    )
    parser.add_argument(
        "--plots-dir", type=str, default=None,
        help="Directory to save PR/ROC plots. "
             "Default: experiments/gat/plots_<run_name>/"
    )
    parser.add_argument(
        "--split", type=str, default="test", choices=["test", "val", "train"],
        help="Dataset split to evaluate on ('test', 'val', or 'train'). Default: 'test'"
    )
    args = parser.parse_args()
    evaluate_temporal_gat(
        model_path=args.model_path,
        seq_len=args.seq_len,
        results_dir=args.results_dir,
        plots_dir=args.plots_dir,
        split=args.split,
    )
