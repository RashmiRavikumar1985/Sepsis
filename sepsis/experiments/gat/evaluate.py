"""
evaluate.py — GAT Phase 2 evaluation script for FedSepsis-KG.

Loads the best GAT checkpoint produced by train.py and evaluates
on the held-out test split. Saves metrics.json and PR/ROC curve plots.

Usage:
    python evaluate.py
    python evaluate.py --config config_gat.json
    python evaluate.py --checkpoint path/to/best_model_full.pt
"""

import os
import sys
import json
import argparse

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
    precision_recall_curve,
)

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except Exception:
    HAS_MATPLOTLIB = False

# ── Path setup ────────────────────────────────────────────────────────
# __file__ = sepsis/experiments/gat/evaluate.py
_THIS_FILE  = os.path.abspath(__file__)
_GAT_DIR    = os.path.dirname(_THIS_FILE)
SEPSIS_ROOT = os.path.dirname(os.path.dirname(_GAT_DIR))
PROJECT_ROOT = os.path.dirname(SEPSIS_ROOT)

sys.path.insert(0, SEPSIS_ROOT)

from src.dataset_grud import PhysioNetDatasetGRUD, collate_fn
from experiments.gat.model import GATBaseline


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate GAT for sepsis prediction")
    parser.add_argument(
        "--config",
        default=os.path.join(_GAT_DIR, "config_gat.json"),
        help="Path to GAT config JSON",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Path to model checkpoint (.pt). Defaults to results/gat/best_model_full.pt",
    )
    args = parser.parse_args()

    # ── Load config ───────────────────────────────────────────────────
    with open(args.config) as f:
        cfg = json.load(f)

    HIDDEN_DIM  = cfg["hidden_dim"]
    OUT_DIM     = cfg["out_dim"]
    HEADS       = cfg["heads"]
    DROPOUT     = cfg["dropout"]
    BATCH_SIZE  = cfg["batch_size"]
    MAX_SEQ_LEN = cfg["max_seq_len"]

    # ── Device ────────────────────────────────────────────────────────
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    # ── Paths — all derived from __file__, nothing hardcoded ──────────
    preprocess_cfg_path = os.path.join(SEPSIS_ROOT, "artifacts", "preprocessing_config.json")
    splits_path         = os.path.join(SEPSIS_ROOT, "artifacts", "splits.json")
    edges_path          = os.path.join(_GAT_DIR, "edges.csv")
    results_dir         = os.path.join(SEPSIS_ROOT, "experiments", "results", "gat")
    plots_dir           = os.path.join(_GAT_DIR, "plots")

    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    # Default checkpoint path
    ckpt_path = args.checkpoint or os.path.join(results_dir, "best_model_full.pt")

    # ── Data directories — two-candidate fallback ─────────────────────
    cand_1 = [
        os.path.join(PROJECT_ROOT, "physionet2019", "training", "training_setA"),
        os.path.join(PROJECT_ROOT, "physionet2019", "training", "training_setB"),
    ]
    cand_2 = [
        os.path.join(PROJECT_ROOT, "training", "training_setA"),
        os.path.join(PROJECT_ROOT, "training", "training_setB"),
    ]
    if os.path.exists(cand_1[0]):
        data_dirs = cand_1
    elif os.path.exists(cand_2[0]):
        data_dirs = cand_2
    else:
        raise FileNotFoundError(
            f"Cannot find training data. Tried:\n  {cand_1[0]}\n  {cand_2[0]}"
        )

    # ── Load data schema ──────────────────────────────────────────────
    with open(preprocess_cfg_path) as f:
        prep_config = json.load(f)
    with open(splits_path) as f:
        splits = json.load(f)

    # Derive sizes from preprocessing config — never from manual constants
    num_nodes   = len(prep_config["dynamic_features"])
    static_size = len(prep_config["static_features"])
    pos_weight  = prep_config["class_weight"]
    test_ids    = splits.get("test", [])

    print(f"Nodes (dynamic features): {num_nodes}")
    print(f"Static features         : {static_size}")
    print(f"Test patients           : {len(test_ids)}")

    if not test_ids:
        print("No test split found in splits.json — exiting.")
        return

    # ── Load graph ────────────────────────────────────────────────────
    if not os.path.exists(edges_path):
        raise FileNotFoundError(
            f"Graph edges not found at {edges_path}. Run graph_builder.py first."
        )
    edges_df    = pd.read_csv(edges_path)
    edge_index  = torch.tensor(
        [edges_df["source"].values, edges_df["target"].values], dtype=torch.long
    ).to(device)
    edge_weight = torch.tensor(edges_df["weight"].values, dtype=torch.float).to(device)
    print(f"Graph: {num_nodes} nodes, {edge_index.size(1)} edges")

    # ── Dataset and loader ────────────────────────────────────────────
    test_ds = PhysioNetDatasetGRUD(data_dirs, test_ids, preprocess_cfg_path, MAX_SEQ_LEN)
    pin = device.type == "cuda"
    test_loader = DataLoader(
        test_ds, batch_size=BATCH_SIZE, shuffle=False,
        collate_fn=collate_fn, num_workers=0, pin_memory=pin,
    )

    # ── Model ─────────────────────────────────────────────────────────
    model = GATBaseline(
        num_nodes=num_nodes,
        input_dim_per_node=3,
        static_size=static_size,
        hidden_dim=HIDDEN_DIM,
        out_dim=OUT_DIM,
        heads=HEADS,
        dropout=DROPOUT,
    ).to(device)

    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"Loaded checkpoint: {ckpt_path}")
        print(f"  Checkpoint epoch  : {ckpt.get('epoch', 'N/A')}")
        print(f"  Val AUROC         : {ckpt.get('val_auroc', 'N/A')}")
        print(f"  Val AUPRC         : {ckpt.get('val_auprc', 'N/A')}")
    else:
        print(f"WARNING: Checkpoint not found at {ckpt_path}")
        print("Evaluating with randomly initialised weights — metrics will be meaningless.")

    # ── Inference ─────────────────────────────────────────────────────
    model.eval()
    all_probs, all_labels = [], []

    with torch.no_grad():
        for values, mask, delta, static_feats, labels, valid_mask in test_loader:
            values       = values.to(device)
            mask         = mask.to(device)
            delta        = delta.to(device)
            static_feats = static_feats.to(device)
            labels       = labels.to(device)
            valid_mask   = valid_mask.to(device)

            logits = model(values, mask, delta, static_feats, edge_index, valid_mask, edge_weight)
            probs  = torch.sigmoid(logits)

            flat_valid = valid_mask.bool().view(-1)
            all_probs.append(probs.view(-1)[flat_valid].cpu())
            all_labels.append(labels.view(-1)[flat_valid].cpu())

    all_probs  = torch.cat(all_probs).numpy()
    all_labels = torch.cat(all_labels).numpy()

    # ── Metrics ───────────────────────────────────────────────────────
    n_pos = int(all_labels.sum())
    n_tot = len(all_labels)
    print(f"\nTimesteps evaluated : {n_tot:,}  |  Positives: {n_pos:,} ({100*n_pos/n_tot:.2f}%)")

    auroc = roc_auc_score(all_labels, all_probs)
    auprc = average_precision_score(all_labels, all_probs)

    # F1-optimal threshold from PR curve
    prec_vals, rec_vals, thresholds = precision_recall_curve(all_labels, all_probs)
    f1_vals  = 2 * prec_vals * rec_vals / (prec_vals + rec_vals + 1e-9)
    best_idx = int(np.argmax(f1_vals[:-1]))
    best_thr = float(thresholds[best_idx])

    bin_preds = (all_probs >= best_thr).astype(int)
    prec  = precision_score(all_labels, bin_preds, zero_division=0)
    rec   = recall_score(all_labels, bin_preds, zero_division=0)
    f1    = f1_score(all_labels, bin_preds, zero_division=0)

    metrics = {
        "Test AUPRC"     : float(auprc),
        "Test AUROC"     : float(auroc),
        "Test Precision" : float(prec),
        "Test Recall"    : float(rec),
        "Test F1"        : float(f1),
        "Best Threshold" : best_thr,
        "N Timesteps"    : n_tot,
        "Pct Positive"   : float(100 * n_pos / n_tot),
    }

    print("\n" + "=" * 50)
    print("GAT Test Results")
    print("=" * 50)
    for k, v in metrics.items():
        print(f"  {k:<20}: {v:.4f}")

    with open(os.path.join(results_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)
    print(f"\nMetrics saved → {results_dir}/metrics.json")

    # ── Plots ─────────────────────────────────────────────────────────
    if HAS_MATPLOTLIB:
        # PR curve
        prec_vals, rec_vals, _ = precision_recall_curve(all_labels, all_probs)
        plt.figure(figsize=(8, 6))
        plt.plot(rec_vals, prec_vals, label=f"AUPRC = {auprc:.4f}")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precision-Recall Curve (GAT)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "pr_curve.png"))
        plt.close()

        # ROC curve
        fpr, tpr, _ = roc_curve(all_labels, all_probs)
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f"AUROC = {auroc:.4f}")
        plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve (GAT)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "roc_curve.png"))
        plt.close()

        print(f"Plots saved → {plots_dir}/")


if __name__ == "__main__":
    main()
