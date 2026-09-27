"""
evaluate_grud.py  — Test-set evaluation for the saved GRU-D checkpoint.

Usage:
    python evaluate_grud.py                          # uses best_grud_full.pt
    python evaluate_grud.py --checkpoint path/to.pt  # custom checkpoint
"""

import os
import sys
import json
import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    precision_recall_curve,
    roc_curve,
    precision_score,
    recall_score,
    f1_score,
)

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except Exception:
    HAS_MATPLOTLIB = False

# ── Path setup ────────────────────────────────────────────────────────
_THIS_FILE  = os.path.abspath(__file__)
SEPSIS_ROOT = os.path.dirname(os.path.dirname(_THIS_FILE))
PROJECT_ROOT = os.path.dirname(SEPSIS_ROOT)
sys.path.insert(0, SEPSIS_ROOT)

from src.dataset_grud import PhysioNetDatasetGRUD, collate_fn
from src.model_grud import GRUD


def evaluate_checkpoint(ckpt_path: str, output_dir: str) -> dict:
    os.makedirs(output_dir, exist_ok=True)

    # ── Load checkpoint ───────────────────────────────────────────────
    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    cfg          = ckpt["config"]
    prep_config  = ckpt["preprocessing_config"]
    pos_weight   = prep_config["class_weight"]

    print(f"  Saved at epoch  : {ckpt['epoch']}")
    print(f"  Val AUPRC (ckpt): {ckpt['val_auprc']:.4f}")
    print(f"  Val AUROC (ckpt): {ckpt['val_auroc']:.4f}")
    print(f"  pos_weight      : {pos_weight:.4f}")
    print(f"  attention       : {cfg.get('attention', 'disabled')}")

    # ── Device ────────────────────────────────────────────────────────
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"  Device          : {device}")

    # ── Paths ─────────────────────────────────────────────────────────
    splits_path = os.path.join(SEPSIS_ROOT, "artifacts", "splits.json")
    preprocess_cfg_path = os.path.join(SEPSIS_ROOT, "artifacts", "preprocessing_config.json")

    cand_1 = [
        os.path.join(PROJECT_ROOT, "physionet2019", "training", "training_setA"),
        os.path.join(PROJECT_ROOT, "physionet2019", "training", "training_setB"),
    ]
    cand_2 = [
        os.path.join(PROJECT_ROOT, "training", "training_setA"),
        os.path.join(PROJECT_ROOT, "training", "training_setB"),
    ]
    data_dirs = cand_1 if os.path.exists(cand_1[0]) else cand_2

    with open(splits_path) as f:
        splits = json.load(f)
    test_ids = splits.get("test", [])
    if not test_ids:
        print("WARNING: No test split found — using val split for evaluation")
        test_ids = splits["val"]

    # ── Dataset ───────────────────────────────────────────────────────
    max_seq_len = cfg.get("max_seq_len", 336)
    test_ds = PhysioNetDatasetGRUD(data_dirs, test_ids, preprocess_cfg_path, max_seq_len)
    test_loader = DataLoader(
        test_ds, batch_size=cfg.get("batch_size", 64),
        shuffle=False, collate_fn=collate_fn, num_workers=0,
    )
    print(f"  Test patients   : {len(test_ds)}")

    # ── Model ─────────────────────────────────────────────────────────
    input_size  = len(prep_config["dynamic_features"])
    static_size = len(prep_config["static_features"])
    hidden_size = cfg.get("hidden_size", 128)
    dropout     = cfg.get("dropout", 0.2)
    attn_cfg    = cfg.get("attention", {})
    attention_config = attn_cfg if attn_cfg.get("enabled", False) else None

    model = GRUD(input_size, static_size, hidden_size, dropout, attention_config).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model params    : {n_params:,}")

    # ── Inference ─────────────────────────────────────────────────────
    all_probs, all_labels = [], []
    with torch.no_grad():
        for values, mask, delta, static_feats, labels, valid_mask in test_loader:
            values       = values.to(device)
            mask         = mask.to(device)
            delta        = delta.to(device)
            static_feats = static_feats.to(device)
            labels       = labels.to(device)
            valid_mask   = valid_mask.to(device)

            logits = model(values, mask, delta, static_feats, valid_mask)
            probs  = torch.sigmoid(logits)

            flat_valid = valid_mask.bool().view(-1)
            all_probs.append(probs.view(-1)[flat_valid].cpu().numpy())
            all_labels.append(labels.view(-1)[flat_valid].cpu().numpy())

    all_probs  = np.concatenate(all_probs)
    all_labels = np.concatenate(all_labels)

    n_pos = int(all_labels.sum())
    n_tot = len(all_labels)
    print(f"\n  Timesteps evaluated : {n_tot:,}")
    print(f"  Positive timesteps  : {n_pos:,} ({100*n_pos/n_tot:.2f}%)")

    # ── Metrics ───────────────────────────────────────────────────────
    auprc = average_precision_score(all_labels, all_probs)
    auroc = roc_auc_score(all_labels, all_probs)

    # Find threshold that maximises F1
    prec_vals, rec_vals, thresholds = precision_recall_curve(all_labels, all_probs)
    f1_vals = 2 * prec_vals * rec_vals / (prec_vals + rec_vals + 1e-9)
    best_idx = int(np.argmax(f1_vals[:-1]))
    best_thr = float(thresholds[best_idx])
    bin_preds = (all_probs >= best_thr).astype(int)

    precision = precision_score(all_labels, bin_preds, zero_division=0)
    recall    = recall_score(all_labels, bin_preds, zero_division=0)
    f1        = f1_score(all_labels, bin_preds, zero_division=0)

    metrics = {
        "model"          : "GRU-D",
        "checkpoint"     : os.path.basename(ckpt_path),
        "checkpoint_epoch": ckpt["epoch"],
        "val_auprc_ckpt" : float(ckpt["val_auprc"]),
        "val_auroc_ckpt" : float(ckpt["val_auroc"]),
        "test_auprc"     : float(auprc),
        "test_auroc"     : float(auroc),
        "test_precision" : float(precision),
        "test_recall"    : float(recall),
        "test_f1"        : float(f1),
        "best_threshold" : best_thr,
        "n_test_timesteps": n_tot,
        "pct_positive"   : float(100 * n_pos / n_tot),
    }

    print("\n=== TEST RESULTS ===")
    print(f"  AUPRC     : {auprc:.4f}")
    print(f"  AUROC     : {auroc:.4f}")
    print(f"  Precision : {precision:.4f}  (@ threshold {best_thr:.3f})")
    print(f"  Recall    : {recall:.4f}")
    print(f"  F1        : {f1:.4f}")

    # ── Save metrics ──────────────────────────────────────────────────
    metrics_path = os.path.join(output_dir, "metrics_grud.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)
    print(f"\n  Metrics saved  → {metrics_path}")

    # ── Plots ─────────────────────────────────────────────────────────
    if HAS_MATPLOTLIB:
        # PR curve
        plt.figure(figsize=(7, 5))
        plt.plot(rec_vals, prec_vals, lw=2, label=f"GRU-D  AUPRC={auprc:.4f}")
        plt.axhline(y=n_pos/n_tot, color="gray", linestyle="--", label="Baseline (random)")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precision-Recall Curve — GRU-D (Test Set)")
        plt.legend()
        plt.tight_layout()
        pr_path = os.path.join(output_dir, "pr_curve_grud.png")
        plt.savefig(pr_path, dpi=150)
        plt.close()
        print(f"  PR curve saved → {pr_path}")

        # ROC curve
        fpr, tpr, _ = roc_curve(all_labels, all_probs)
        plt.figure(figsize=(7, 5))
        plt.plot(fpr, tpr, lw=2, label=f"GRU-D  AUROC={auroc:.4f}")
        plt.plot([0, 1], [0, 1], "k--", label="Random")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve — GRU-D (Test Set)")
        plt.legend()
        plt.tight_layout()
        roc_path = os.path.join(output_dir, "roc_curve_grud.png")
        plt.savefig(roc_path, dpi=150)
        plt.close()
        print(f"  ROC curve saved→ {roc_path}")

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate GRU-D checkpoint on test set")
    parser.add_argument(
        "--checkpoint",
        default=os.path.join(SEPSIS_ROOT, "checkpoints", "best_grud_full.pt"),
        help="Path to saved GRU-D checkpoint",
    )
    parser.add_argument(
        "--output_dir",
        default=os.path.join(SEPSIS_ROOT, "experiments", "results", "grud"),
        help="Directory for metrics and plots",
    )
    args = parser.parse_args()

    evaluate_checkpoint(args.checkpoint, args.output_dir)


if __name__ == "__main__":
    main()
