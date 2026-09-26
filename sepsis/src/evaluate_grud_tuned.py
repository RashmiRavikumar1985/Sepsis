"""
evaluate_grud_tuned.py — Test-set evaluation for the tuned GRU-D checkpoint.

Loads checkpoints/best_grud_tuned_full.pt (or artifacts/tuned_grud.pt).

Usage:
    python src/evaluate_grud_tuned.py
    python src/evaluate_grud_tuned.py --checkpoint path/to/checkpoint.pt
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

_THIS_FILE   = os.path.abspath(__file__)
SEPSIS_ROOT  = os.path.dirname(os.path.dirname(_THIS_FILE))
PROJECT_ROOT = os.path.dirname(SEPSIS_ROOT)
sys.path.insert(0, SEPSIS_ROOT)

from src.dataset_grud import PhysioNetDatasetGRUD, collate_fn
from src.model_grud import GRUD


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument(
        "--output_dir",
        default=os.path.join(SEPSIS_ROOT, "experiments", "results", "grud_tuned"),
    )
    args = parser.parse_args()

    # ── Find checkpoint ───────────────────────────────────────────────
    ckpt_path = args.checkpoint
    if not ckpt_path:
        ckpt_path = os.path.join(SEPSIS_ROOT, "checkpoints", "best_grud_tuned_full.pt")
    if not os.path.exists(ckpt_path):
        ckpt_path = os.path.join(SEPSIS_ROOT, "artifacts", "tuned_grud.pt")
    if not os.path.exists(ckpt_path):
        print(f"ERROR: No tuned GRU-D checkpoint found.")
        print("Train first: python src/train_grud_tuned.py")
        return

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg         = ckpt["config"]
    prep_config = ckpt["preprocessing_config"]
    pos_weight  = prep_config["class_weight"]

    print(f"  Saved epoch    : {ckpt['epoch']}")
    print(f"  Val AUPRC      : {ckpt['val_auprc']:.4f}")
    print(f"  Val AUROC      : {ckpt['val_auroc']:.4f}")
    print(f"  dropout        : {cfg.get('dropout')}")
    print(f"  learning_rate  : {cfg.get('learning_rate')}")
    print(f"  attention      : {cfg.get('attention')}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device         : {device}")

    splits_path         = os.path.join(SEPSIS_ROOT, "artifacts", "splits.json")
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
    test_ids = splits.get("test", splits.get("val", []))

    test_ds = PhysioNetDatasetGRUD(
        data_dirs, test_ids, preprocess_cfg_path, cfg.get("max_seq_len", 336)
    )
    test_loader = DataLoader(
        test_ds, batch_size=cfg.get("batch_size", 64),
        shuffle=False, collate_fn=collate_fn, num_workers=0,
    )
    print(f"  Test patients  : {len(test_ds)}")

    input_size  = len(prep_config["dynamic_features"])
    static_size = len(prep_config["static_features"])
    attn_cfg    = cfg.get("attention", {})
    attn_config = attn_cfg if attn_cfg.get("enabled", False) else None

    model = GRUD(input_size, static_size, cfg.get("hidden_size", 128),
                 cfg.get("dropout", 0.3), attn_config).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
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
            logits = model(values, mask, delta, static_feats, valid_mask)
            probs  = torch.sigmoid(logits)
            flat_valid = valid_mask.bool().view(-1)
            all_probs.append(probs.view(-1)[flat_valid].cpu().numpy())
            all_labels.append(labels.view(-1)[flat_valid].cpu().numpy())

    all_probs  = np.concatenate(all_probs)
    all_labels = np.concatenate(all_labels)
    n_pos = int(all_labels.sum())
    n_tot = len(all_labels)
    print(f"\n  Timesteps      : {n_tot:,}  |  Positives: {n_pos:,} ({100*n_pos/n_tot:.2f}%)")

    auprc = average_precision_score(all_labels, all_probs)
    auroc = roc_auc_score(all_labels, all_probs)
    prec_vals, rec_vals, thresholds = precision_recall_curve(all_labels, all_probs)
    f1_vals  = 2 * prec_vals * rec_vals / (prec_vals + rec_vals + 1e-9)
    best_idx = int(np.argmax(f1_vals[:-1]))
    best_thr = float(thresholds[best_idx])
    bin_preds = (all_probs >= best_thr).astype(int)
    precision = precision_score(all_labels, bin_preds, zero_division=0)
    recall    = recall_score(all_labels, bin_preds, zero_division=0)
    f1        = f1_score(all_labels, bin_preds, zero_division=0)

    print("\n=== TEST RESULTS (Tuned GRU-D) ===")
    print(f"  AUPRC     : {auprc:.4f}")
    print(f"  AUROC     : {auroc:.4f}")
    print(f"  Precision : {precision:.4f}  (@ threshold {best_thr:.3f})")
    print(f"  Recall    : {recall:.4f}")
    print(f"  F1        : {f1:.4f}")

    metrics = {
        "Test AUPRC"     : float(auprc),
        "Test AUROC"     : float(auroc),
        "Test Precision" : float(precision),
        "Test Recall"    : float(recall),
        "Test F1"        : float(f1),
        "Best Threshold" : best_thr,
        "N Timesteps"    : n_tot,
        "Pct Positive"   : float(100 * n_pos / n_tot),
    }
    out_path = os.path.join(args.output_dir, "metrics.json")
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=4)
    print(f"\n  Metrics saved  → {out_path}")

    if HAS_MATPLOTLIB:
        plt.figure(figsize=(7, 5))
        plt.plot(rec_vals, prec_vals, lw=2, label=f"Tuned GRU-D  AUPRC={auprc:.4f}")
        plt.axhline(y=n_pos/n_tot, color="gray", linestyle="--", label="Random baseline")
        plt.xlabel("Recall"); plt.ylabel("Precision")
        plt.title("PR Curve — Tuned GRU-D (Test Set)")
        plt.legend(); plt.tight_layout()
        pr_path = os.path.join(args.output_dir, "pr_curve_grud_tuned.png")
        plt.savefig(pr_path, dpi=150); plt.close()

        fpr, tpr, _ = roc_curve(all_labels, all_probs)
        plt.figure(figsize=(7, 5))
        plt.plot(fpr, tpr, lw=2, label=f"Tuned GRU-D  AUROC={auroc:.4f}")
        plt.plot([0, 1], [0, 1], "k--")
        plt.xlabel("FPR"); plt.ylabel("TPR")
        plt.title("ROC Curve — Tuned GRU-D (Test Set)")
        plt.legend(); plt.tight_layout()
        roc_path = os.path.join(args.output_dir, "roc_curve_grud_tuned.png")
        plt.savefig(roc_path, dpi=150); plt.close()
        print(f"  Plots saved    → {args.output_dir}/")


if __name__ == "__main__":
    main()
