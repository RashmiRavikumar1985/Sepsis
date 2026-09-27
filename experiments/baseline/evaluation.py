"""
baseline/evaluation.py — Proper test-set evaluation for the saved GRU-D checkpoint.

Runs full inference on the test split and records AUPRC, AUROC,
Precision, Recall, and F1 (using the F1-optimal threshold).
"""

import os
import sys
import json

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    f1_score,
)

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from src.dataset_grud import PhysioNetDatasetGRUD, collate_fn
from src.model_grud import GRUD


def evaluate_baseline():
    print("=== Evaluating GRU-D Baseline (Test Set) ===")

    results_dir  = os.path.join(project_root, "experiments", "results", "baseline")
    artifacts_dir = os.path.join(project_root, "artifacts")
    os.makedirs(results_dir, exist_ok=True)

    # ── Find checkpoint ───────────────────────────────────────────────
    ckpt_path = os.path.join(project_root, "checkpoints", "best_grud_full.pt")
    if not os.path.exists(ckpt_path):
        ckpt_path = os.path.join(artifacts_dir, "baseline_grud.pt")
    if not os.path.exists(ckpt_path):
        print(f"ERROR: No GRU-D checkpoint found. Looked at:\n  {ckpt_path}")
        return

    print(f"Checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    cfg         = ckpt["config"]
    prep_config = ckpt["preprocessing_config"]
    pos_weight  = prep_config["class_weight"]

    print(f"  Saved epoch  : {ckpt['epoch']}")
    print(f"  Val AUPRC    : {ckpt['val_auprc']:.4f}")
    print(f"  Val AUROC    : {ckpt['val_auroc']:.4f}")
    print(f"  pos_weight   : {pos_weight:.4f}")

    # ── Device ────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device       : {device}")

    # ── Data paths ────────────────────────────────────────────────────
    splits_path         = os.path.join(project_root, "artifacts", "splits.json")
    preprocess_cfg_path = os.path.join(project_root, "artifacts", "preprocessing_config.json")

    cand_1 = [
        os.path.join(project_root, "physionet2019", "training", "training_setA"),
        os.path.join(project_root, "physionet2019", "training", "training_setB"),
    ]
    cand_2 = [
        os.path.join(project_root, "training", "training_setA"),
        os.path.join(project_root, "training", "training_setB"),
    ]
    # sepsis.1/physionet2019 lives one level above sepsis/
    sepsis1_root = os.path.dirname(project_root)
    cand_3 = [
        os.path.join(sepsis1_root, "physionet2019", "training", "training_setA"),
        os.path.join(sepsis1_root, "physionet2019", "training", "training_setB"),
    ]
    if os.path.exists(cand_1[0]):
        data_dirs = cand_1
    elif os.path.exists(cand_2[0]):
        data_dirs = cand_2
    elif os.path.exists(cand_3[0]):
        data_dirs = cand_3
    else:
        raise FileNotFoundError(
            f"Cannot find training data. Tried:\n  {cand_1[0]}\n  {cand_2[0]}\n  {cand_3[0]}"
        )

    with open(splits_path) as f:
        splits = json.load(f)
    test_ids = splits.get("test", splits.get("val", []))

    # ── Dataset ───────────────────────────────────────────────────────
    max_seq_len = cfg.get("max_seq_len", 336)
    batch_size  = cfg.get("batch_size", 64)

    test_ds = PhysioNetDatasetGRUD(data_dirs, test_ids, preprocess_cfg_path, max_seq_len)
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=0,
    )
    print(f"  Test patients: {len(test_ds)}")

    # ── Model ─────────────────────────────────────────────────────────
    input_size   = len(prep_config["dynamic_features"])
    static_size  = len(prep_config["static_features"])
    hidden_size  = cfg.get("hidden_size", 128)
    dropout      = cfg.get("dropout", 0.2)
    attn_cfg     = cfg.get("attention", {})
    attn_config  = attn_cfg if attn_cfg.get("enabled", False) else None

    model = GRUD(input_size, static_size, hidden_size, dropout, attn_config).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

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
    print(f"  Timesteps    : {n_tot:,}  |  Positives: {n_pos:,} ({100*n_pos/n_tot:.2f}%)")

    # ── Metrics ───────────────────────────────────────────────────────
    auprc = average_precision_score(all_labels, all_probs)
    auroc = roc_auc_score(all_labels, all_probs)

    # F1-optimal threshold
    prec_vals, rec_vals, thresholds = precision_recall_curve(all_labels, all_probs)
    f1_vals  = 2 * prec_vals * rec_vals / (prec_vals + rec_vals + 1e-9)
    best_idx = int(np.argmax(f1_vals[:-1]))
    best_thr = float(thresholds[best_idx])

    bin_preds = (all_probs >= best_thr).astype(int)
    precision = precision_score(all_labels, bin_preds, zero_division=0)
    recall    = recall_score(all_labels, bin_preds, zero_division=0)
    f1        = f1_score(all_labels, bin_preds, zero_division=0)

    print("\n=== TEST RESULTS ===")
    print(f"  AUPRC     : {auprc:.4f}")
    print(f"  AUROC     : {auroc:.4f}")
    print(f"  Precision : {precision:.4f}  (@ threshold {best_thr:.3f})")
    print(f"  Recall    : {recall:.4f}")
    print(f"  F1        : {f1:.4f}")

    # ── Save ──────────────────────────────────────────────────────────
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
    out_path = os.path.join(results_dir, "metrics.json")
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=4)
    print(f"\n  Saved → {out_path}")


if __name__ == "__main__":
    evaluate_baseline()
