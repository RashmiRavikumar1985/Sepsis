"""
train_grud_tuned.py — Tuned GRU-D training script for FedSepsis-KG.

Uses src/config_grud_tuned.json.
Saves checkpoint  → checkpoints/best_grud_tuned_full.pt
Saves artifact    → artifacts/tuned_grud.pt
Saves log         → checkpoints/training_log_tuned_full.json
Saves metrics     → experiments/results/grud_tuned/metrics.json

Does NOT touch any baseline GRU-D files.

Usage:
    python src/train_grud_tuned.py
    python src/train_grud_tuned.py --subset 2000
"""

import os
import sys
import json
import random
import argparse

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    f1_score,
)
from tqdm import tqdm

_THIS_FILE   = os.path.abspath(__file__)
SEPSIS_ROOT  = os.path.dirname(os.path.dirname(_THIS_FILE))
PROJECT_ROOT = os.path.dirname(SEPSIS_ROOT)
sys.path.insert(0, SEPSIS_ROOT)

from src.dataset_grud import PhysioNetDatasetGRUD, collate_fn
from src.model_grud import GRUD


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def masked_bce_loss(logits, labels, valid_mask, pos_weight):
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pos_weight], device=logits.device),
        reduction="none",
    )
    per_hour = criterion(logits, labels) * valid_mask.float()
    return per_hour.sum() / (valid_mask.float().sum() + 1e-9)


@torch.no_grad()
def evaluate(model, loader, pos_weight, device):
    model.eval()
    all_probs, all_labels = [], []
    total_loss, total_hours = 0.0, 0
    for values, mask, delta, static_feats, labels, valid_mask in loader:
        values       = values.to(device)
        mask         = mask.to(device)
        delta        = delta.to(device)
        static_feats = static_feats.to(device)
        labels       = labels.to(device)
        valid_mask   = valid_mask.to(device)
        logits = model(values, mask, delta, static_feats, valid_mask)
        loss   = masked_bce_loss(logits, labels, valid_mask, pos_weight)
        n_valid = int(valid_mask.sum().item())
        total_loss  += loss.item() * n_valid
        total_hours += n_valid
        probs = torch.sigmoid(logits)
        flat_valid = valid_mask.bool().view(-1)
        all_probs.append(probs.view(-1)[flat_valid].cpu())
        all_labels.append(labels.view(-1)[flat_valid].cpu())
    all_probs  = torch.cat(all_probs).numpy()
    all_labels = torch.cat(all_labels).numpy()
    avg_loss   = total_loss / (total_hours + 1e-9)
    if len(set(all_labels.astype(int))) < 2:
        return avg_loss, 0.0, 0.0
    return avg_loss, roc_auc_score(all_labels, all_probs), average_precision_score(all_labels, all_probs)


@torch.no_grad()
def evaluate_full(model, loader, pos_weight, device):
    model.eval()
    all_probs, all_labels = [], []
    total_loss, total_hours = 0.0, 0
    for values, mask, delta, static_feats, labels, valid_mask in loader:
        values       = values.to(device)
        mask         = mask.to(device)
        delta        = delta.to(device)
        static_feats = static_feats.to(device)
        labels       = labels.to(device)
        valid_mask   = valid_mask.to(device)
        logits = model(values, mask, delta, static_feats, valid_mask)
        loss   = masked_bce_loss(logits, labels, valid_mask, pos_weight)
        n_valid = int(valid_mask.sum().item())
        total_loss  += loss.item() * n_valid
        total_hours += n_valid
        probs = torch.sigmoid(logits)
        flat_valid = valid_mask.bool().view(-1)
        all_probs.append(probs.view(-1)[flat_valid].cpu())
        all_labels.append(labels.view(-1)[flat_valid].cpu())
    all_probs  = torch.cat(all_probs).numpy()
    all_labels = torch.cat(all_labels).numpy()
    avg_loss   = total_loss / (total_hours + 1e-9)
    if len(set(all_labels.astype(int))) < 2:
        return avg_loss, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5
    auroc = roc_auc_score(all_labels, all_probs)
    auprc = average_precision_score(all_labels, all_probs)
    prec_vals, rec_vals, thresholds = precision_recall_curve(all_labels, all_probs)
    f1_vals  = 2 * prec_vals * rec_vals / (prec_vals + rec_vals + 1e-9)
    best_idx = int(np.argmax(f1_vals[:-1]))
    best_thr = float(thresholds[best_idx])
    bin_preds = (all_probs >= best_thr).astype(int)
    precision = precision_score(all_labels, bin_preds, zero_division=0)
    recall    = recall_score(all_labels, bin_preds, zero_division=0)
    f1        = f1_score(all_labels, bin_preds, zero_division=0)
    return avg_loss, auroc, auprc, precision, recall, f1, best_thr


def main():
    parser = argparse.ArgumentParser(description="Train Tuned GRU-D for sepsis prediction")
    parser.add_argument(
        "--config",
        default=os.path.join(os.path.dirname(_THIS_FILE), "config_grud_tuned.json"),
        help="Path to tuned GRU-D config JSON",
    )
    parser.add_argument("--subset", type=int, default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = json.load(f)

    HIDDEN_SIZE    = cfg["hidden_size"]
    DROPOUT        = cfg["dropout"]
    BATCH_SIZE     = cfg["batch_size"]
    LR             = cfg["learning_rate"]
    EPOCHS         = cfg["epochs"]
    PATIENCE       = cfg["patience"]
    MAX_SEQ_LEN    = cfg["max_seq_len"]
    SEED           = cfg["seed"]
    GRAD_CLIP      = cfg.get("grad_clip", 1.0)
    SCHED_FACTOR   = cfg.get("scheduler_factor", 0.5)
    SCHED_PATIENCE = cfg.get("scheduler_patience", 3)

    set_seed(SEED)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    preprocess_cfg_path = os.path.join(SEPSIS_ROOT, "artifacts", "preprocessing_config.json")
    splits_path         = os.path.join(SEPSIS_ROOT, "artifacts", "splits.json")
    checkpoint_dir      = os.path.join(SEPSIS_ROOT, "checkpoints")
    results_dir         = os.path.join(SEPSIS_ROOT, "experiments", "results", "grud_tuned")
    artifacts_dir       = os.path.join(SEPSIS_ROOT, "artifacts")

    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    cand_1 = [
        os.path.join(PROJECT_ROOT, "physionet2019", "training", "training_setA"),
        os.path.join(PROJECT_ROOT, "physionet2019", "training", "training_setB"),
    ]
    cand_2 = [
        os.path.join(PROJECT_ROOT, "training", "training_setA"),
        os.path.join(PROJECT_ROOT, "training", "training_setB"),
    ]
    data_dirs = cand_1 if os.path.exists(cand_1[0]) else cand_2

    with open(preprocess_cfg_path) as f:
        prep_config = json.load(f)
    with open(splits_path) as f:
        splits = json.load(f)

    pos_weight  = prep_config["class_weight"]
    input_size  = len(prep_config["dynamic_features"])
    static_size = len(prep_config["static_features"])

    print(f"Dynamic features     : {input_size}")
    print(f"Static features      : {static_size}")
    print(f"Positive class weight: {pos_weight:.2f}")
    print(f"Config               : {args.config}")

    train_ids = splits["train"]
    val_ids   = splits["val"]
    test_ids  = splits.get("test", [])

    if args.subset is not None:
        print(f"Subsetting to {args.subset} training patients")
        train_ids = train_ids[:args.subset]
        val_ids   = val_ids[:max(100, args.subset // 5)]

    train_ds = PhysioNetDatasetGRUD(data_dirs, train_ids, preprocess_cfg_path, MAX_SEQ_LEN)
    val_ds   = PhysioNetDatasetGRUD(data_dirs, val_ids,   preprocess_cfg_path, MAX_SEQ_LEN)

    g = torch.Generator()
    g.manual_seed(SEED)
    pin = device.type == "cuda"

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              collate_fn=collate_fn, num_workers=0, pin_memory=pin, generator=g)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              collate_fn=collate_fn, num_workers=0, pin_memory=pin)

    print(f"Train: {len(train_ds)} patients | Val: {len(val_ds)} patients")

    attn_cfg = cfg.get("attention", {})
    attention_config = attn_cfg if attn_cfg.get("enabled", False) else None
    model = GRUD(input_size, static_size, HIDDEN_SIZE, DROPOUT, attention_config).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")
    if attention_config:
        print(f"Attention: {attention_config['num_heads']} heads, "
              f"dropout={attention_config.get('attention_dropout')}, "
              f"temp={attention_config.get('temperature', 1.0)}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=cfg.get("weight_decay", 1e-4))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=SCHED_FACTOR, patience=SCHED_PATIENCE
    )
    print(f"Optimizer: AdamW | LR: {LR} | Scheduler: ReduceLROnPlateau")

    best_auprc, best_auroc, best_epoch = 0.0, 0.0, 0
    patience_counter = 0
    suffix    = f"subset_{args.subset}" if args.subset else "full"
    ckpt_path = os.path.join(checkpoint_dir, f"best_grud_tuned_{suffix}.pt")
    log_path  = os.path.join(checkpoint_dir, f"training_log_tuned_{suffix}.json")

    history = {
        "config": {
            **cfg,
            "input_size": input_size,
            "static_size": static_size,
            "pos_weight": pos_weight,
            "subset": args.subset,
            "config_file": args.config,
        },
        "epochs": [],
    }

    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_loss, epoch_hours = 0.0, 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}")
        for values, mask, delta, static_feats, labels, valid_mask in pbar:
            values       = values.to(device)
            mask         = mask.to(device)
            delta        = delta.to(device)
            static_feats = static_feats.to(device)
            labels       = labels.to(device)
            valid_mask   = valid_mask.to(device)
            logits = model(values, mask, delta, static_feats, valid_mask)
            loss   = masked_bce_loss(logits, labels, valid_mask, pos_weight)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            n = int(valid_mask.sum().item())
            epoch_loss  += loss.item() * n
            epoch_hours += n
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_train_loss = epoch_loss / (epoch_hours + 1e-9)
        val_loss, val_auroc, val_auprc = evaluate(model, val_loader, pos_weight, device)

        lr_before = optimizer.param_groups[0]["lr"]
        scheduler.step(val_auprc)
        lr_after  = optimizer.param_groups[0]["lr"]
        lr_note   = f" | LR: {lr_after:.2e}" + (" (reduced)" if lr_after < lr_before else "")

        print(f"  Epoch {epoch:02d} | Train Loss: {avg_train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | Val AUROC: {val_auroc:.4f} | "
              f"Val AUPRC: {val_auprc:.4f}{lr_note}")

        history["epochs"].append({
            "epoch": epoch, "train_loss": avg_train_loss,
            "val_loss": val_loss, "val_auroc": val_auroc,
            "val_auprc": val_auprc, "lr": lr_after,
        })

        if val_auprc > best_auprc:
            best_auprc, best_auroc, best_epoch = val_auprc, val_auroc, epoch
            patience_counter = 0
            ckpt = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_auprc": val_auprc,
                "val_auroc": val_auroc,
                "config": history["config"],
                "preprocessing_config": prep_config,
                "dynamic_features": prep_config["dynamic_features"],
                "static_features_list": prep_config["static_features"],
                "input_size": input_size,
                "static_size": static_size,
            }
            torch.save(ckpt, ckpt_path)
            torch.save(ckpt, os.path.join(artifacts_dir, "tuned_grud.pt"))
            print(f"  ** New best AUPRC={val_auprc:.4f} — checkpoint saved")
        else:
            patience_counter += 1
            print(f"  No improvement ({patience_counter}/{PATIENCE})")
            if patience_counter >= PATIENCE:
                print("Early stopping triggered.")
                break

    history["best_epoch"]     = best_epoch
    history["best_val_auprc"] = best_auprc
    history["best_val_auroc"] = best_auroc

    # ── Test evaluation ───────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Final Evaluation on Held-Out Test Set")
    print("=" * 60)

    if test_ids and os.path.exists(ckpt_path):
        best_ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(best_ckpt["model_state_dict"])
        test_ds = PhysioNetDatasetGRUD(data_dirs, test_ids, preprocess_cfg_path, MAX_SEQ_LEN)
        test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                                 collate_fn=collate_fn, num_workers=0, pin_memory=pin)

        test_loss, test_auroc, test_auprc, test_prec, test_rec, test_f1, best_thr = \
            evaluate_full(model, test_loader, pos_weight, device)

        history["test_results"] = {
            "test_patients"  : len(test_ds),
            "test_loss"      : test_loss,
            "test_auroc"     : test_auroc,
            "test_auprc"     : test_auprc,
            "test_precision" : test_prec,
            "test_recall"    : test_rec,
            "test_f1"        : test_f1,
            "best_threshold" : best_thr,
        }

        metrics = {
            "Test AUPRC"     : float(test_auprc),
            "Test AUROC"     : float(test_auroc),
            "Test Precision" : float(test_prec),
            "Test Recall"    : float(test_rec),
            "Test F1"        : float(test_f1),
            "Best Threshold" : best_thr,
            "Best Val AUPRC" : best_auprc,
            "Best Val AUROC" : best_auroc,
            "Best Epoch"     : best_epoch,
        }
        with open(os.path.join(results_dir, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=4)

        print(f"Test patients  : {len(test_ds)}")
        print(f"Test AUROC     : {test_auroc:.4f}")
        print(f"Test AUPRC     : {test_auprc:.4f}")
        print(f"Test Precision : {test_prec:.4f}  (@ threshold {best_thr:.3f})")
        print(f"Test Recall    : {test_rec:.4f}")
        print(f"Test F1        : {test_f1:.4f}")
        print(f"Metrics saved  → {results_dir}/metrics.json")
    else:
        print("No test splits or checkpoint missing — skipping.")

    with open(log_path, "w") as f:
        json.dump(history, f, indent=4)
    print(f"\nTraining log → {log_path}")
    print(f"Best Val AUPRC: {best_auprc:.4f} | Best Val AUROC: {best_auroc:.4f} | Epoch: {best_epoch}")


if __name__ == "__main__":
    main()
