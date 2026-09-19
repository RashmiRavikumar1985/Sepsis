import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import math
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import average_precision_score, roc_auc_score
from tqdm import tqdm

from src.dataset_grud import PhysioNetDatasetGRUD, collate_fn
from src.model_grud import GRUD

# ──────────────────────────────────────────────────────────────────────
# Deterministic Reproducibility
# ──────────────────────────────────────────────────────────────────────
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        try:
            torch.mps.manual_seed(seed)
        except AttributeError:
            pass
    os.environ['PYTHONHASHSEED'] = str(seed)

set_seed(42)

# ──────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

cand_dirs_1 = [
    os.path.join(PROJECT_ROOT, "physionet2019", "training", "training_setA"),
    os.path.join(PROJECT_ROOT, "physionet2019", "training", "training_setB")
]
cand_dirs_2 = [
    os.path.join(PROJECT_ROOT, "training", "training_setA"),
    os.path.join(PROJECT_ROOT, "training", "training_setB")
]
if os.path.exists(cand_dirs_1[0]):
    DATA_DIRS = cand_dirs_1
elif os.path.exists(cand_dirs_2[0]):
    DATA_DIRS = cand_dirs_2
else:
    raise FileNotFoundError("Could not locate training_setA and training_setB directories.")

CONFIG_PATH = os.path.join(PROJECT_ROOT, "artifacts", "preprocessing_config.json")
SPLITS_PATH = os.path.join(PROJECT_ROOT, "artifacts", "splits.json")
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "checkpoints")
ARTIFACTS_DIR = os.path.join(PROJECT_ROOT, "artifacts")

HIDDEN_SIZE = 64
DROPOUT = 0.3
BATCH_SIZE = 32
LR = 1e-3
EPOCHS = 20
PATIENCE = 5          # early-stopping patience
MAX_SEQ_LEN = 336     # clip very long stays

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")


# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────
def masked_bce_loss(logits, labels, valid_mask, pos_weight):
    """
    Binary cross-entropy computed ONLY on valid (non-padded) hours.
    pos_weight handles the severe class imbalance.
    """
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pos_weight], device=logits.device),
        reduction='none',
    )
    per_hour_loss = criterion(logits, labels)          # (B, T)
    per_hour_loss = per_hour_loss * valid_mask          # zero out padding
    return per_hour_loss.sum() / (valid_mask.sum() + 1e-9)


@torch.no_grad()
def evaluate(model, loader, pos_weight):
    """Returns loss, AUROC, and AUPRC over an entire DataLoader."""
    model.eval()
    all_probs, all_labels = [], []
    total_loss = 0.0
    total_hours = 0

    for values, mask, delta, static_features, labels, valid_mask in loader:
        values = values.to(DEVICE)
        mask = mask.to(DEVICE)
        delta = delta.to(DEVICE)
        static_features = static_features.to(DEVICE)
        labels = labels.to(DEVICE)
        valid_mask = valid_mask.to(DEVICE)

        logits = model(values, mask, delta, static_features)
        loss = masked_bce_loss(logits, labels, valid_mask, pos_weight)
        n_valid = valid_mask.sum().item()
        total_loss += loss.item() * n_valid
        total_hours += n_valid

        probs = torch.sigmoid(logits)
        # Flatten and keep only valid hours
        flat_valid = valid_mask.bool().view(-1)
        all_probs.append(probs.view(-1)[flat_valid].cpu())
        all_labels.append(labels.view(-1)[flat_valid].cpu())

    all_probs = torch.cat(all_probs).numpy()
    all_labels = torch.cat(all_labels).numpy()

    avg_loss = total_loss / (total_hours + 1e-9)

    # Guard against single-class batches
    if len(set(all_labels.astype(int))) < 2:
        return avg_loss, 0.0, 0.0

    auroc = roc_auc_score(all_labels, all_probs)
    auprc = average_precision_score(all_labels, all_probs)
    return avg_loss, auroc, auprc


# ──────────────────────────────────────────────────────────────────────
# Main training loop
# ──────────────────────────────────────────────────────────────────────
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--subset', type=int, default=None, help='Number of patients to subset training/validation to (for faster CPU runs)')
    args = parser.parse_args()

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # Load config and splits
    with open(CONFIG_PATH, 'r') as f:
        config = json.load(f)
    with open(SPLITS_PATH, 'r') as f:
        splits = json.load(f)

    pos_weight = config['class_weight']
    input_size = len(config['dynamic_features'])
    static_size = len(config['static_features'])

    print(f"Device: {DEVICE}")
    print(f"Input features: {input_size}")
    print(f"Static features: {static_size}")
    print(f"Positive class weight: {pos_weight:.2f}")

    train_splits = splits['train']
    val_splits = splits['val']
    if args.subset is not None:
        print(f"Subsetting dataset to {args.subset} patients for faster CPU training...")
        # Maintain stratification style by just slicing a subset
        train_splits = train_splits[:args.subset]
        val_splits = val_splits[:max(100, int(args.subset * 0.2))]

    # Datasets
    train_ds = PhysioNetDatasetGRUD(DATA_DIRS, train_splits, CONFIG_PATH, MAX_SEQ_LEN)
    val_ds = PhysioNetDatasetGRUD(DATA_DIRS, val_splits, CONFIG_PATH, MAX_SEQ_LEN)

    # Deterministic DataLoaders
    g = torch.Generator()
    g.manual_seed(42)

    use_pin_memory = (DEVICE.type == "cuda")

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        collate_fn=collate_fn, num_workers=0, pin_memory=use_pin_memory,
        generator=g
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        collate_fn=collate_fn, num_workers=0, pin_memory=use_pin_memory,
        generator=g
    )

    print(f"Train patients: {len(train_ds)}, Val patients: {len(val_ds)}")

    # Model
    model = GRUD(input_size, static_size, HIDDEN_SIZE, DROPOUT).to(DEVICE)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=2
    )

    # Training state and history logging
    best_auprc = 0.0
    best_auroc = 0.0
    best_epoch = 0
    patience_counter = 0
    suffix = f"subset_{args.subset}" if args.subset is not None else "full"
    ckpt_path = os.path.join(CHECKPOINT_DIR, f"best_grud_{suffix}.pt")
    log_path = os.path.join(CHECKPOINT_DIR, f"training_log_{suffix}.json")

    history = {
        "config": {
            "hidden_size": HIDDEN_SIZE,
            "dropout": DROPOUT,
            "batch_size": BATCH_SIZE,
            "lr": LR,
            "epochs": EPOCHS,
            "patience": PATIENCE,
            "max_seq_len": MAX_SEQ_LEN,
            "subset": args.subset,
            "input_size": input_size,
            "static_size": static_size,
            "pos_weight": pos_weight,
            "seed": 42,
        },
        "epochs": []
    }

    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_loss = 0.0
        epoch_hours = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}")
        for values, mask, delta, static_features, labels, valid_mask in pbar:
            values = values.to(DEVICE)
            mask = mask.to(DEVICE)
            delta = delta.to(DEVICE)
            static_features = static_features.to(DEVICE)
            labels = labels.to(DEVICE)
            valid_mask = valid_mask.to(DEVICE)

            logits = model(values, mask, delta, static_features)
            loss = masked_bce_loss(logits, labels, valid_mask, pos_weight)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            n_valid = valid_mask.sum().item()
            epoch_loss += loss.item() * n_valid
            epoch_hours += n_valid
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_train_loss = epoch_loss / (epoch_hours + 1e-9)

        # Validation
        val_loss, val_auroc, val_auprc = evaluate(model, val_loader, pos_weight)
        scheduler.step(val_auprc)

        epoch_record = {
            "epoch": epoch,
            "train_loss": avg_train_loss,
            "val_loss": val_loss,
            "val_auroc": val_auroc,
            "val_auprc": val_auprc,
            "lr": optimizer.param_groups[0]['lr']
        }
        history["epochs"].append(epoch_record)

        print(f"  Train Loss: {avg_train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"Val AUROC: {val_auroc:.4f} | "
              f"Val AUPRC: {val_auprc:.4f}")

        # Early stopping on AUPRC
        if val_auprc > best_auprc:
            best_auprc = val_auprc
            best_auroc = val_auroc
            best_epoch = epoch
            patience_counter = 0
            ckpt_dict = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_auprc': val_auprc,
                'val_auroc': val_auroc,
                'config': history["config"]
            }
            torch.save(ckpt_dict, ckpt_path)
            torch.save(ckpt_dict, os.path.join(ARTIFACTS_DIR, "baseline_grud.pt"))
            print(f"  ** New best AUPRC={val_auprc:.4f}  (saved checkpoint as {os.path.basename(ckpt_path)})")
        else:
            patience_counter += 1
            print(f"  No improvement ({patience_counter}/{PATIENCE})")
            if patience_counter >= PATIENCE:
                print("Early stopping triggered.")
                break

    # Save training metrics JSON log
    history["best_epoch"] = best_epoch
    history["best_val_auprc"] = best_auprc
    history["best_val_auroc"] = best_auroc

    # ──────────────────────────────────────────────────────────────────
    # Final Test Set Evaluation using Best Model Checkpoint
    # ──────────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Running Final Evaluation on Held-Out Test Set...")
    print("=" * 60)
    test_splits = splits.get('test', [])
    if len(test_splits) > 0 and os.path.exists(ckpt_path):
        # Load best model checkpoint
        best_ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
        model.load_state_dict(best_ckpt['model_state_dict'])
        
        test_ds = PhysioNetDatasetGRUD(DATA_DIRS, test_splits, CONFIG_PATH, MAX_SEQ_LEN)
        test_loader = DataLoader(
            test_ds, batch_size=BATCH_SIZE, shuffle=False,
            collate_fn=collate_fn, num_workers=0, pin_memory=use_pin_memory,
            generator=g
        )
        test_loss, test_auroc, test_auprc = evaluate(model, test_loader, pos_weight)
        history["test_results"] = {
            "test_patients": len(test_ds),
            "test_loss": test_loss,
            "test_auroc": test_auroc,
            "test_auprc": test_auprc
        }
        print(f"Test Patients: {len(test_ds)} | Test AUROC: {test_auroc:.4f} | Test AUPRC: {test_auprc:.4f}")
    else:
        print("No test splits found or checkpoint missing.")

    # Save final JSON history
    with open(log_path, 'w') as f:
        json.dump(history, f, indent=4)
    print(f"Training metrics log saved to: {log_path}")

    print(f"\nTraining complete. Best Val AUPRC: {best_auprc:.4f} | Best Val AUROC: {best_auroc:.4f}")


if __name__ == "__main__":
    main()
