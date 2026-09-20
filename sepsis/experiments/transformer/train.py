import os
import sys
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import average_precision_score, roc_auc_score
from tqdm import tqdm

try:
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except Exception as e:
    print(f"Warning: Could not import matplotlib: {e}")
    HAS_MATPLOTLIB = False

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sepsis_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, sepsis_root)

from src.dataset_grud import PhysioNetDatasetGRUD, collate_fn
from experiments.transformer.model import TemporalTransformer

def set_seed(seed=42):
    import numpy as np
    import random
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_valid = 0

    all_preds = []
    all_labels = []

    for values, mask, delta, static_features, labels, valid_mask in loader:
        values = values.to(device)
        mask = mask.to(device)
        delta = delta.to(device)
        static_features = static_features.to(device)
        labels = labels.to(device)
        valid_mask = valid_mask.to(device)

        logits = model(values, mask, delta, static_features, valid_mask=valid_mask)
        loss_matrix = criterion(logits, labels)

        valid = valid_mask.float()
        loss = (loss_matrix * valid).sum() / valid.sum().clamp(min=1.0)

        n_valid = int(valid.sum().item())
        total_loss += loss.item() * n_valid
        total_valid += n_valid

        probs = torch.sigmoid(logits)
        valid_bool = valid_mask.bool()

        all_preds.append(probs[valid_bool].cpu())
        all_labels.append(labels[valid_bool].cpu())

    if len(all_preds) > 0:
        all_preds = torch.cat(all_preds).numpy()
        all_labels = torch.cat(all_labels).numpy()
        avg_loss = total_loss / max(total_valid, 1)
        auprc = average_precision_score(all_labels, all_preds)
        auroc = roc_auc_score(all_labels, all_preds)
    else:
        avg_loss, auroc, auprc = 0.0, 0.0, 0.0

    return avg_loss, auroc, auprc

def train_transformer():
    print("=== Training Temporal Transformer (35-Feature Baseline) ===")
    
    # Load configs with clear ownership
    config_path = os.path.join(os.path.dirname(__file__), "config.json")
    with open(config_path, 'r') as f:
        config = json.load(f)
        
    preprocess_cfg = os.path.join(sepsis_root, "artifacts", "preprocessing_config.json")
    with open(preprocess_cfg, 'r') as f:
        prep_config = json.load(f)
        
    seed = config.get('seed', 42)
    set_seed(seed)
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Load Splits
    splits_path = os.path.join(sepsis_root, "artifacts", "splits.json")
    with open(splits_path, 'r') as f:
        splits = json.load(f)
        
    # Dataset Paths
    cand_dirs_1 = [
        os.path.join(project_root, "physionet2019", "training", "training_setA"),
        os.path.join(project_root, "physionet2019", "training", "training_setB")
    ]
    cand_dirs_2 = [
        os.path.join(project_root, "training", "training_setA"),
        os.path.join(project_root, "training", "training_setB")
    ]
    if os.path.exists(cand_dirs_1[0]):
        data_dirs = cand_dirs_1
    elif os.path.exists(cand_dirs_2[0]):
        data_dirs = cand_dirs_2
    else:
        raise FileNotFoundError("Could not locate training_setA and training_setB data directories.")

    # CLEAN CONFIG OWNERSHIP: preprocessing_config.json owns data schema
    dynamic_features = prep_config["dynamic_features"]
    static_features = prep_config["static_features"] 
    input_size = len(dynamic_features)
    static_size = len(static_features)
    pos_weight = prep_config.get('class_weight', 54.54)

    # Baseline validation assertions
    assert input_size == 35, f"Expected 35 dynamic features for baseline, got {input_size}"
    assert static_size == 5, f"Expected 5 static features for baseline, got {static_size}"
    assert "ICULOS" in dynamic_features, "ICULOS must be present in 35-feature baseline"
    assert input_size * 3 == 105, f"Expected 105 Transformer input channels, got {input_size * 3}"

    print(f"Dynamic features: {input_size} (includes ICULOS)")
    print(f"Static features: {static_size}")
    print(f"Transformer input channels: {input_size * 3}")
    print(f"ICULOS index: {dynamic_features.index('ICULOS')}")
    print(f"Positive class weight: {pos_weight:.2f}")

    train_ids = splits['train']
    val_ids = splits['val']
    test_ids = splits['test']

    g = torch.Generator()
    g.manual_seed(seed)
    
    # Device-specific optimization for data loading
    use_pin_memory = (device.type == "cuda")
    
    train_ds = PhysioNetDatasetGRUD(data_dirs, train_ids, preprocess_cfg, config['max_sequence_length'])
    val_ds = PhysioNetDatasetGRUD(data_dirs, val_ids, preprocess_cfg, config['max_sequence_length'])
    test_ds = PhysioNetDatasetGRUD(data_dirs, test_ids, preprocess_cfg, config['max_sequence_length'])
    
    train_loader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True, collate_fn=collate_fn, 
                             generator=g, pin_memory=use_pin_memory, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=config['batch_size'], shuffle=False, collate_fn=collate_fn,
                           pin_memory=use_pin_memory, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=config['batch_size'], shuffle=False, collate_fn=collate_fn,
                            pin_memory=use_pin_memory, num_workers=0)
    
    model = TemporalTransformer(
        input_size=input_size,                      # From preprocessing config
        static_size=static_size,                    # From preprocessing config  
        d_model=config['d_model'],                  # From model config
        n_heads=config['n_heads'],                  # From model config
        num_layers=config['num_layers'],            # From model config
        dim_feedforward=config['dim_feedforward'],  # From model config
        dropout=config['dropout']                   # From model config
    ).to(device)
    
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, dtype=torch.float32).to(device), reduction='none')
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    
    best_val_auprc = 0.0
    best_val_auroc = 0.0
    best_epoch = 0
    patience_counter = 0

    results_dir = os.path.join(project_root, "experiments", "results", "transformer")
    artifacts_dir = os.path.join(project_root, "artifacts")
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(artifacts_dir, exist_ok=True)
    
    train_losses = []
    val_losses = []

    for epoch in range(config['epochs']):
        model.train()
        train_loss_sum = 0.0
        train_valid_hours = 0
        
        for values, mask, delta, static_features, labels, valid_mask in tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['epochs']}"):
            values, mask = values.to(device), mask.to(device)
            delta, static_features = delta.to(device), static_features.to(device)
            labels, valid_mask = labels.to(device), valid_mask.to(device)
            
            optimizer.zero_grad()
            logits = model(values, mask, delta, static_features, valid_mask=valid_mask)
            loss_matrix = criterion(logits, labels)
            
            valid = valid_mask.float()
            loss = (loss_matrix * valid).sum() / valid.sum().clamp(min=1.0)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            n_valid = int(valid.sum().item())
            train_loss_sum += loss.item() * n_valid
            train_valid_hours += n_valid
            
        avg_train_loss = train_loss_sum / max(train_valid_hours, 1)
        train_losses.append(avg_train_loss)
            
        # Validation evaluation using reusable evaluate() function
        val_loss, val_auroc, val_auprc = evaluate(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        
        print(f"Epoch {epoch+1:02d}: Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f} | Val AUPRC: {val_auprc:.4f} | Val AUROC: {val_auroc:.4f}")
        
        if val_auprc > best_val_auprc:
            best_val_auprc = val_auprc
            best_val_auroc = val_auroc
            best_epoch = epoch + 1
            patience_counter = 0
            
            # Save checkpoint with complete metadata for reproducibility
            checkpoint_data = {
                "model_state_dict": model.state_dict(),
                "model_config": config,                          # Training hyperparameters
                "preprocessing_config": prep_config,             # Complete preprocessing snapshot
                "preprocessing_config_path": preprocess_cfg,     # Path for reference
                "dynamic_features": dynamic_features,            # Feature list snapshot
                "static_features": static_features,              # Static feature list
                "input_size": input_size,                        # 35 
                "static_size": static_size,                      # 5
                "input_channels": input_size * 3,                # 105
                "best_val_auprc": float(best_val_auprc),
                "best_val_auroc": float(best_val_auroc),
                "best_epoch": best_epoch,
                "seed": seed,
                "pos_weight": float(pos_weight)
            }
            torch.save(checkpoint_data, os.path.join(artifacts_dir, "baseline_transformer.pt"))
            torch.save(model.state_dict(), os.path.join(results_dir, "best_model.pt"))
        else:
            patience_counter += 1
            if patience_counter >= config['patience']:
                print(f"Early stopping triggered at epoch {epoch+1}!")
                break

    # Load best model for Held-out Test Evaluation with verification
    print("\nEvaluating Best Model on Held-out Test Set...")
    checkpoint = torch.load(os.path.join(artifacts_dir, "baseline_transformer.pt"), map_location=device)
    
    # Verify checkpoint consistency
    assert checkpoint["input_size"] == input_size, f"Checkpoint input_size mismatch: {checkpoint['input_size']} vs {input_size}"
    assert checkpoint["static_size"] == static_size, f"Checkpoint static_size mismatch: {checkpoint['static_size']} vs {static_size}"
    assert checkpoint["input_channels"] == 105, f"Checkpoint input_channels mismatch: {checkpoint['input_channels']} vs 105"
    assert checkpoint["dynamic_features"] == dynamic_features, "Checkpoint dynamic_features mismatch"
    
    model.load_state_dict(checkpoint["model_state_dict"])

    test_loss, test_auroc, test_auprc = evaluate(model, test_loader, criterion, device)
    print(f"=== Baseline Transformer Test Results ===")
    print(f"  Test Loss:  {test_loss:.4f}")
    print(f"  Test AUROC: {test_auroc:.4f}")
    print(f"  Test AUPRC: {test_auprc:.4f}")

    # Save comprehensive baseline metrics
    metrics = {
        "model": "Transformer Baseline (35-Feature)",
        "input_features": input_size,
        "static_features": static_size,
        "input_channels": input_size * 3,
        "includes_iculos": True,
        "iculos_index": dynamic_features.index("ICULOS"),
        "best_epoch": best_epoch,
        "best_val_auprc": float(best_val_auprc),
        "best_val_auroc": float(best_val_auroc),
        "test_loss": float(test_loss),
        "test_auprc": float(test_auprc),
        "test_auroc": float(test_auroc),
        "pos_weight": float(pos_weight),
        "seed": seed,
        "config_source": "Clean ownership: preprocessing_config.json for data schema, config.json for hyperparams"
    }
    metrics_path = os.path.join(artifacts_dir, "baseline_metrics.json")
    with open(metrics_path, "w") as fp:
        json.dump(metrics, fp, indent=4)
    print(f"Saved baseline metrics → {metrics_path}")

    # Plot losses
    if HAS_MATPLOTLIB:
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(train_losses) + 1), train_losses, label='Train Loss')
        plt.plot(range(1, len(val_losses) + 1), val_losses, label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Transformer Training and Validation Loss')
        plt.legend()
        plot_dir = os.path.join(os.path.dirname(__file__), "plots")
        os.makedirs(plot_dir, exist_ok=True)
        plt.savefig(os.path.join(plot_dir, "loss_curve.png"))
        plt.close()

if __name__ == "__main__":
    train_transformer()
