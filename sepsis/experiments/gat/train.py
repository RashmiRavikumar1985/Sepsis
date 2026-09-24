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

# Derive paths from this file's location
# __file__ = f:\Sepsis\sepsis.1\sepsis\experiments\gat\train.py
# dirname once  = f:\Sepsis\sepsis.1\sepsis\experiments\gat
# dirname twice = f:\Sepsis\sepsis.1\sepsis\experiments
# dirname thrice = f:\Sepsis\sepsis.1\sepsis  (project_root / sepsis_root)
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# project_root = f:\Sepsis\sepsis.1\sepsis
physionet_root = os.path.dirname(project_root)  # f:\Sepsis\sepsis.1

sys.path.insert(0, project_root)

from src.dataset_grud import PhysioNetDatasetGRUD, collate_fn
from experiments.gat.model import GATBaseline

def set_seed(seed=42):
    import numpy as np
    import random
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def train_gat():
    print("=== Training GAT Baseline ===")

    # Load hyperparameters from transformer config (epochs, patience, lr, etc.)
    config_path = os.path.join(project_root, "experiments", "transformer", "config.json")
    with open(config_path, 'r') as f:
        config = json.load(f)

    set_seed(config.get('seed', 42))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load splits
    splits_path = os.path.join(project_root, "artifacts", "splits.json")
    with open(splits_path, 'r') as f:
        splits = json.load(f)

    # Preprocessing config owns: data schema, feature counts, pos_weight
    preprocess_cfg = os.path.join(project_root, "artifacts", "preprocessing_config.json")
    with open(preprocess_cfg, 'r') as f:
        prep_config = json.load(f)

    num_nodes   = len(prep_config["dynamic_features"])   # 35
    static_size = len(prep_config["static_features"])    # 5
    pos_weight  = prep_config["class_weight"]

    print(f"Dynamic features (num_nodes): {num_nodes}")
    print(f"Static features: {static_size}")
    print(f"Positive class weight: {pos_weight:.2f}")

    # Dataset paths with two-candidate fallback (same pattern as transformer/train.py)
    cand_dirs_1 = [
        os.path.join(physionet_root, "physionet2019", "training", "training_setA"),
        os.path.join(physionet_root, "physionet2019", "training", "training_setB"),
    ]
    cand_dirs_2 = [
        os.path.join(physionet_root, "training", "training_setA"),
        os.path.join(physionet_root, "training", "training_setB"),
    ]
    if os.path.exists(cand_dirs_1[0]):
        data_dirs = cand_dirs_1
    elif os.path.exists(cand_dirs_2[0]):
        data_dirs = cand_dirs_2
    else:
        raise FileNotFoundError("Could not locate training_setA and training_setB data directories.")

    # Load graph edges — path anchored to this script's directory
    graph_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "edges.csv")
    if not os.path.exists(graph_path):
        print("Graph not found! Please run graph_builder.py first.")
        return

    import pandas as pd
    edges_df = pd.read_csv(graph_path)
    sources = edges_df['source'].values
    targets = edges_df['target'].values
    weights = edges_df['weight'].values

    edge_index = torch.tensor([sources, targets], dtype=torch.long).to(device)
    edge_weight = torch.tensor(weights, dtype=torch.float).to(device)

    train_ids = splits['train']
    val_ids = splits['val']

    train_ds = PhysioNetDatasetGRUD(data_dirs, train_ids, preprocess_cfg, config['max_sequence_length'])
    val_ds = PhysioNetDatasetGRUD(data_dirs, val_ids, preprocess_cfg, config['max_sequence_length'])

    # Use smaller batch size for GAT due to high memory requirement (flattening batch*seq_len*nodes)
    batch_size = 8
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    model = GATBaseline(
        num_nodes=num_nodes,
        input_dim_per_node=3,         # value, mask, delta
        static_size=static_size,
        hidden_dim=32,
        out_dim=64,
        heads=2,
        dropout=config['dropout']
    ).to(device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight).to(device), reduction='none')
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])

    best_auprc = 0.0
    patience_counter = 0
    results_dir = os.path.join(project_root, "experiments", "results", "gat")
    os.makedirs(results_dir, exist_ok=True)

    train_losses = []
    val_losses = []

    for epoch in range(config['epochs']):
        model.train()
        train_loss = 0
        for values, mask, delta, static_features, labels, valid_mask in tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['epochs']}"):
            values, mask = values.to(device), mask.to(device)
            delta, static_features = delta.to(device), static_features.to(device)
            labels, valid_mask = labels.to(device), valid_mask.to(device)

            optimizer.zero_grad()
            logits = model(values, mask, delta, static_features, edge_index, edge_weight)
            loss_matrix = criterion(logits, labels)

            loss = (loss_matrix * valid_mask).sum() / valid_mask.sum().clamp(min=1.0)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Validation
        model.eval()
        val_loss = 0
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for values, mask, delta, static_features, labels, valid_mask in val_loader:
                values, mask = values.to(device), mask.to(device)
                delta, static_features = delta.to(device), static_features.to(device)
                labels, valid_mask = labels.to(device), valid_mask.to(device)

                logits = model(values, mask, delta, static_features, edge_index, edge_weight)
                loss_matrix = criterion(logits, labels)
                loss = (loss_matrix * valid_mask).sum() / valid_mask.sum().clamp(min=1.0)
                val_loss += loss.item()

                probs = torch.sigmoid(logits)
                valid_idx = valid_mask.bool()
                all_preds.extend(probs[valid_idx].cpu().numpy().tolist())
                all_labels.extend(labels[valid_idx].cpu().numpy().tolist())

        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        val_auprc = average_precision_score(all_labels, all_preds)
        val_auroc = roc_auc_score(all_labels, all_preds)

        print(f"Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val AUPRC: {val_auprc:.4f} | Val AUROC: {val_auroc:.4f}")

        if val_auprc > best_auprc:
            best_auprc = val_auprc
            torch.save(model.state_dict(), os.path.join(results_dir, "best_model.pt"))
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= config['patience']:
                print("Early stopping triggered!")
                break

    # Plot losses
    if HAS_MATPLOTLIB:
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(train_losses) + 1), train_losses, label='Train Loss')
        plt.plot(range(1, len(val_losses) + 1), val_losses, label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('GAT Training and Validation Loss')
        plt.legend()
        plot_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plots")
        os.makedirs(plot_dir, exist_ok=True)
        plt.savefig(os.path.join(plot_dir, "loss_curve.png"))
        plt.close()

if __name__ == "__main__":
    train_gat()
