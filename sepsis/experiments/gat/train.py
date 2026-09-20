import os
import sys
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import average_precision_score, roc_auc_score
from tqdm import tqdm
import matplotlib.pyplot as plt

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
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
    
    # Load config from transformer for hyperparams (or define custom)
    config_path = os.path.join(project_root, "experiments", "transformer", "config.json")
    with open(config_path, 'r') as f:
        config = json.load(f)
        
    set_seed(config.get('seed', 42))
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Load Splits
    splits_path = os.path.join(project_root, "artifacts", "splits.json")
    with open(splits_path, 'r') as f:
        splits = json.load(f)
        
    # Dataset Paths
    data_dirs = [
        os.path.join(project_root, "physionet2019", "training", "training_setA"),
        os.path.join(project_root, "physionet2019", "training", "training_setB")
    ]
    preprocess_cfg = os.path.join(project_root, "artifacts", "preprocessing_config.json")
    
    # Load class balance context for pos_weight
    pos_weight = config.get('pos_weight', None)
    if pos_weight is None:
        context_path = os.path.join(project_root, "artifacts", "class_balance_context.json")
        if not os.path.exists(context_path):
            context_path = os.path.join(project_root, "experiments", "results", "eda", "class_balance_context.json")
        if os.path.exists(context_path):
            with open(context_path, 'r') as f:
                ctx = json.load(f)
                pos_weight = ctx.get('pos_weight', 54.54)
        else:
            pos_weight = 54.54 # fallback
    print(f"Positive class weight: {pos_weight:.2f}")
            
    # Load Graph edges
    graph_path = os.path.join(project_root, "experiments", "gat", "edges.csv")
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
        num_nodes=config['input_dim'],
        input_dim_per_node=3, # value, mask, delta
        static_size=config['static_dim'],
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
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(train_losses) + 1), train_losses, label='Train Loss')
    plt.plot(range(1, len(val_losses) + 1), val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('GAT Training and Validation Loss')
    plt.legend()
    plot_dir = os.path.join(os.path.dirname(__file__), "plots")
    os.makedirs(plot_dir, exist_ok=True)
    plt.savefig(os.path.join(plot_dir, "loss_curve.png"))
    plt.close()
                
if __name__ == "__main__":
    train_gat()
