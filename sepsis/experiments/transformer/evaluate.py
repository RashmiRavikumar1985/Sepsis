import os
import sys
import json
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
    precision_recall_curve
)

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except Exception:
    HAS_MATPLOTLIB = False

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from src.dataset_grud import PhysioNetDatasetGRUD, collate_fn
from experiments.transformer.model import TemporalTransformer

def evaluate_transformer():
    print("=== Evaluating Temporal Transformer ===")
    
    config_path = os.path.join(os.path.dirname(__file__), "config.json")
    with open(config_path, 'r') as f:
        config = json.load(f)
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load Splits
    splits_path = os.path.join(project_root, "artifacts", "splits.json")
    with open(splits_path, 'r') as f:
        splits = json.load(f)
        
    # Data directory paths - check actual structure
    cand_dirs_1 = [
        os.path.join(project_root, "physionet2019", "training", "training_setA"),
        os.path.join(project_root, "physionet2019", "training", "training_setB")
    ]
    cand_dirs_2 = [
        os.path.join(project_root, "training", "training_setA"),
        os.path.join(project_root, "training", "training_setB")
    ]
    # Check parent directory structure (physionet2019 at same level as sepsis)
    parent_root = os.path.dirname(project_root)
    cand_dirs_3 = [
        os.path.join(parent_root, "physionet2019", "training", "training_setA"),
        os.path.join(parent_root, "physionet2019", "training", "training_setB")
    ]
    
    if os.path.exists(cand_dirs_1[0]):
        data_dirs = cand_dirs_1
    elif os.path.exists(cand_dirs_2[0]):
        data_dirs = cand_dirs_2
    elif os.path.exists(cand_dirs_3[0]):
        data_dirs = cand_dirs_3
    else:
        raise FileNotFoundError(
            f"Could not locate training_setA and training_setB data directories.\n"
            f"Checked:\n"
            f"1. {cand_dirs_1[0]}\n"
            f"2. {cand_dirs_2[0]}\n"
            f"3. {cand_dirs_3[0]}"
        )

    preprocess_cfg = os.path.join(project_root, "artifacts", "preprocessing_config.json")
    with open(preprocess_cfg, 'r') as f:
        prep_config = json.load(f)

    input_size = len(prep_config['dynamic_features'])
    static_size = len(prep_config['static_features'])
    
    # CRITICAL: Verify this is the Phase-3 35-feature baseline
    assert input_size == 35, (
        f"Expected 35 dynamic features for Phase-3 baseline, got {input_size}"
    )
    assert static_size == 5, (
        f"Expected 5 static features, got {static_size}"
    )
    assert "ICULOS" in prep_config["dynamic_features"], (
        "ICULOS must be present in the 35-feature Phase-3 baseline"
    )
    assert input_size * 3 == 105, (
        f"Expected 105 Transformer input channels, got {input_size * 3}"
    )
    
    print(f"Dynamic features: {input_size}")
    print(f"Static features: {static_size}")
    print(f"Transformer input channels: {input_size * 3}")
    print("ICULOS: present")
    
    test_ids = splits['test']
    
    test_ds = PhysioNetDatasetGRUD(data_dirs, test_ids, preprocess_cfg, config['max_sequence_length'])
    test_loader = DataLoader(test_ds, batch_size=config['batch_size'], shuffle=False, collate_fn=collate_fn)
    
    model = TemporalTransformer(
        input_size=input_size,
        static_size=static_size,
        d_model=config['d_model'],
        n_heads=config['n_heads'],
        num_layers=config['num_layers'],
        dim_feedforward=config['dim_feedforward'],
        dropout=config['dropout']
    ).to(device)
    
    # CRITICAL: DO NOT evaluate untrained model
    results_dir = os.path.join(project_root, "experiments", "results", "transformer")
    artifacts_dir = os.path.join(project_root, "artifacts")
    model_path = os.path.join(artifacts_dir, "baseline_transformer.pt")
    if not os.path.exists(model_path):
        model_path = os.path.join(results_dir, "best_model.pt")
    
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device)
        
        # Verify checkpoint compatibility
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            checkpoint_input_size = checkpoint.get("input_size")
            checkpoint_static_size = checkpoint.get("static_size") 
            checkpoint_input_channels = checkpoint.get("input_channels")
            
            if checkpoint_input_size is not None:
                assert checkpoint_input_size == input_size, (
                    f"Checkpoint input_size={checkpoint_input_size}, "
                    f"current input_size={input_size}"
                )
            
            if checkpoint_static_size is not None:
                assert checkpoint_static_size == static_size, (
                    f"Checkpoint static_size={checkpoint_static_size}, "
                    f"current static_size={static_size}"
                )
            
            if checkpoint_input_channels is not None:
                assert checkpoint_input_channels == input_size * 3, (
                    f"Checkpoint input_channels={checkpoint_input_channels}, "
                    f"current={input_size * 3}"
                )
            
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)
        print(f"Loaded trained Transformer checkpoint from {model_path}")
    else:
        raise FileNotFoundError(
            f"Trained Transformer checkpoint not found. "
            f"Checked:\n{artifacts_dir}\n{results_dir}\n"
            f"Cannot evaluate untrained model - this would generate meaningless metrics."
        )
        
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for values, mask, delta, static_features, labels, valid_mask in test_loader:
            # CRITICAL: Verify valid_mask dtype
            assert valid_mask.dtype == torch.bool, (
                f"Expected valid_mask to be bool, got {valid_mask.dtype}"
            )
            
            values = values.to(device)
            mask = mask.to(device)
            delta = delta.to(device)
            static_features = static_features.to(device)
            labels = labels.to(device)
            valid_mask = valid_mask.to(device)
            
            logits = model(values, mask, delta, static_features, valid_mask=valid_mask)
            probs = torch.sigmoid(logits)
            valid_idx = valid_mask.bool()
            
            all_preds.extend(probs[valid_idx].cpu().numpy().tolist())
            all_labels.extend(labels[valid_idx].cpu().numpy().tolist())
            
    # Calculate Metrics
    auprc = average_precision_score(all_labels, all_preds)
    auroc = roc_auc_score(all_labels, all_preds)
    
    # Binarize predictions at 0.5 for secondary metrics
    bin_preds = [1 if p >= 0.5 else 0 for p in all_preds]
    precision = precision_score(all_labels, bin_preds, zero_division=0)
    recall = recall_score(all_labels, bin_preds, zero_division=0)
    f1 = f1_score(all_labels, bin_preds, zero_division=0)
    
    # Save comprehensive metrics with metadata
    metrics = {
        "model": "TemporalTransformer",
        "experiment": "Phase-3 35-feature baseline",
        "dynamic_features": input_size,
        "static_features": static_size,
        "input_channels": input_size * 3,
        "iculos_included": "ICULOS" in prep_config["dynamic_features"],
        "test_patients": len(test_ids),
        "test_valid_timesteps": len(all_labels),
        
        "Test AUPRC": auprc,
        "Test AUROC": auroc,
        "Test Precision @ 0.5": precision,
        "Test Recall @ 0.5": recall,
        "Test F1 @ 0.5": f1
    }
    
    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, "metrics.json"), 'w') as f:
        json.dump(metrics, f, indent=4)
        
    print(f"Test AUPRC: {auprc:.4f} | Test AUROC: {auroc:.4f}")
    print(f"Test Patients: {len(test_ids)} | Valid Timesteps: {len(all_labels)}")
    
    # Plots
    if HAS_MATPLOTLIB:
        plot_dir = os.path.join(os.path.dirname(__file__), "plots")
        os.makedirs(plot_dir, exist_ok=True)
        
        # PR Curve
        precision_vals, recall_vals, _ = precision_recall_curve(all_labels, all_preds)
        plt.figure(figsize=(8,6))
        plt.plot(recall_vals, precision_vals, label=f'AUPRC = {auprc:.4f}')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve (Transformer)')
        plt.legend()
        plt.savefig(os.path.join(plot_dir, "pr_curve.png"))
        plt.close()
        
        # ROC Curve
        fpr, tpr, _ = roc_curve(all_labels, all_preds)
        plt.figure(figsize=(8,6))
        plt.plot(fpr, tpr, label=f'AUROC = {auroc:.4f}')
        plt.plot([0, 1], [0, 1], linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve (Transformer)')
        plt.legend()
        plt.savefig(os.path.join(plot_dir, "roc_curve.png"))
        plt.close()
    
if __name__ == "__main__":
    evaluate_transformer()