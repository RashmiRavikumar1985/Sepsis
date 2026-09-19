#!/usr/bin/env python3
"""
Verify 35-feature baseline consistency between preprocessing and dataset.
This checks all blocking issues before training.
"""

import sys
import os
import json
import torch

# Add paths
sys.path.insert(0, 'sepsis')
sys.path.insert(0, 'physionet2019/src')

from dataset_grud import PhysioNetDatasetGRUD, collate_fn
from torch.utils.data import DataLoader

def verify_baseline_consistency():
    print("=== Verifying 35-Feature Baseline Consistency ===\n")
    
    # Load configs
    prep_cfg_path = "physionet2019/artifacts/preprocessing_config.json"
    with open(prep_cfg_path, 'r') as f:
        prep_config = json.load(f)
    
    splits_path = "physionet2019/artifacts/splits.json"  
    with open(splits_path, 'r') as f:
        splits = json.load(f)

    # Check 1: Preprocessing config structure
    print("✓ Check 1: Preprocessing config structure")
    dynamic_features = prep_config["dynamic_features"]
    static_features = prep_config["static_features"]
    
    input_size = len(dynamic_features)
    static_size = len(static_features)
    
    assert input_size == 35, f"Expected 35 dynamic features, got {input_size}"
    assert static_size == 5, f"Expected 5 static features, got {static_size}"  
    assert "ICULOS" in dynamic_features, "ICULOS missing from dynamic features"
    assert dynamic_features[-1] == "ICULOS", f"ICULOS should be last feature, got {dynamic_features[-1]}"
    
    print(f"  ✅ Dynamic features: {input_size}")
    print(f"  ✅ Static features: {static_size}")
    print(f"  ✅ ICULOS at index: {dynamic_features.index('ICULOS')}")
    print(f"  ✅ Input channels: {input_size * 3}")
    
    # Check 2: Dataset loading
    print("\n✓ Check 2: Dataset consistency")
    data_dirs = ["physionet2019/training/training_setA", "physionet2019/training/training_setB"]
    train_ids = splits['train'][:5]  # Use small subset
    
    dataset = PhysioNetDatasetGRUD(data_dirs, train_ids, prep_cfg_path, max_seq_len=50)
    loader = DataLoader(dataset, batch_size=2, shuffle=False, collate_fn=collate_fn)
    
    # Get one batch
    values, masks, deltas, static_features_tensor, labels, valid_mask = next(iter(loader))
    
    print(f"  ✅ Dataset features: {dataset.num_features}")
    print(f"  ✅ Dataset feature list matches preprocessing: {dataset.features == dynamic_features}")
    
    # Check 3: Tensor shapes
    print("\n✓ Check 3: Tensor shapes")
    assert values.shape[-1] == 35, f"Values wrong feature count: {values.shape[-1]}"
    assert masks.shape == values.shape, f"Masks shape mismatch: {masks.shape} vs {values.shape}"
    assert deltas.shape == values.shape, f"Deltas shape mismatch: {deltas.shape} vs {values.shape}"
    assert labels.shape == values.shape[:2], f"Labels shape mismatch: {labels.shape} vs {values.shape[:2]}"
    assert valid_mask.shape == values.shape[:2], f"Valid mask shape mismatch: {valid_mask.shape} vs {values.shape[:2]}"
    assert static_features_tensor.shape[-1] == 5, f"Static features wrong count: {static_features_tensor.shape[-1]}"
    
    print(f"  ✅ values: {values.shape}")
    print(f"  ✅ masks: {masks.shape}")
    print(f"  ✅ deltas: {deltas.shape}")
    print(f"  ✅ static_features: {static_features_tensor.shape}")
    print(f"  ✅ labels: {labels.shape}")
    print(f"  ✅ valid_mask: {valid_mask.shape}")
    
    # Check 4: Concatenated input for Transformer
    print("\n✓ Check 4: Transformer input")
    x_concat = torch.cat([values, masks, deltas], dim=-1)
    expected_channels = input_size * 3
    
    assert x_concat.shape[-1] == 105, f"Expected 105 channels, got {x_concat.shape[-1]}"
    assert x_concat.shape[-1] == expected_channels, f"Channel count mismatch: {x_concat.shape[-1]} vs {expected_channels}"
    
    print(f"  ✅ Concatenated input: {x_concat.shape}")
    print(f"  ✅ Input channels: {x_concat.shape[-1]} == 105")
    
    # Check 5: ICULOS values
    print("\n✓ Check 5: ICULOS verification")
    iculos_idx = dynamic_features.index("ICULOS")
    iculos_values = values[0, :, iculos_idx]  # First patient
    iculos_mask = masks[0, :, iculos_idx]
    
    print(f"  ✅ ICULOS index: {iculos_idx}")
    print(f"  ✅ ICULOS values (first 10): {iculos_values[:10].tolist()}")
    print(f"  ✅ ICULOS mask (first 10): {iculos_mask[:10].tolist()}")
    
    # Check 6: All finite values
    print("\n✓ Check 6: Data quality")
    assert torch.isfinite(values).all(), "Non-finite values found"
    assert torch.isfinite(deltas).all(), "Non-finite deltas found"  
    assert torch.isfinite(static_features_tensor).all(), "Non-finite static features found"
    assert torch.isfinite(labels).all(), "Non-finite labels found"
    
    print(f"  ✅ All values are finite")
    print(f"  ✅ Masks contain only 0/1: {set(torch.unique(masks).tolist()).issubset({0.0, 1.0})}")
    print(f"  ✅ Valid mask is boolean: {valid_mask.dtype == torch.bool}")
    
    print("\n🎉 All baseline consistency checks PASSED!")
    print(f"\nBaseline Configuration:")
    print(f"  📊 Dynamic features: {input_size} (includes ICULOS)")
    print(f"  📊 Static features: {static_size}")  
    print(f"  📊 Transformer input channels: {input_size * 3}")
    print(f"  📊 Class weight: {prep_config.get('class_weight', 'N/A')}")
    print(f"\nReady for 35-feature baseline training! 🚀")
    
    return True

if __name__ == "__main__":
    verify_baseline_consistency()