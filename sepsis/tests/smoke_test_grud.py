"""Quick smoke test: load 1 batch through the full pipeline and verify shapes."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import torch
from src.dataset_grud import PhysioNetDatasetGRUD, collate_fn
from src.model_grud import GRUD
from torch.utils.data import DataLoader

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIRS = [
    os.path.join(PROJECT_ROOT, "training", "training_setA"),
    os.path.join(PROJECT_ROOT, "training", "training_setB"),
]
CONFIG_PATH = os.path.join(PROJECT_ROOT, "artifacts", "preprocessing_config.json")
SPLITS_PATH = os.path.join(PROJECT_ROOT, "artifacts", "splits.json")

with open(CONFIG_PATH) as f:
    config = json.load(f)
with open(SPLITS_PATH) as f:
    splits = json.load(f)

# Use just 20 patients for the smoke test
patient_ids = splits['train'][:20]
ds = PhysioNetDatasetGRUD(DATA_DIRS, patient_ids, CONFIG_PATH, max_seq_len=100)
loader = DataLoader(ds, batch_size=4, shuffle=False, collate_fn=collate_fn)

print(f"Dataset size: {len(ds)}")
values, mask, delta, static_features, labels, valid_mask = next(iter(loader))
print(f"Batch shapes:")
print(f"  values:          {values.shape}")
print(f"  mask:            {mask.shape}")
print(f"  delta:           {delta.shape}")
print(f"  static_features: {static_features.shape}")
print(f"  labels:          {labels.shape}")
print(f"  valid_mask:      {valid_mask.shape}")

D = values.shape[2]
S = static_features.shape[1]
model = GRUD(input_size=D, static_size=S, hidden_size=64, dropout=0.0)
logits = model(values, mask, delta, static_features)
print(f"  logits:          {logits.shape}")

# Quick gradient check
loss = (logits * valid_mask).sum()
loss.backward()
grad_norm = sum(p.grad.norm().item() for p in model.parameters() if p.grad is not None)
print(f"  grad_norm:       {grad_norm:.4f}")

print("\n=== Smoke test PASSED ===")
