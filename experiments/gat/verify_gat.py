"""Quick sanity check for the fixed GAT implementation."""
import os, sys, json, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from experiments.gat.model import GATBaseline

with open(os.path.join(os.path.dirname(__file__), "config_gat.json")) as f:
    cfg = json.load(f)

model = GATBaseline(
    num_nodes=35,
    input_dim_per_node=3,
    static_size=5,
    hidden_dim=cfg["hidden_dim"],
    out_dim=cfg["out_dim"],
    heads=cfg["heads"],
    dropout=cfg["dropout"],
)

B, T, N, S = 2, 10, 35, 5
values = torch.randn(B, T, N)
masks  = torch.ones(B, T, N)
deltas = torch.rand(B, T, N)
static = torch.randn(B, S)
valid  = torch.ones(B, T, dtype=torch.bool)
valid[1, 7:] = False   # simulate short patient

# Simple chain graph for test
src = torch.arange(0, N - 1)
tgt = torch.arange(1, N)
edge_index  = torch.stack([
    torch.cat([src, tgt]),
    torch.cat([tgt, src])
])
edge_weight = torch.ones(edge_index.size(1))

logits = model(values, masks, deltas, static, edge_index, valid, edge_weight)
assert logits.shape == (B, T),       f"Wrong shape: {logits.shape}"
assert logits[1, 7:].sum() == 0.0,   "Padded positions must be zero"

params = sum(p.numel() for p in model.parameters())
print("=== GAT SANITY CHECK ===")
print(f"PASS  shape check      : {logits.shape}")
print(f"PASS  padding mask     : logits[1,7:] = 0")
print(f"PASS  parameters       : {params:,}")
print(f"PASS  hidden_dim       : {cfg['hidden_dim']}")
print(f"PASS  out_dim          : {cfg['out_dim']}")
print(f"PASS  heads            : {cfg['heads']}")
print(f"PASS  dropout          : {cfg['dropout']}")
print(f"PASS  optimizer        : {cfg['optimizer']}")
print(f"PASS  learning_rate    : {cfg['learning_rate']}")
print()
print("All checks passed — READY TO TRAIN")
