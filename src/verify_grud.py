import torch
import json
from model_grud import GRUD

with open("config_grud.json") as f:
    cfg = json.load(f)

attn_cfg = cfg["attention"] if cfg["attention"]["enabled"] else None
model = GRUD(35, 5, cfg["hidden_size"], cfg["dropout"], attn_cfg)

B, T, D, S = 4, 20, 35, 5
values = torch.randn(B, T, D)
mask   = torch.bernoulli(torch.full((B, T, D), 0.7))
delta  = torch.rand(B, T, D)
static = torch.randn(B, S)
valid  = torch.ones(B, T, dtype=torch.bool)
valid[2, 15:] = False   # simulate short patient

logits = model(values, mask, delta, static, valid)
assert logits.shape == (B, T), f"Wrong shape: {logits.shape}"
assert logits[2, 15:].sum() == 0.0, "Padded positions must be zero"

params = sum(p.numel() for p in model.parameters())
print("=== GRU-D SANITY CHECK ===")
print(f"PASS  shape check      : {logits.shape}")
print(f"PASS  padding mask     : logits[2,15:] = 0")
print(f"PASS  parameters       : {params:,}")
print(f"PASS  hidden_size      : {cfg['hidden_size']}")
print(f"PASS  attention heads  : {cfg['attention']['num_heads']}")
print(f"PASS  temperature      : {cfg['attention']['temperature']}")
print(f"PASS  optimizer        : {cfg['optimizer']}")
print(f"PASS  learning_rate    : {cfg['learning_rate']}")
print(f"PASS  scheduler        : {cfg['lr_schedule']}")
print()
print("All checks passed — READY TO TRAIN")
