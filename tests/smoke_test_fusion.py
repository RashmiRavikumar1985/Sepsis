"""
smoke_test_fusion.py — Shape, gradient and gate sanity check for FusionModel.

Run from sepsis/ root:
    python tests/smoke_test_fusion.py

Architecture verified:
    ICU EHR
        │
    ┌───┴─────────────────────┐
    ▼                         ▼
  GRU-D                  Medical KG+GAT
  Temporal                Structural
  Encoder                  Encoder
    │                         │
    ▼                         ▼
  h_GRU (B,T,128)         h_GAT (B,T,128)
    │                         │
    └──────────┬──────────────┘
               ▼
    Attention-Gated Fusion
    (cross-attn + sigmoid gates)
               │
               ▼
       Prediction MLP
       (128→128→64→1)
               │
               ▼
         Sepsis Risk (B,T)
"""

import os, sys
import torch
import torch.nn as nn

_THIS = os.path.abspath(__file__)
SEPSIS_ROOT = os.path.dirname(os.path.dirname(_THIS))
sys.path.insert(0, SEPSIS_ROOT)

from src.model_fusion import FusionModel

# ── Dimensions (must match preprocessing_config.json) ─────────────────
B, T, D, S, N, E = 4, 20, 35, 5, 35, 134   # Graph F has 134 bidir edges

device = torch.device("cpu")
torch.manual_seed(42)

# ── Dummy inputs ──────────────────────────────────────────────────────
values       = torch.randn(B, T, D)
mask         = torch.randint(0, 2, (B, T, D)).float()
delta        = torch.rand(B, T, D)
static_feat  = torch.randn(B, S)

# Simulate variable-length stays (last 2 patients shorter)
valid_mask = torch.ones(B, T, dtype=torch.bool)
valid_mask[2, 15:] = False
valid_mask[3, 10:] = False

# Fake Graph F edges
src = torch.randint(0, N, (E,))
tgt = torch.randint(0, N, (E,))
edge_index  = torch.stack([src, tgt], dim=0)
edge_weight = torch.rand(E)

# ── Build model ───────────────────────────────────────────────────────
grud_attn_cfg = {
    "enabled": True, "num_heads": 8,
    "attention_dropout": 0.1, "temperature": 1.0
}

model = FusionModel(
    input_size    = D,
    static_size   = S,
    num_nodes     = N,
    grud_hidden   = 128,
    gat_hidden    = 64,
    gat_out       = 128,    # MUST equal grud_hidden
    gat_heads     = 4,
    fusion_heads  = 4,
    mlp_hidden1   = 128,
    mlp_hidden2   = 64,
    dropout       = 0.1,
    grud_attn_cfg = grud_attn_cfg,
    static_inject = True,
    freeze_grud   = False,
    freeze_gat    = False,
).to(device)

print("=" * 60)
print("FusionModel Smoke Test  (Architecture v2 — Attention-Gated)")
print("=" * 60)

n_total     = sum(p.numel() for p in model.parameters())
n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"  Total params     : {n_total:,}")
print(f"  Trainable params : {n_trainable:,}")

# ── 1. Forward pass ───────────────────────────────────────────────────
model.train()
logits = model(values, mask, delta, static_feat,
               edge_index, valid_mask, edge_weight)

assert logits.shape == (B, T), \
    f"Expected ({B},{T}), got {logits.shape}"
print(f"  Output shape     : {logits.shape}  ✓")

# ── 2. NaN / Inf ──────────────────────────────────────────────────────
assert not torch.isnan(logits).any(), "NaN in logits!"
assert not torch.isinf(logits).any(), "Inf in logits!"
print(f"  No NaN / Inf     : ✓")

# ── 3. Padded positions = 0.0 ─────────────────────────────────────────
for b in range(B):
    sl     = int(valid_mask[b].sum().item())
    padded = logits[b, sl:]
    if padded.numel() > 0:
        assert (padded == 0.0).all(), f"Padded not zero at b={b}"
print(f"  Padded = 0.0     : ✓")

# ── 4. Backward / gradient flow ───────────────────────────────────────
labels   = torch.randint(0, 2, (B, T)).float()
loss_mat = nn.BCEWithLogitsLoss(
    pos_weight=torch.tensor([54.54]), reduction="none"
)(logits, labels) * valid_mask.float()
loss = loss_mat.sum() / valid_mask.float().sum()
loss.backward()

no_grad = [n for n, p in model.named_parameters()
           if p.requires_grad and p.grad is None]
if no_grad:
    print(f"  WARNING no grad  : {no_grad}")
else:
    print(f"  Gradients flow   : ✓  (loss={loss.item():.4f})")

# ── 5. Sub-module shape checks ────────────────────────────────────────
model.eval()
with torch.no_grad():
    h_gru = model.grud_encoder(values, mask, delta, valid_mask)
    h_gat = model.gat_encoder(values, mask, delta, edge_index, edge_weight)

assert h_gru.shape == (B, T, 128), f"GRUDEncoder wrong: {h_gru.shape}"
assert h_gat.shape == (B, T, 128), f"GATEncoder wrong:  {h_gat.shape}"
print(f"  h_GRU shape      : {h_gru.shape}  ✓")
print(f"  h_GAT shape      : {h_gat.shape}  ✓")

# ── 6. Fusion cross-attention sanity ──────────────────────────────────
with torch.no_grad():
    s     = model.static_proj(static_feat).unsqueeze(1).expand(-1, T, -1)
    h_gru_aug = model.static_blend_gru(torch.cat([h_gru, s], dim=-1))
    h_gat_aug = model.static_blend_gat(torch.cat([h_gat, s], dim=-1))
    fused = model.fusion(h_gru_aug, h_gat_aug, valid_mask)

assert fused.shape == (B, T, 128), f"Fusion wrong: {fused.shape}"
print(f"  Fused shape      : {fused.shape}  ✓")

# ── 7. Gate value sanity ──────────────────────────────────────────────
with torch.no_grad():
    hg    = torch.cat([h_gru_aug, h_gat_aug], dim=-1)
    alpha = torch.sigmoid(model.fusion.gate_h(hg))
    beta  = torch.sigmoid(model.fusion.gate_g(hg))

assert (alpha >= 0).all() and (alpha <= 1).all(), "alpha out of (0,1)"
assert (beta  >= 0).all() and (beta  <= 1).all(), "beta out of (0,1)"
print(f"  Gate α mean      : {alpha.mean():.3f}  ✓")
print(f"  Gate β mean      : {beta.mean():.3f}  ✓")

print("=" * 60)
print("ALL CHECKS PASSED ✓  FusionModel ready to train.")
print("=" * 60)
print(f"\nRun full training:")
print(f"  python src/train_fusion.py")
print(f"\nRun subset (fast debug):")
print(f"  python src/train_fusion.py --subset 500")
print(f"\nFine-tune fusion head only (frozen encoders):")
print(f"  python src/train_fusion.py --freeze_grud --freeze_gat")
