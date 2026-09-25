"""
temporal_model.py  -  GAT-2: Temporal GAT (DENSE / DEVICE-AGNOSTIC)
====================================================================
Redesigned for speed on CPU, MPS, and CUDA.

Root cause of original slowness:
  Sparse scatter_reduce_ over T*35 nodes is poorly optimised on Apple MPS.

Solution — split into two fast dense operations:

  1. Clinical GAT (dense, per-timestep):
       Input: [B*T, 35, H]
       Use the pre-built 35x35 clinical adjacency as a dense attention mask.
       Standard multi-head attention over 35 nodes — pure matmul, no scatter.

  2. Temporal Mixing (causal, feature-wise):
       For each feature f at each timestep t:
           h[b,t,f] = h[b,t,f] + gate * h[b,t-1,f]   (past->current ONLY)
       Implemented as a masked shifted add — no scatter, no loops.

Architecture:
    [B,T,35]  values/masks/deltas
       ↓
    Node proj  Linear(3, H)  → [B,T,35,H]
       ↓
    Clinical GAT  (dense 35×35 per-timestep attention, L layers)
       → [B,T,35,H]
       ↓
    Temporal Mixing  (causal gated update, t-1→t)
       → [B,T,35,H]
       ↓
    Feature-attention pooling  [B,T,35,H] → [B,T,H]
       ↓
    Static fusion + Classifier  → [B,T]

Output: [B,T]  timestep-level logits
"""

import os, sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.models import TemporalEncoder

NUM_FEATURES = 35


# ─────────────────────────────────────────────────────────────────
#  Dense Clinical GAT Layer
#  Operates on [B*T, 35, H] — pure matmul, no scatter
# ─────────────────────────────────────────────────────────────────
class DenseClinicalGATLayer(nn.Module):
    """
    Dense multi-head GAT over 35 feature nodes.

    Uses a pre-computed 35×35 adjacency bias mask derived from the
    clinical correlation graph so that attention is zero for non-edges.

    All ops are dense matmul — fast on CPU, MPS, and CUDA.
    """

    def __init__(self, in_dim: int, out_dim: int, num_heads: int = 2,
                 dropout: float = 0.1, concat: bool = True):
        super().__init__()
        self.H = num_heads
        self.D = out_dim
        self.concat = concat

        self.W     = nn.Linear(in_dim, num_heads * out_dim, bias=False)
        self.a_src = nn.Linear(out_dim, 1, bias=False)   # attention query
        self.a_dst = nn.Linear(out_dim, 1, bias=False)   # attention key
        self.drop  = nn.Dropout(dropout)
        self.leaky = nn.LeakyReLU(0.2)

        # adj_bias: registered as buffer, updated once from edges.csv
        # Shape [1, H, 35, 35]: -inf for non-edges, 0 for edges
        self.register_buffer('adj_bias', torch.zeros(1, num_heads, NUM_FEATURES, NUM_FEATURES))

    def set_adjacency(self, sources: np.ndarray, targets: np.ndarray,
                      add_self_loops: bool = True):
        """Call once before training to bake in the clinical graph."""
        F = NUM_FEATURES
        bias = torch.full((1, self.H, F, F), float('-inf'))
        bias[:, :, sources, targets] = 0.0
        if add_self_loops:
            idx = torch.arange(F)
            bias[:, :, idx, idx] = 0.0
        self.adj_bias = bias  # [1, H, F, F]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [N, Fn, in_dim]   N = B*T (batched timesteps), Fn = 35 features
        returns: [N, Fn, H*D] if concat else [N, Fn, D]
        """
        N, Fn, _ = x.shape
        H, D = self.H, self.D

        # Project: [N, Fn, H*D] -> reshape [N, Fn, H, D]
        Wx = self.W(x).view(N, Fn, H, D)           # [N, Fn, H, D]

        Wx_h = Wx.permute(0, 2, 1, 3)              # [N, H, Fn, D]
        e_src = self.a_src(Wx_h)                    # [N, H, Fn, 1]
        e_dst = self.a_dst(Wx_h)                    # [N, H, Fn, 1]

        # e[i,j] = e_src[i] + e_dst[j]  -> [N, H, Fn, Fn]
        e = self.leaky(e_src + e_dst.permute(0, 1, 3, 2))  # [N, H, Fn, Fn]

        # Apply clinical adjacency mask
        e = e + self.adj_bias.to(x.device)          # broadcast [1,H,Fn,Fn]

        alpha = torch.softmax(e, dim=-1)             # [N, H, Fn, Fn]
        alpha = torch.nan_to_num(alpha, nan=0.0)
        alpha = self.drop(alpha)

        # Aggregate: [N, H, Fn, Fn] x [N, H, Fn, D] -> [N, H, Fn, D]
        out = torch.matmul(alpha, Wx_h)             # [N, H, Fn, D]

        if self.concat:
            out = out.permute(0, 2, 1, 3).reshape(N, Fn, H * D)  # [N, Fn, H*D]
        else:
            out = out.mean(dim=1)                    # [N, Fn, D]   avg heads

        return out


# ─────────────────────────────────────────────────────────────────
#  Temporal Mixing  (causal gated, no scatter)
# ─────────────────────────────────────────────────────────────────
class TemporalMixingLayer(nn.Module):
    """
    Causal temporal mixing: h[t] += gate * h[t-1].
    Implemented as a masked shifted addition — no loops, no scatter.
    PAST -> CURRENT only. No future leakage.

    Input:  [B, T, F, H]
    Output: [B, T, F, H]
    """

    def __init__(self, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        # Learned gate: how much to blend from previous timestep
        self.gate_proj = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid(),
        )
        self.drop = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        """
        x:          [B, T, F, H]
        valid_mask: [B, T]  bool
        returns:    [B, T, F, H]
        """
        B, T, F, H = x.shape

        # Shifted x: x_prev[t] = x[t-1], x_prev[0] = 0
        x_prev = torch.zeros_like(x)
        x_prev[:, 1:] = x[:, :-1]              # shift right (past->current)

        # Gate: [B, T, F, H] concat -> [B, T, F, H]
        gate_in = torch.cat([x, x_prev], dim=-1)    # [B, T, F, 2H]
        gate    = self.gate_proj(gate_in)            # [B, T, F, H]

        # Gated update
        x_new = x + gate * x_prev                   # [B, T, F, H]
        x_new = self.drop(x_new)
        x_new = self.norm(x_new)

        # Zero out padded timesteps
        vm = valid_mask[:, :, None, None].float()   # [B, T, 1, 1]
        x_new = x_new * vm

        return x_new


# ─────────────────────────────────────────────────────────────────
#  Full Temporal GAT model  (GAT-2, dense)
# ─────────────────────────────────────────────────────────────────
class TemporalGAT(TemporalEncoder):
    """
    GAT-2: Temporal GAT for sepsis prediction.
    Dense implementation — fast on CPU, MPS, and CUDA.

    call set_adjacency(sources, targets) ONCE before training to bake in
    the clinical correlation graph.

    forward() does NOT require edge_index at all — kept in signature
    for backward-compat with train/eval scripts (just ignored).
    """

    def __init__(
        self,
        num_features: int = 35,
        static_size: int = 5,
        hidden_dim: int = 64,
        out_dim: int = 64,
        num_heads: int = 2,
        num_gat_layers: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()
        assert num_features == 35 and static_size == 5

        self.F          = num_features
        self.hidden_dim = hidden_dim
        self.out_dim    = out_dim
        self.num_heads  = num_heads

        # ── Node projection ──
        self.node_proj = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

        # ── Clinical GAT layers ──
        gat_layers, gat_norms = [], []
        in_dim = hidden_dim
        for i in range(num_gat_layers):
            is_last = (i == num_gat_layers - 1)
            concat  = not is_last
            layer = DenseClinicalGATLayer(
                in_dim=in_dim, out_dim=out_dim,
                num_heads=num_heads, dropout=dropout, concat=concat,
            )
            gat_layers.append(layer)
            gat_norms.append(nn.LayerNorm(num_heads * out_dim if concat else out_dim))
            in_dim = num_heads * out_dim if concat else out_dim

        self.gat_layers  = nn.ModuleList(gat_layers)
        self.gat_norms   = nn.ModuleList(gat_norms)
        self.gat_drop    = nn.Dropout(dropout)
        self.gat_out_dim = in_dim  # dimension after last GAT layer

        # residual projection if dims differ
        self.res_proj = (nn.Linear(hidden_dim, self.gat_out_dim, bias=False)
                         if hidden_dim != self.gat_out_dim else nn.Identity())

        # ── Temporal mixing ──
        self.temporal_mix = TemporalMixingLayer(self.gat_out_dim, dropout)

        # ── Feature attention pooling: [35, H] -> [1, H] per timestep ──
        self.feat_attn = nn.Linear(self.gat_out_dim, 1, bias=True)

        # ── Static projection ──
        self.static_proj = nn.Sequential(
            nn.Linear(static_size, out_dim // 2),
            nn.GELU(),
            nn.LayerNorm(out_dim // 2),
        )

        # ── Fusion ──
        fusion_in = self.gat_out_dim + out_dim // 2
        self.fusion = nn.Sequential(
            nn.LayerNorm(fusion_in),
            nn.Linear(fusion_in, out_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(out_dim),
        )

        # ── Classifier ──
        self.classifier = nn.Linear(out_dim, 1)

    def set_adjacency(self, sources, targets, add_self_loops: bool = True):
        """Bake the clinical graph into each GAT layer. Call ONCE before training."""
        for layer in self.gat_layers:
            layer.set_adjacency(sources, targets, add_self_loops)

    def forward(
        self,
        values: torch.Tensor,           # [B, T, F]
        masks: torch.Tensor,            # [B, T, F]
        deltas: torch.Tensor,           # [B, T, F]
        static_features: torch.Tensor,  # [B, S]
        edge_index=None,                # ignored — kept for API compat
        valid_mask: Optional[torch.Tensor] = None,  # [B, T] bool
    ) -> torch.Tensor:
        B_, T_, Fn_ = values.shape
        assert Fn_ == self.F

        if valid_mask is None:
            valid_mask = torch.ones(B_, T_, dtype=torch.bool, device=values.device)
        elif valid_mask.dtype != torch.bool:
            valid_mask = valid_mask.bool()

        # ── 1. Node features [B, T, Fn, 3] -> [B, T, Fn, H] ──
        x = torch.stack([values, masks, deltas], dim=-1)       # [B, T, Fn, 3]
        x = self.node_proj(x)                                   # [B, T, Fn, H]

        # ── 2. Clinical GAT (per-timestep, dense) ──
        BT_ = B_ * T_
        x_bt = x.view(BT_, Fn_, self.hidden_dim)               # [B*T, Fn, H_in]
        x_res = self.res_proj(x_bt)                             # [B*T, Fn, H_out]

        for layer, norm in zip(self.gat_layers, self.gat_norms):
            x_new = layer(x_bt)                                 # [B*T, Fn, H_out]
            x_new = norm(x_new)
            if x_bt.size(-1) == x_new.size(-1):
                x_bt = F.gelu(x_new + x_bt)
            else:
                x_bt = F.gelu(x_new)
            x_bt = self.gat_drop(x_bt)

        # Residual from input projection
        x_bt = x_bt + x_res
        x = x_bt.view(B_, T_, Fn_, self.gat_out_dim)           # [B, T, Fn, H_out]

        # ── 3. Temporal mixing (causal gated, past->current only) ──
        x = self.temporal_mix(x, valid_mask)                    # [B, T, Fn, H_out]

        # ── 4. Feature-attention pooling ──
        attn = self.feat_attn(x).squeeze(-1)                   # [B, T, Fn]
        vm_f = valid_mask.unsqueeze(-1).expand_as(attn)        # [B, T, Fn]
        attn = attn.masked_fill(~vm_f, float('-inf'))
        attn = torch.softmax(attn, dim=-1)
        attn = torch.nan_to_num(attn, nan=0.0)
        ts_rep = (attn.unsqueeze(-1) * x).sum(dim=2)           # [B, T, H_out]

        # ── 5. Static fusion ──
        s_h = self.static_proj(static_features)                # [B, H//2]
        s_h = s_h.unsqueeze(1).expand(-1, T_, -1)             # [B, T, H//2]
        fused = self.fusion(torch.cat([ts_rep, s_h], dim=-1)) # [B, T, out_dim]

        # ── 6. Classify ──
        logits = self.classifier(fused).squeeze(-1)            # [B, T]
        logits = logits.masked_fill(~valid_mask, 0.0)
        return logits
