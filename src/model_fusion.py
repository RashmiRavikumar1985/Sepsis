"""
model_fusion.py — Phase 3: FedSepsis-KG GRU-D + GAT Fusion Model.

Exact architecture from the project specification diagram:

    ICU EHR
        │
    ┌───┴───────────────────┐
    │                       │
    ▼                       ▼
  GRU-D                Medical KG
  Temporal              Features + GAT
  Encoder               Encoder
    │                       │
    ▼                       ▼
  h_GRU (B,T,128)      h_GAT  (B,T,128)
    │                       │
    └──────────┬────────────┘
               ▼
    Attention-Gated Fusion
    ─────────────────────────
    Cross-attention: h_GRU queries h_GAT
    Cross-attention: h_GAT queries h_GRU
    Element-wise sigmoid gates (α·h_GRU + β·h_GAT)
    LayerNorm residual
               │
               ▼
       Prediction MLP
       ─────────────
       Linear(fused_dim → 128) → GELU → Dropout
       Linear(128 → 64)        → GELU → Dropout
       Linear(64  → 1)
               │
               ▼
         Sepsis Risk  (B, T)  per-hour logit

Design:
  • Both branches receive the SAME raw input (values, masks, deltas)
  • GRU-D captures temporal decay dynamics + missing data patterns
  • GAT captures spatial feature correlations via Graph F topology
  • Cross-attention fusion lets each branch selectively query the other
  • Sigmoid gates learn how much each branch contributes per timestep
  • MLP head with 3 layers (not a single linear) as shown in diagram
  • Pretrained encoder weights can be loaded from Phase 1/2 checkpoints
  • Padded timesteps are zeroed in the final logit output

References:
  WaveGNN (arXiv 2412.10621, 2024) — decay-aware GNN + temporal fusion
  PathSepsisNet (Springer 2026)    — gated spatial-temporal attention
"""

import os
import sys
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

_THIS_FILE  = os.path.abspath(__file__)
SEPSIS_ROOT = os.path.dirname(os.path.dirname(_THIS_FILE))
sys.path.insert(0, SEPSIS_ROOT)

from src.model_grud import GRUDCell, CausalMultiHeadAttention
from torch_geometric.nn import GATConv


# ══════════════════════════════════════════════════════════════════════
# Branch 1 — GRU-D Temporal Encoder
#   Same cell + causal attention as Phase 1 GRUD.
#   Returns h_GRU  (B, T, hidden_size)  — NO classifier head.
# ══════════════════════════════════════════════════════════════════════
class GRUDEncoder(nn.Module):
    """
    GRU-D temporal encoder branch.

    Processes:  values, masks, deltas  →  h_GRU (B, T, H)

    Internals identical to Phase 1 GRUD class but exposes the
    full hidden-state sequence instead of per-timestep logits.
    Causal multi-head attention is applied on top of the GRU-D
    hidden states to let the model focus on critical past windows.
    """

    def __init__(
        self,
        input_size:      int,
        hidden_size:     int,
        dropout:         float = 0.2,
        attention_config: dict = None,
    ) -> None:
        super().__init__()
        self.input_size   = input_size
        self.hidden_size  = hidden_size
        self.use_attention = (
            attention_config is not None
            and attention_config.get("enabled", False)
        )

        self.cell    = GRUDCell(input_size, hidden_size)
        self.dropout = nn.Dropout(dropout)

        if self.use_attention:
            self.attention = CausalMultiHeadAttention(
                hidden_size=hidden_size,
                num_heads=attention_config.get("num_heads", 8),
                dropout=attention_config.get("attention_dropout", 0.1),
                temperature=attention_config.get("temperature", 1.0),
            )

    def forward(
        self,
        values:     torch.Tensor,   # (B, T, D)
        mask:       torch.Tensor,   # (B, T, D)
        delta:      torch.Tensor,   # (B, T, D)
        valid_mask: torch.Tensor,   # (B, T) bool
    ) -> torch.Tensor:              # (B, T, H)
        if valid_mask.dtype != torch.bool:
            valid_mask = valid_mask.bool()

        B, T, D = values.shape
        device  = values.device

        h      = torch.zeros(B, self.hidden_size, device=device)
        x_last = torch.zeros(B, D, device=device)
        h_list = []

        for t in range(T):
            valid_t = valid_mask[:, t].unsqueeze(1)         # (B, 1)
            h_new, x_last_new = self.cell(
                values[:, t, :], mask[:, t, :], delta[:, t, :], h, x_last
            )
            # Only update hidden state on real (non-padded) timesteps
            h      = torch.where(valid_t, h_new, h)
            x_last = torch.where(valid_t.expand_as(x_last_new), x_last_new, x_last)
            h_list.append(h)

        h_states = torch.stack(h_list, dim=1)               # (B, T, H)

        if self.use_attention:
            h_states = self.attention(h_states, valid_mask)

        return h_states                                      # (B, T, H)


# ══════════════════════════════════════════════════════════════════════
# Branch 2 — Medical KG + GAT Structural Encoder
#   Same GATConv stack as Phase 2 GATBaseline.
#   Returns h_GAT  (B, T, out_dim)  — NO classifier head.
# ══════════════════════════════════════════════════════════════════════
class GATEncoder(nn.Module):
    """
    Medical knowledge-graph + GAT structural encoder branch.

    Each clinical feature = one graph node.
    Node features at time t: [value_t, mask_t, delta_t]  (3 dims).
    Two GATConv layers propagate information along Graph F edges.
    Mean pooling over nodes gives the graph-level embedding h_GAT.

    Processes:  values, masks, deltas + edge_index  →  h_GAT (B, T, out_dim)
    """

    def __init__(
        self,
        num_nodes:          int,
        input_dim_per_node: int   = 3,
        hidden_dim:         int   = 64,
        out_dim:            int   = 128,
        heads:              int   = 4,
        dropout:            float = 0.2,
    ) -> None:
        super().__init__()
        self.num_nodes = num_nodes

        # Project raw [value, mask, delta] → hidden_dim
        self.node_proj = nn.Sequential(
            nn.Linear(input_dim_per_node, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # GAT layer 1: hidden_dim → hidden_dim×heads  (concat)
        self.gat1  = GATConv(hidden_dim, hidden_dim, heads=heads,
                             concat=True,  dropout=dropout, add_self_loops=True)
        # GAT layer 2: hidden_dim×heads → out_dim      (mean)
        self.gat2  = GATConv(hidden_dim * heads, out_dim, heads=1,
                             concat=False, dropout=dropout, add_self_loops=True)

        self.norm1 = nn.LayerNorm(hidden_dim * heads)
        self.norm2 = nn.LayerNorm(out_dim)

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for name, param in self.named_parameters():
            if "weight" in name and param.dim() >= 2:
                nn.init.xavier_uniform_(param)
            elif "bias" in name:
                nn.init.zeros_(param)

    def forward(
        self,
        values:      torch.Tensor,          # (B, T, N)
        masks:       torch.Tensor,          # (B, T, N)
        deltas:      torch.Tensor,          # (B, T, N)
        edge_index:  torch.Tensor,          # (2, E)
        edge_weight: torch.Tensor = None,   # (E,)
    ) -> torch.Tensor:                      # (B, T, out_dim)
        B, T, N = values.shape
        device  = values.device

        # Stack node features: (B, T, N, 3) → flatten for PyG
        x      = torch.stack([values, masks, deltas], dim=-1)  # (B, T, N, 3)
        x      = x.view(B * T, N, 3)
        x      = self.node_proj(x)                             # (B*T, N, hidden)
        x_flat = x.view(B * T * N, -1)

        # Build batched edge index: shift by N per graph in the batch
        shifts     = torch.arange(B * T, device=device) * N
        batched_ei = (
            edge_index.unsqueeze(2) + shifts.view(1, 1, -1)
        ).permute(0, 2, 1).reshape(2, -1)

        bew = edge_weight.repeat(B * T) if edge_weight is not None else None

        # Two GATConv layers
        x1 = F.elu(self.norm1(self.gat1(x_flat, batched_ei, bew)))  # (B*T*N, H*heads)
        x2 = F.elu(self.norm2(self.gat2(x1,    batched_ei, bew)))   # (B*T*N, out_dim)

        # Mean pool over N nodes → graph-level embedding per timestep
        x2        = x2.view(B, T, N, -1)
        h_gat     = x2.mean(dim=2)                                   # (B, T, out_dim)
        return h_gat


# ══════════════════════════════════════════════════════════════════════
# Attention-Gated Fusion Module
#   Implements the "Attention-Gated Fusion" box in the diagram.
#
#   Step 1 — Cross-attention:
#     h_GRU  queries  h_GAT  → context_g  (GAT informs temporal)
#     h_GAT  queries  h_GRU  → context_h  (temporal informs GAT)
#
#   Step 2 — Sigmoid gates:
#     gate_h = σ(W_h · [h_GRU, h_GAT])
#     gate_g = σ(W_g · [h_GRU, h_GAT])
#     fused  = gate_h ⊙ h_GRU + gate_g ⊙ h_GAT
#            + gate_h ⊙ context_g + gate_g ⊙ context_h
#
#   Step 3 — LayerNorm residual → output dim = d_model
# ══════════════════════════════════════════════════════════════════════
class AttentionGatedFusion(nn.Module):
    """
    Cross-modal attention + sigmoid gating fusion.

    Implements the "Attention-Gated Fusion" stage from the architecture
    diagram. Both h_GRU and h_GAT must have the same dimension d_model.

    Args:
        d_model    : common embedding dim of both branches (e.g. 128)
        n_heads    : number of cross-attention heads
        dropout    : dropout on attention weights and projections
    """

    def __init__(self, d_model: int, n_heads: int = 4, dropout: float = 0.2) -> None:
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale    = math.sqrt(self.head_dim)

        # Projections for cross-attention: h_GRU → Q, h_GAT → K,V
        self.q_gru = nn.Linear(d_model, d_model, bias=False)  # GRU queries GAT
        self.k_gat = nn.Linear(d_model, d_model, bias=False)
        self.v_gat = nn.Linear(d_model, d_model, bias=False)

        # Projections for cross-attention: h_GAT → Q, h_GRU → K,V
        self.q_gat = nn.Linear(d_model, d_model, bias=False)  # GAT queries GRU
        self.k_gru = nn.Linear(d_model, d_model, bias=False)
        self.v_gru = nn.Linear(d_model, d_model, bias=False)

        self.attn_dropout = nn.Dropout(dropout)

        # Output projections after attention
        self.out_gru = nn.Linear(d_model, d_model)
        self.out_gat = nn.Linear(d_model, d_model)

        # Sigmoid gate: reads concat(h_GRU, h_GAT) → gate_h, gate_g ∈ (0,1)
        self.gate_h = nn.Linear(2 * d_model, d_model)
        self.gate_g = nn.Linear(2 * d_model, d_model)

        # Layer norms
        self.norm_h = nn.LayerNorm(d_model)
        self.norm_g = nn.LayerNorm(d_model)
        self.norm_out = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for name, param in self.named_parameters():
            if "weight" in name and param.dim() >= 2:
                nn.init.xavier_uniform_(param)
            elif "bias" in name:
                nn.init.zeros_(param)

    def _cross_attn(
        self,
        Q_proj: torch.Tensor,   # (B, T, d_model)
        K_proj: torch.Tensor,   # (B, T, d_model)
        V_proj: torch.Tensor,   # (B, T, d_model)
        valid_mask: torch.Tensor,  # (B, T)
    ) -> torch.Tensor:
        """
        Multi-head cross-attention — Q attends to K,V.
        Causal mask + padding mask applied.
        Returns (B, T, d_model).
        """
        B, T, _ = Q_proj.shape

        # Reshape to multi-head
        def reshape(x):
            return x.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
                                                       # (B, n_heads, T, head_dim)
        Q = reshape(Q_proj)
        K = reshape(K_proj)
        V = reshape(V_proj)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale  # (B, H, T, T)

        # Causal mask — only attend to current + past positions
        causal = torch.triu(
            torch.ones(T, T, device=scores.device, dtype=torch.bool), diagonal=1
        )
        scores = scores.masked_fill(causal, float("-inf"))

        # Padding mask
        if valid_mask is not None:
            pad_mask = ~valid_mask.unsqueeze(1).unsqueeze(1)        # (B, 1, 1, T)
            scores   = scores.masked_fill(pad_mask, float("-inf"))

        attn = F.softmax(scores, dim=-1)
        attn = torch.nan_to_num(attn, nan=0.0)
        attn = self.attn_dropout(attn)

        out  = torch.matmul(attn, V)                               # (B, H, T, head_dim)
        out  = out.transpose(1, 2).contiguous().view(B, T, self.d_model)
        return out

    def forward(
        self,
        h_gru:      torch.Tensor,   # (B, T, d_model)
        h_gat:      torch.Tensor,   # (B, T, d_model)
        valid_mask: torch.Tensor,   # (B, T) bool
    ) -> torch.Tensor:              # (B, T, d_model)
        """
        Returns fused representation (B, T, d_model).
        """
        # ── Step 1: Cross-attention ──────────────────────────────────
        # h_GRU queries h_GAT: temporal asks "what graph context is relevant?"
        ctx_g = self._cross_attn(
            self.q_gru(h_gru), self.k_gat(h_gat), self.v_gat(h_gat), valid_mask
        )
        ctx_g = self.out_gru(ctx_g)        # (B, T, d_model)

        # h_GAT queries h_GRU: graph asks "what temporal context is relevant?"
        ctx_h = self._cross_attn(
            self.q_gat(h_gat), self.k_gru(h_gru), self.v_gru(h_gru), valid_mask
        )
        ctx_h = self.out_gat(ctx_h)        # (B, T, d_model)

        # Residual update: each branch absorbs cross-modal context
        h_gru_aug = self.norm_h(h_gru + self.dropout(ctx_g))  # (B, T, d_model)
        h_gat_aug = self.norm_g(h_gat + self.dropout(ctx_h))  # (B, T, d_model)

        # ── Step 2: Sigmoid gating ───────────────────────────────────
        hg         = torch.cat([h_gru_aug, h_gat_aug], dim=-1) # (B, T, 2*d_model)
        alpha      = torch.sigmoid(self.gate_h(hg))             # (B, T, d_model)
        beta       = torch.sigmoid(self.gate_g(hg))             # (B, T, d_model)

        fused = alpha * h_gru_aug + beta * h_gat_aug             # (B, T, d_model)

        # ── Step 3: LayerNorm residual ───────────────────────────────
        fused = self.norm_out(fused)

        # Zero padded positions
        if valid_mask is not None:
            fused = fused.masked_fill(~valid_mask.unsqueeze(-1), 0.0)

        return fused                                             # (B, T, d_model)


# ══════════════════════════════════════════════════════════════════════
# Prediction MLP
#   Implements the "Prediction MLP" box in the diagram.
#   3-layer MLP: fused_dim → 128 → 64 → 1
# ══════════════════════════════════════════════════════════════════════
class PredictionMLP(nn.Module):
    """
    3-layer prediction head as shown in the architecture diagram.

    fused (B, T, d_model) → hidden1 → hidden2 → logit (B, T)

    Uses GELU activations, LayerNorm between layers, and dropout
    for regularisation under class imbalance.
    """

    def __init__(
        self,
        in_dim:   int,
        hidden1:  int   = 128,
        hidden2:  int   = 64,
        dropout:  float = 0.2,
    ) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden1),
            nn.LayerNorm(hidden1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden1, hidden2),
            nn.LayerNorm(hidden2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden2, 1),
        )
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for m in self.mlp:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, in_dim)  →  logits (B, T)"""
        return self.mlp(x).squeeze(-1)


# ══════════════════════════════════════════════════════════════════════
# FusionModel — Full Phase 3 Model
# ══════════════════════════════════════════════════════════════════════
class FusionModel(nn.Module):
    """
    Phase 3 — FedSepsis-KG GRU-D + GAT Fusion Model.

    Follows the exact architecture diagram:

        ICU EHR
            │
        ┌───┴─────────────────────┐
        ▼                         ▼
      GRU-D                  Medical KG + GAT
      Temporal                Structural
      Encoder                  Encoder
        │                         │
        ▼                         ▼
     h_GRU (B,T,128)         h_GAT (B,T,128)
        │                         │
        └──────────┬──────────────┘
                   ▼
       Attention-Gated Fusion
       (cross-attention + sigmoid gates)
                   │
                   ▼
           Prediction MLP
           (128 → 128 → 64 → 1)
                   │
                   ▼
             Sepsis Risk (B, T)

    Args:
        input_size    : 35 (dynamic features = graph nodes)
        static_size   : 5
        num_nodes     : 35 (= input_size; one node per feature)
        grud_hidden   : 128 (GRU-D hidden size)
        gat_hidden    : 64  (GAT internal dim)
        gat_out       : 128 (GAT output after pooling = must equal grud_hidden)
        gat_heads     : 4
        fusion_heads  : 4   (cross-attention heads in fusion)
        mlp_hidden1   : 128
        mlp_hidden2   : 64
        dropout       : 0.2
        grud_attn_cfg : GRU-D causal attention config dict
        static_inject : if True, inject static features into fusion input
        freeze_grud   : freeze GRU-D encoder (for fine-tuning)
        freeze_gat    : freeze GAT encoder (for fine-tuning)
    """

    def __init__(
        self,
        input_size:    int,
        static_size:   int,
        num_nodes:     int,
        grud_hidden:   int   = 128,
        gat_hidden:    int   = 64,
        gat_out:       int   = 128,
        gat_heads:     int   = 4,
        fusion_heads:  int   = 4,
        mlp_hidden1:   int   = 128,
        mlp_hidden2:   int   = 64,
        dropout:       float = 0.2,
        grud_attn_cfg: dict  = None,
        static_inject: bool  = True,
        freeze_grud:   bool  = False,
        freeze_gat:    bool  = False,
    ) -> None:
        super().__init__()

        assert gat_out == grud_hidden, (
            f"gat_out ({gat_out}) must equal grud_hidden ({grud_hidden}) "
            f"so both branches have the same dimension before fusion."
        )

        self.grud_hidden   = grud_hidden
        self.static_inject = static_inject
        d_model            = grud_hidden   # common dim = 128

        # ── Branch 1: GRU-D Temporal Encoder ────────────────────────
        self.grud_encoder = GRUDEncoder(
            input_size=input_size,
            hidden_size=grud_hidden,
            dropout=dropout,
            attention_config=grud_attn_cfg,
        )

        # ── Branch 2: Medical KG + GAT Encoder ──────────────────────
        self.gat_encoder = GATEncoder(
            num_nodes=num_nodes,
            input_dim_per_node=3,
            hidden_dim=gat_hidden,
            out_dim=gat_out,
            heads=gat_heads,
            dropout=dropout,
        )

        # ── Static feature injection (optional) ─────────────────────
        # Static features (age, gender, unit, HospAdmTime) are injected
        # into both branches BEFORE fusion via a small residual MLP.
        # This mirrors the per-branch static fusion in Phase 1/2 baselines.
        if static_inject:
            static_proj_dim = d_model  # project to same size as branches
            self.static_proj = nn.Sequential(
                nn.Linear(static_size, static_proj_dim),
                nn.LayerNorm(static_proj_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            # Blend static into h_GRU and h_GAT via additive residual
            self.static_blend_gru = nn.Sequential(
                nn.Linear(d_model + static_proj_dim, d_model),
                nn.LayerNorm(d_model),
                nn.GELU(),
            )
            self.static_blend_gat = nn.Sequential(
                nn.Linear(d_model + static_proj_dim, d_model),
                nn.LayerNorm(d_model),
                nn.GELU(),
            )

        # ── Attention-Gated Fusion ───────────────────────────────────
        self.fusion = AttentionGatedFusion(
            d_model=d_model,
            n_heads=fusion_heads,
            dropout=dropout,
        )

        # ── Prediction MLP ───────────────────────────────────────────
        self.pred_mlp = PredictionMLP(
            in_dim=d_model,
            hidden1=mlp_hidden1,
            hidden2=mlp_hidden2,
            dropout=dropout,
        )

        # ── Freeze encoders if requested ─────────────────────────────
        if freeze_grud:
            for p in self.grud_encoder.parameters():
                p.requires_grad = False

        if freeze_gat:
            for p in self.gat_encoder.parameters():
                p.requires_grad = False

    def forward(
        self,
        values:          torch.Tensor,   # (B, T, D)
        mask:            torch.Tensor,   # (B, T, D)
        delta:           torch.Tensor,   # (B, T, D)
        static_features: torch.Tensor,  # (B, S)
        edge_index:      torch.Tensor,  # (2, E)
        valid_mask:      torch.Tensor,  # (B, T) bool
        edge_weight:     torch.Tensor = None,  # (E,)
    ) -> torch.Tensor:                  # (B, T)
        """
        Returns logits (B, T) — padded positions are 0.0.
        """
        if valid_mask.dtype != torch.bool:
            valid_mask = valid_mask.bool()

        B, T, _ = values.shape

        # ── GRU-D temporal encoding ──────────────────────────────────
        h_gru = self.grud_encoder(values, mask, delta, valid_mask)   # (B, T, 128)

        # ── Medical KG + GAT structural encoding ─────────────────────
        h_gat = self.gat_encoder(values, mask, delta,
                                 edge_index, edge_weight)             # (B, T, 128)

        # ── Static feature injection ─────────────────────────────────
        if self.static_inject:
            s = self.static_proj(static_features)                    # (B, 128)
            s = s.unsqueeze(1).expand(-1, T, -1)                     # (B, T, 128)

            h_gru = self.static_blend_gru(
                torch.cat([h_gru, s], dim=-1)
            )                                                         # (B, T, 128)
            h_gat = self.static_blend_gat(
                torch.cat([h_gat, s], dim=-1)
            )                                                         # (B, T, 128)

        # ── Attention-Gated Fusion ───────────────────────────────────
        fused = self.fusion(h_gru, h_gat, valid_mask)                # (B, T, 128)

        # ── Prediction MLP ───────────────────────────────────────────
        logits = self.pred_mlp(fused)                                 # (B, T)

        # ── Mask padding positions ────────────────────────────────────
        logits = logits.masked_fill(~valid_mask, 0.0)

        return logits

    # ── Pretrained weight loaders ────────────────────────────────────

    def load_pretrained_grud(
        self, ckpt_path: str, device: torch.device = None
    ) -> None:
        """
        Load GRU-D encoder weights from a Phase 1 checkpoint.
        Keys mapped: cell.* and attention.* only (skips classifier head).
        """
        if device is None:
            device = next(self.parameters()).device
        ckpt  = torch.load(ckpt_path, map_location=device, weights_only=False)
        state = ckpt.get("model_state_dict", ckpt)

        enc_keys  = ("cell.", "attention.", "dropout.")
        grud_state = {k: v for k, v in state.items()
                      if any(k.startswith(pk) for pk in enc_keys)}

        missing, unexpected = self.grud_encoder.load_state_dict(
            grud_state, strict=False
        )
        print(f"[GRU-D ckpt] loaded {os.path.basename(ckpt_path)}")
        if missing:
            print(f"  missing keys   : {missing}")
        if unexpected:
            print(f"  unexpected keys: {unexpected}")

    def load_pretrained_gat(
        self, ckpt_path: str, device: torch.device = None
    ) -> None:
        """
        Load GAT encoder weights from a Phase 2 checkpoint.
        Keys mapped: node_proj.*, gat1.*, gat2.*, norm1.*, norm2.*
        (skips static_proj, fusion MLP, classifier head).
        """
        if device is None:
            device = next(self.parameters()).device
        ckpt  = torch.load(ckpt_path, map_location=device, weights_only=False)
        state = ckpt.get("model_state_dict", ckpt)

        enc_keys = ("node_proj.", "gat1.", "gat2.", "norm1.", "norm2.")
        gat_state = {k: v for k, v in state.items()
                     if any(k.startswith(pk) for pk in enc_keys)}

        missing, unexpected = self.gat_encoder.load_state_dict(
            gat_state, strict=False
        )
        print(f"[GAT ckpt] loaded {os.path.basename(ckpt_path)}")
        if missing:
            print(f"  missing keys   : {missing}")
        if unexpected:
            print(f"  unexpected keys: {unexpected}")
