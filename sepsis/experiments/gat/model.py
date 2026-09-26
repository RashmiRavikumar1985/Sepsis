"""
model.py — GAT baseline model for FedSepsis-KG Phase 2.

Architecture:
  Per timestep, each clinical feature is a node in a graph.
  Node features at time t: [value, mask, delta] → 3 dims per node.
  Two GATConv layers refine node representations using the
  pre-built clinical correlation graph.
  Mean pooling over nodes produces a graph-level vector g_t.
  g_t is fused with static patient features and passed to a
  per-timestep classifier.

Output logits are masked to 0.0 at padded positions so that
downstream sigmoid / threshold operations are safe.
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from torch_geometric.nn import GATConv


class GATBaseline(nn.Module):
    """
    Graph Attention Network baseline for hourly sepsis prediction.

    Args:
        num_nodes:         number of dynamic features (graph nodes), e.g. 35
        input_dim_per_node: features per node = 3 (value, mask, delta)
        static_size:       number of static patient features
        hidden_dim:        hidden dimension inside GAT layers
        out_dim:           output dimension after GAT (graph-level representation)
        heads:             number of attention heads in first GATConv layer
        dropout:           dropout rate applied throughout
    """

    def __init__(
        self,
        num_nodes: int,
        input_dim_per_node: int,
        static_size: int,
        hidden_dim: int = 64,
        out_dim: int = 128,
        heads: int = 4,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.num_nodes = num_nodes
        self.dropout_rate = dropout

        # Project raw node features (value, mask, delta) → hidden_dim
        self.node_proj = nn.Sequential(
            nn.Linear(input_dim_per_node, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # GAT layer 1: hidden_dim → hidden_dim × heads (concat=True)
        self.gat1 = GATConv(
            hidden_dim,
            hidden_dim,
            heads=heads,
            concat=True,
            dropout=dropout,
            add_self_loops=True,
        )

        # GAT layer 2: hidden_dim × heads → out_dim (concat=False → mean)
        self.gat2 = GATConv(
            hidden_dim * heads,
            out_dim,
            heads=1,
            concat=False,
            dropout=dropout,
            add_self_loops=True,
        )

        # Layer norms for stable training
        self.norm1 = nn.LayerNorm(hidden_dim * heads)
        self.norm2 = nn.LayerNorm(out_dim)

        # Deep static fusion network
        self.static_proj = nn.Sequential(
            nn.Linear(static_size, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Feature fusion: graph rep + static → classifier
        self.fusion = nn.Sequential(
            nn.Linear(out_dim + hidden_dim, out_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(out_dim, out_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.classifier = nn.Linear(out_dim // 2, 1)

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for name, param in self.named_parameters():
            if "weight" in name and param.dim() >= 2:
                nn.init.xavier_uniform_(param)
            elif "bias" in name:
                nn.init.zeros_(param)

    def forward(
        self,
        values: torch.Tensor,           # (B, T, N)
        masks: torch.Tensor,            # (B, T, N)
        deltas: torch.Tensor,           # (B, T, N)
        static_features: torch.Tensor,  # (B, S)
        edge_index: torch.Tensor,       # (2, E)
        valid_mask: torch.Tensor,       # (B, T) bool — True = real timestep
        edge_weight: torch.Tensor = None,  # (E,) optional
    ) -> torch.Tensor:
        """
        Returns:
            logits: (B, T)  raw logits; padded positions are 0.0
        """
        if valid_mask is None:
            raise ValueError("valid_mask is required — use collate_fn which always provides it")
        if valid_mask.dtype != torch.bool:
            valid_mask = valid_mask.bool()

        B, T, N = values.shape
        device = values.device
        E = edge_index.size(1)

        # ── Build per-node features: stack [value, mask, delta] ──────
        # x: (B, T, N, 3)
        x = torch.stack([values, masks, deltas], dim=-1)

        # Flatten to (B*T, N, 3) then project nodes → (B*T, N, hidden_dim)
        x = x.view(B * T, N, 3)
        x = self.node_proj(x)                   # (B*T, N, hidden_dim)

        # Flatten to (B*T*N, hidden_dim) for PyG
        x_flat = x.view(B * T * N, -1)

        # ── Build batched edge index for B*T disconnected graphs ──────
        # Each of the B*T graphs has N nodes.
        # Shift edge indices by i*N for graph i.
        shifts = torch.arange(B * T, device=device) * N  # (B*T,)
        # batched_edge_index: (2, E * B*T)
        batched_edge_index = (
            edge_index.unsqueeze(2) + shifts.view(1, 1, -1)
        ).permute(0, 2, 1).reshape(2, -1)

        if edge_weight is not None:
            batched_edge_weight = edge_weight.repeat(B * T)
        else:
            batched_edge_weight = None

        # ── GAT layer 1 ───────────────────────────────────────────────
        x1 = self.gat1(x_flat, batched_edge_index, batched_edge_weight)
        x1 = self.norm1(x1)
        x1 = F.elu(x1)

        # ── GAT layer 2 ───────────────────────────────────────────────
        x2 = self.gat2(x1, batched_edge_index, batched_edge_weight)
        x2 = self.norm2(x2)
        x2 = F.elu(x2)

        # ── Reshape and mean-pool over nodes ──────────────────────────
        # x2: (B*T*N, out_dim) → (B, T, N, out_dim)
        x2 = x2.view(B, T, N, -1)
        graph_rep = x2.mean(dim=2)              # (B, T, out_dim)

        # ── Static feature projection ─────────────────────────────────
        static_h = self.static_proj(static_features)            # (B, hidden_dim)
        static_h = static_h.unsqueeze(1).expand(-1, T, -1)      # (B, T, hidden_dim)

        # ── Fusion and classification ─────────────────────────────────
        fused = torch.cat([graph_rep, static_h], dim=-1)        # (B, T, out_dim+hidden_dim)
        fused = self.fusion(fused)                               # (B, T, out_dim//2)
        logits = self.classifier(fused).squeeze(-1)              # (B, T)

        # ── Zero padded positions ────────────────────────────────────
        logits = logits.masked_fill(~valid_mask, 0.0)
        return logits
