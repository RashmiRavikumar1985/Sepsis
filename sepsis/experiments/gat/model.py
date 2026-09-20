import torch
import torch.nn as nn
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.models import TemporalEncoder
from torch_geometric.nn import GATConv

class GATBaseline(TemporalEncoder):
    def __init__(self, num_nodes, input_dim_per_node, static_size, hidden_dim=32, out_dim=64, heads=2, dropout=0.3):
        super(GATBaseline, self).__init__()
        self.num_nodes = num_nodes
        
        # input_dim_per_node = 3 (value, mask, delta)
        self.node_proj = nn.Linear(input_dim_per_node, hidden_dim)
        
        self.gat1 = GATConv(hidden_dim, hidden_dim, heads=heads, concat=True, dropout=dropout)
        self.gat2 = GATConv(hidden_dim * heads, out_dim, heads=1, concat=False, dropout=dropout)
        
        # The graph representation at time t will be the flattened nodes: num_nodes * out_dim
        # Alternatively, mean pooling over nodes. Let's use mean pooling for simplicity and robustness.
        self.fusion = nn.Sequential(
            nn.Linear(out_dim + static_size, out_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.classifier = nn.Linear(out_dim, 1)

    def forward(self, values, masks, deltas, static_features, edge_index, edge_weight=None):
        """
        values: (batch, seq_len, num_nodes)
        masks: (batch, seq_len, num_nodes)
        deltas: (batch, seq_len, num_nodes)
        static_features: (batch, S)
        edge_index: (2, num_edges)
        """
        batch_size, seq_len, num_nodes = values.shape
        
        # Combine node features: (batch, seq_len, num_nodes, 3)
        x = torch.stack([values, masks, deltas], dim=-1)
        
        # We need to process this over time. For a baseline, we'll process each (batch*seq_len) as a graph.
        x = x.view(batch_size * seq_len, num_nodes, 3)
        
        # Project node features
        x = self.node_proj(x) # (B*T, num_nodes, hidden_dim)
        
        # GAT expects (num_nodes, features), but PyG doesn't easily support batched dense graphs out of the box 
        # without DataLoaders unless we do loop or sparse batching.
        # Since the graph topology is identical for all samples, we can broadcast or loop.
        # For simplicity and given the small node count (35), we loop over B*T, or implement a batched GCN.
        # Since PyG GATConv doesn't take batched 3D inputs directly, let's convert to a large disconnected graph.
        
        # Actually, to make it fast without fully relying on PyG batching, let's just use an edge_index 
        # that connects all corresponding nodes across the batch but disjointly.
        # PyG provides `DataBatch` for this, but building it dynamically inside forward is slow.
        # Instead, we will iterate over batch elements if B*T is small, OR use a standard PyTorch multi-head attention.
        
        # fallback: standard attention over nodes if PyG fails
        # Let's try PyG approach:
        # Flatten x to (B*T*N, H)
        x_flat = x.view(batch_size * seq_len * num_nodes, -1)
        
        # Create a batched edge_index
        # edge_index is (2, E). We need to shift it by i*N for each graph i in B*T
        # This can be done via torch.arange
        device = x.device
        E = edge_index.size(1)
        shifts = torch.arange(batch_size * seq_len, device=device) * num_nodes
        shifted_edges = edge_index.unsqueeze(2) + shifts.view(1, 1, -1) # (2, E, B*T)
        batched_edge_index = shifted_edges.permute(0, 2, 1).reshape(2, -1) # (2, E*B*T)
        
        if edge_weight is not None:
            batched_edge_weight = edge_weight.repeat(batch_size * seq_len)
        else:
            batched_edge_weight = None

        x_gat = self.gat1(x_flat, batched_edge_index, batched_edge_weight)
        x_gat = torch.relu(x_gat)
        x_gat = self.gat2(x_gat, batched_edge_index, batched_edge_weight)
        
        # Reshape back to (batch, seq_len, num_nodes, out_dim)
        x_gat = x_gat.view(batch_size, seq_len, num_nodes, -1)
        
        # Pool over nodes (Mean Pooling)
        graph_rep = x_gat.mean(dim=2) # (batch, seq_len, out_dim)
        
        # Static feature fusion
        static_expanded = static_features.unsqueeze(1).expand(-1, seq_len, -1)
        fused = torch.cat([graph_rep, static_expanded], dim=-1)
        
        fused_rep = self.fusion(fused)
        
        # Classification
        logits = self.classifier(fused_rep).squeeze(-1)
        return logits
