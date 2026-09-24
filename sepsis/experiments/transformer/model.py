import torch
import torch.nn as nn
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.models import TemporalEncoder

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

class TemporalTransformer(TemporalEncoder):
    def __init__(self, input_size, static_size, d_model=64, n_heads=4, num_layers=2, dim_feedforward=128, dropout=0.3):
        super(TemporalTransformer, self).__init__()
        self.input_size = input_size
        
        # We concatenate values(input_size), masks(input_size), and deltas(input_size)
        self.feature_proj = nn.Linear(input_size * 3, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=n_heads, 
            dim_feedforward=dim_feedforward, 
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Static feature projection
        self.static_proj = nn.Sequential(
            nn.Linear(static_size, d_model // 2),
            nn.GELU(),
            nn.LayerNorm(d_model // 2)
        )
        
        # Fusion of transformer output (d_model) with static projection (d_model // 2)
        fusion_dim = d_model + d_model // 2
        self.fusion = nn.Sequential(
            nn.LayerNorm(fusion_dim),
            nn.Linear(fusion_dim, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(d_model // 2)
        )
        
        # Classification head for sepsis prediction (outputting logits)
        self.classifier = nn.Linear(d_model // 2, 1)

    def forward(self, values, masks, deltas, static_features, valid_mask=None):
        """
        values: (batch, seq_len, D)
        masks: (batch, seq_len, D)
        deltas: (batch, seq_len, D)
        static_features: (batch, S)
        valid_mask: (batch, seq_len) boolean tensor indicating valid time steps
        """
        if valid_mask is None:
            raise ValueError("valid_mask is required for padded clinical sequences")
        if valid_mask.dtype != torch.bool:
            valid_mask = valid_mask.bool()

        batch_size, seq_len, _ = values.shape
        
        # 1. Feature concatenation
        x = torch.cat([values, masks, deltas], dim=-1)  # (batch, seq_len, 3 * D)
        x = self.feature_proj(x)                        # (batch, seq_len, d_model)
        
        # 2. Add Positional Encoding
        x = self.pos_encoder(x)                         # (batch, seq_len, d_model)
        
        # 3. Transformer Encoder with Causal Mask and Key Padding Mask
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device), diagonal=1)
        key_padding_mask = ~valid_mask
        
        memory = self.transformer_encoder(
            x, 
            mask=causal_mask, 
            src_key_padding_mask=key_padding_mask
        )  # (batch, seq_len, d_model)
        
        # 4. Static feature fusion
        static_h = self.static_proj(static_features)                            # (batch, d_model // 2)
        static_h = static_h.unsqueeze(1).expand(-1, seq_len, -1)                # (batch, seq_len, d_model // 2)
        fused = torch.cat([memory, static_h], dim=-1)                           # (batch, seq_len, d_model + d_model // 2)
        
        fused_rep = self.fusion(fused)                                          # (batch, seq_len, d_model // 2)
        
        # 5. Classification
        logits = self.classifier(fused_rep).squeeze(-1)                         # (batch, seq_len)
        logits = logits.masked_fill(~valid_mask, 0.0)
        return logits

