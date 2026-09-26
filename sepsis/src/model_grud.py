import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class CausalMultiHeadAttention(nn.Module):
    """
    Multi-head causal attention mechanism for GRU-D.
    
    Only attends to past timesteps to prevent data leakage in online prediction.
    Uses temperature scaling for better focus on critical time windows.
    """
    
    def __init__(self, hidden_size: int, num_heads: int = 8, dropout: float = 0.1, temperature: float = 1.0):
        super().__init__()
        assert hidden_size % num_heads == 0, "hidden_size must be divisible by num_heads"
        
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.temperature = temperature  # 1.0 = standard scaled dot-product attention
        
        self.q_linear = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_linear = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_linear = nn.Linear(hidden_size, hidden_size, bias=False)
        self.out_linear = nn.Linear(hidden_size, hidden_size)
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_size)
        
    def forward(self, hidden_states: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states: (B, T, H) - GRU-D hidden states at each timestep
            valid_mask: (B, T) - True for real timesteps, False for padding
            
        Returns:
            attended_states: (B, T, H) - attention-enhanced hidden states
        """
        B, T, H = hidden_states.shape
        
        # Multi-head projections
        Q = self.q_linear(hidden_states).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)  # (B, num_heads, T, head_dim)
        K = self.k_linear(hidden_states).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)  # (B, num_heads, T, head_dim)
        V = self.v_linear(hidden_states).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)  # (B, num_heads, T, head_dim)
        
        # Scaled dot-product attention with temperature
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (math.sqrt(self.head_dim) * self.temperature)  # (B, num_heads, T, T)
        
        # Causal masking - only attend to past and current timesteps
        causal_mask = torch.triu(torch.ones(T, T, device=scores.device, dtype=torch.bool), diagonal=1)
        scores.masked_fill_(causal_mask, float('-inf'))
        
        # Padding mask - don't attend to padded positions
        if valid_mask is not None:
            padding_mask = ~valid_mask.unsqueeze(1).unsqueeze(1)  # (B, 1, 1, T)
            scores.masked_fill_(padding_mask, float('-inf'))
        
        # Apply softmax and dropout
        # Guard: rows that are entirely -inf (fully masked) produce NaN after softmax.
        # Replace NaN with 0 so padded query positions contribute nothing.
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = torch.nan_to_num(attn_weights, nan=0.0)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        attended = torch.matmul(attn_weights, V)  # (B, num_heads, T, head_dim)
        attended = attended.transpose(1, 2).contiguous().view(B, T, H)  # (B, T, H)
        
        # Zero attended output at padded query positions before residual
        if valid_mask is not None:
            attended = attended.masked_fill(~valid_mask.unsqueeze(-1), 0.0)

        # Output projection and residual connection
        output = self.out_linear(attended)
        output = self.layer_norm(output + hidden_states)  # Residual connection
        
        return output


class GRUDCell(nn.Module):
    """
    Single GRU-D cell implementing learned input and hidden-state decay.

    Follows Che et al. 2018 "Recurrent Neural Networks for Multivariate Time
    Series with Missing Values" exactly.

    At each timestep t:
        gamma_x  = exp(-max(0, W_gamma_x @ delta_t + b_gamma_x))   input decay
        gamma_h  = exp(-max(0, W_gamma_h @ delta_t + b_gamma_h))   hidden decay

        x_hat_t  = m_t * x_t + (1 - m_t) * (gamma_x * x_last + (1 - gamma_x) * x_mean)
                 = m_t * x_t + (1 - m_t) * gamma_x * x_last
                   [x_mean = 0 after z-scoring, so the (1-gamma_x)*x_mean term vanishes]

        h_decayed = gamma_h * h_prev

        GRU update on concat(x_hat_t, m_t) with h_decayed as the prior hidden state
    """

    def __init__(self, input_size: int, hidden_size: int) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # Input decay: maps delta (B, D) -> gamma_x (B, D)
        self.W_gamma_x = nn.Linear(input_size, input_size, bias=True)
        # Hidden decay: maps delta (B, D) -> gamma_h (B, H)
        self.W_gamma_h = nn.Linear(input_size, hidden_size, bias=True)

        # GRU gates — input is concat(x_hat, mask) of size 2*D
        gru_input = input_size * 2
        self.linear_z = nn.Linear(gru_input + hidden_size, hidden_size)
        self.linear_r = nn.Linear(gru_input + hidden_size, hidden_size)
        self.linear_n = nn.Linear(gru_input + hidden_size, hidden_size)

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for name, param in self.named_parameters():
            if "weight" in name:
                nn.init.xavier_uniform_(param)
            elif "bias" in name:
                nn.init.zeros_(param)

    def forward(
        self,
        x_t: torch.Tensor,      # (B, D)  z-scored values, NaN replaced with 0
        m_t: torch.Tensor,      # (B, D)  observation mask: 1=observed, 0=missing
        delta_t: torch.Tensor,  # (B, D)  normalised time gap since last observation
        h_prev: torch.Tensor,   # (B, H)  previous hidden state
        x_last: torch.Tensor,   # (B, D)  last observed value per feature
    ):
        """
        Returns:
            h_new     (B, H)  updated hidden state
            x_last_new (B, D) updated last-observed values
        """
        # 1. Decay factors
        gamma_x = torch.exp(-torch.relu(self.W_gamma_x(delta_t)))  # (B, D)
        gamma_h = torch.exp(-torch.relu(self.W_gamma_h(delta_t)))  # (B, H)

        # 2. Imputed input
        #    x_mean = 0 after z-scoring, so full Che et al. formula reduces to:
        #    x_hat = m * x + (1-m) * gamma_x * x_last
        x_hat = m_t * x_t + (1.0 - m_t) * gamma_x * x_last        # (B, D)

        # 3. Decay hidden state
        h_decayed = gamma_h * h_prev                                # (B, H)

        # 4. GRU update
        combined = torch.cat([x_hat, m_t], dim=1)                  # (B, 2D)
        zh = torch.cat([combined, h_decayed], dim=1)               # (B, 2D+H)

        z = torch.sigmoid(self.linear_z(zh))
        r = torch.sigmoid(self.linear_r(zh))

        rh = torch.cat([combined, r * h_decayed], dim=1)           # (B, 2D+H)
        n = torch.tanh(self.linear_n(rh))

        h_new = (1.0 - z) * h_decayed + z * n                     # (B, H)

        # 5. Carry forward: use observed value where available
        x_last_new = m_t * x_t + (1.0 - m_t) * x_last

        return h_new, x_last_new


class GRUD(nn.Module):
    """
    Attention-Enhanced GRU-D model for hourly sepsis prediction.
    
    Combines GRU-D's temporal modeling with causal multi-head attention
    to focus on critical time windows for early sepsis detection.
    
    Architecture:
    1. GRU-D cells process sequential data with missing value handling
    2. Causal attention identifies important past timesteps
    3. Static feature fusion at each timestep
    4. Linear classifier produces per-hour logits
    """

    def __init__(
        self,
        input_size: int,
        static_size: int,
        hidden_size: int,
        dropout: float = 0.2,
        attention_config: dict = None,
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.static_size = static_size
        self.hidden_size = hidden_size
        self.use_attention = attention_config is not None and attention_config.get("enabled", False)

        self.cell = GRUDCell(input_size, hidden_size)
        self.dropout = nn.Dropout(dropout)

        # Causal attention mechanism
        if self.use_attention:
            self.attention = CausalMultiHeadAttention(
                hidden_size=hidden_size,
                num_heads=attention_config.get("num_heads", 8),
                dropout=attention_config.get("attention_dropout", 0.1),
                temperature=attention_config.get("temperature", 1.0)
            )
        
        # Feature fusion and classifier
        self.feature_fusion = nn.Sequential(
            nn.Linear(hidden_size + static_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.classifier = nn.Linear(hidden_size // 2, 1)

    def forward(
        self,
        values: torch.Tensor,           # (B, T, D)
        mask: torch.Tensor,             # (B, T, D)
        delta: torch.Tensor,            # (B, T, D)
        static_features: torch.Tensor,  # (B, S)
        valid_mask: torch.Tensor,       # (B, T)  True = real timestep
    ) -> torch.Tensor:
        """
        Args:
            values:          (B, T, D)  z-scored observations, NaN -> 0
            mask:            (B, T, D)  observation mask
            delta:           (B, T, D)  normalised time gaps
            static_features: (B, S)     static patient attributes
            valid_mask:      (B, T)     True for real hours, False for padding

        Returns:
            logits: (B, T)  raw logits; padded positions are set to 0.0
        """
        if valid_mask is None:
            raise ValueError("valid_mask is required — use collate_fn which always provides it")
        if valid_mask.dtype != torch.bool:
            valid_mask = valid_mask.bool()

        B, T, D = values.shape
        device = values.device

        h = torch.zeros(B, self.hidden_size, device=device)
        x_last = torch.zeros(B, D, device=device)
        
        # Collect all hidden states for attention
        hidden_states = []

        # Forward pass through GRU-D cells
        # Fix: skip GRU-D update for padded timesteps to avoid corrupting hidden state
        for t in range(T):
            x_t = values[:, t, :]          # (B, D)
            m_t = mask[:, t, :]            # (B, D)
            d_t = delta[:, t, :]           # (B, D)
            valid_t = valid_mask[:, t].unsqueeze(1)  # (B, 1)

            h_new, x_last_new = self.cell(x_t, m_t, d_t, h, x_last)

            # Only update state for real (non-padded) timesteps
            h      = torch.where(valid_t,                          h_new,      h)
            x_last = torch.where(valid_t.expand_as(x_last_new),   x_last_new, x_last)

            hidden_states.append(h)

        hidden_states = torch.stack(hidden_states, dim=1)  # (B, T, H)

        # Apply causal attention if enabled
        if self.use_attention:
            hidden_states = self.attention(hidden_states, valid_mask)

        # Feature fusion and classification at each timestep
        logits_list = []
        for t in range(T):
            h_t = hidden_states[:, t, :]  # (B, H)
            
            # Fuse with static features
            combined = torch.cat([h_t, static_features], dim=1)  # (B, H+S)
            fused = self.feature_fusion(combined)                # (B, H//2)
            logit = self.classifier(fused).squeeze(-1)           # (B,)
            logits_list.append(logit)

        logits = torch.stack(logits_list, dim=1)  # (B, T)

        # Zero out padded positions — safe for inference and evaluation
        logits = logits.masked_fill(~valid_mask, 0.0)
        return logits