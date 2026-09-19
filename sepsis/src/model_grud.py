import torch
import torch.nn as nn
import math


class GRUDCell(nn.Module):
    """
    Single GRU-D cell implementing learned input & hidden-state decay.

    At each timestep t the cell receives:
        x_t     (B, D)   z-scored values (NaN already replaced with 0)
        m_t     (B, D)   observation mask
        delta_t (B, D)   hours since last observation
        h_prev  (B, H)   previous hidden state

    Equations (following Che et al. 2018):
        gamma_x = exp(-max(0, W_gamma_x @ delta_t + b_gamma_x))
        gamma_h = exp(-max(0, W_gamma_h @ delta_t + b_gamma_h))

        x_hat_t = m_t * x_t + (1 - m_t) * (gamma_x * x_last + (1 - gamma_x) * x_mean)
        h_decayed = gamma_h * h_prev

        GRU update on  concat(x_hat_t, m_t)  with  h_decayed
    """

    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # --- Decay parameters ---
        self.W_gamma_x = nn.Linear(input_size, input_size, bias=True)
        self.W_gamma_h = nn.Linear(input_size, hidden_size, bias=True)

        # --- GRU gates ---
        # Input is concat(x_hat, mask) -> 2*input_size
        gru_input = input_size * 2
        self.linear_z = nn.Linear(gru_input + hidden_size, hidden_size)
        self.linear_r = nn.Linear(gru_input + hidden_size, hidden_size)
        self.linear_n = nn.Linear(gru_input + hidden_size, hidden_size)

        self._reset_parameters()

    def _reset_parameters(self):
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)

    def forward(self, x_t, m_t, delta_t, h_prev, x_last):
        """
        Args:
            x_t:     (B, D)  current observed (z-scored, NaN->0)
            m_t:     (B, D)  mask
            delta_t: (B, D)  time gap
            h_prev:  (B, H)  previous hidden state
            x_last:  (B, D)  last observed value per feature (carried forward)

        Returns:
            h_new:   (B, H)
            x_last_new: (B, D)  updated last-observed values
        """
        # 1. Decay factors
        gamma_x = torch.exp(-torch.relu(self.W_gamma_x(delta_t)))   # (B, D)
        gamma_h = torch.exp(-torch.relu(self.W_gamma_h(delta_t)))   # (B, H)

        # 2. Imputed input  (x_mean is 0 after z-scoring)
        x_hat = m_t * x_t + (1 - m_t) * (gamma_x * x_last)         # (B, D)

        # 3. Decay hidden state
        h_decayed = gamma_h * h_prev                                 # (B, H)

        # 4. GRU update with concat(x_hat, mask)
        combined = torch.cat([x_hat, m_t], dim=1)                    # (B, 2D)
        zh = torch.cat([combined, h_decayed], dim=1)                 # (B, 2D+H)

        z = torch.sigmoid(self.linear_z(zh))
        r = torch.sigmoid(self.linear_r(zh))

        rh = torch.cat([combined, r * h_decayed], dim=1)
        n = torch.tanh(self.linear_n(rh))

        h_new = (1 - z) * h_decayed + z * n                         # (B, H)

        # 5. Update x_last: where observed, use current; else keep old
        x_last_new = m_t * x_t + (1 - m_t) * x_last

        return h_new, x_last_new


class GRUD(nn.Module):
    """
    Full GRU-D model for hourly sepsis prediction.

    Unrolls a GRUDCell over the time dimension, fuses static patient
    features, and applies a linear head to produce per-hour logits.
    """

    def __init__(self, input_size, static_size, hidden_size, dropout=0.3):
        super().__init__()
        self.input_size = input_size
        self.static_size = static_size
        self.hidden_size = hidden_size

        self.cell = GRUDCell(input_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        
        # Classifier maps fused hidden representation (H + S) to a single logit
        self.fc = nn.Linear(hidden_size + static_size, 1)

    def forward(self, values, mask, delta, static_features):
        """
        Args:
            values:          (B, T, D)  z-scored observations
            mask:            (B, T, D)  observation mask
            delta:           (B, T, D)  time gaps
            static_features: (B, S)     static patient attributes

        Returns:
            logits: (B, T)  raw logits (apply sigmoid externally)
        """
        B, T, D = values.shape
        device = values.device

        h = torch.zeros(B, self.hidden_size, device=device)
        x_last = torch.zeros(B, D, device=device)

        logits_list = []

        for t in range(T):
            x_t = values[:, t, :]
            m_t = mask[:, t, :]
            d_t = delta[:, t, :]

            h, x_last = self.cell(x_t, m_t, d_t, h, x_last)
            
            # Fuse recurrent representation and static features at each timestep
            h_fused = torch.cat([h, static_features], dim=1)  # (B, H + S)
            
            logit = self.fc(self.dropout(h_fused)).squeeze(-1)   # (B,)
            logits_list.append(logit)

        logits = torch.stack(logits_list, dim=1)            # (B, T)
        return logits
