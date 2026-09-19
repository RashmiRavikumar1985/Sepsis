import torch
import torch.nn as nn
import torch.nn.functional as F

class TemporalEncoder(nn.Module):
    """
    Abstract Base Class for all temporal model architectures in FedSepsis-KG.
    Enforces canonical interface: forward(values, masks, deltas, static_features, valid_mask)
    """
    def forward(self, values, masks, deltas, static_features=None, valid_mask=None):
        raise NotImplementedError


class TemporalDecay(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.W = nn.Parameter(torch.Tensor(input_size, input_size))
        self.b = nn.Parameter(torch.Tensor(input_size))
        nn.init.uniform_(self.W, -0.1, 0.1)
        nn.init.zeros_(self.b)

    def forward(self, d):
        # Decay factor: exp(-max(0, W*d + b))
        gamma = torch.exp(-torch.relu(torch.matmul(d, self.W) + self.b))
        return gamma

class RITS(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # Temporal decay for hidden state and input
        self.temp_decay_h = TemporalDecay(input_size)
        self.temp_decay_x = TemporalDecay(input_size)
        
        # RNN Cell (e.g., GRU)
        self.rnn_cell = nn.GRUCell(input_size * 2, hidden_size)
        
        # History-based estimator
        self.hist_reg = nn.Linear(hidden_size, input_size)
        
        # Feature-based estimator
        self.feat_reg = nn.Linear(input_size * 2, input_size)
        
        # Combine weights
        self.weight_combine = nn.Linear(input_size * 2, input_size)
        
    def forward(self, x, mask, delta):
        # x, mask, delta: [Batch, Seq, Features]
        batch_size, seq_len, num_features = x.size()
        
        h = torch.zeros(batch_size, self.hidden_size).to(x.device)
        c_prev = torch.zeros(batch_size, num_features).to(x.device)
        
        imputed_hist = []
        imputed_feat = []
        imputed_comb = []
        
        for t in range(seq_len):
            x_t = x[:, t, :]
            m_t = mask[:, t, :]
            d_t = delta[:, t, :]
            
            # 1. Decay
            gamma_h = self.temp_decay_h(d_t)
            gamma_x = self.temp_decay_x(d_t)
            
            # Decay hidden state (simplified approximation)
            h = h * gamma_h.mean(dim=1, keepdim=True)
            
            # History-based estimate
            x_hist = self.hist_reg(h)
            imputed_hist.append(x_hist.unsqueeze(1))
            
            # 2. Complement observation
            x_c = m_t * x_t + (1 - m_t) * c_prev
            
            # Feature-based estimate (using mask to indicate which are imputed)
            feat_in = torch.cat([x_c, m_t], dim=1)
            x_feat = self.feat_reg(feat_in)
            imputed_feat.append(x_feat.unsqueeze(1))
            
            # 3. Combine
            comb_in = torch.cat([x_hist, x_feat], dim=1)
            gamma_c = torch.sigmoid(self.weight_combine(comb_in))
            x_comb = gamma_c * x_hist + (1 - gamma_c) * x_feat
            imputed_comb.append(x_comb.unsqueeze(1))
            
            # 4. Final input for next step
            c_prev = m_t * x_t + (1 - m_t) * x_comb
            
            # 5. RNN step
            rnn_in = torch.cat([c_prev, m_t], dim=1)
            h = self.rnn_cell(rnn_in, h)
            
        imputed_hist = torch.cat(imputed_hist, dim=1)
        imputed_feat = torch.cat(imputed_feat, dim=1)
        imputed_comb = torch.cat(imputed_comb, dim=1)
        
        return imputed_hist, imputed_feat, imputed_comb

class SepsisClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, rnn_hidden_size):
        super().__init__()
        self.rits = RITS(input_size, rnn_hidden_size)
        
        # Classifier takes: imputed values, masks, time-gaps
        clf_input_size = input_size * 3
        self.classifier_rnn = nn.GRU(clf_input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
        
    def forward(self, x, mask, delta):
        x_hist, x_feat, x_comb = self.rits(x, mask, delta)
        
        # Replace missing with combined estimate for classifier
        x_filled = mask * x + (1 - mask) * x_comb
        
        clf_in = torch.cat([x_filled, mask, delta], dim=2)
        out, _ = self.classifier_rnn(clf_in)
        
        preds = torch.sigmoid(self.fc(out)).squeeze(-1)
        
        return preds, x_hist, x_feat, x_comb
