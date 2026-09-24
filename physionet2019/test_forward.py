import torch
from src.models import SepsisClassifier

def test_forward():
    batch_size = 4
    seq_len = 10
    num_features = 39 # Based on dataset.py
    
    # Dummy inputs
    x = torch.randn(batch_size, seq_len, num_features)
    mask = torch.randint(0, 2, (batch_size, seq_len, num_features)).float()
    delta = torch.rand(batch_size, seq_len, num_features) * 5
    
    model = SepsisClassifier(input_size=num_features, hidden_size=64, rnn_hidden_size=64)
    
    preds, x_hist, x_feat, x_comb = model(x, mask, delta)
    
    print(f"Predictions shape: {preds.shape} (Expected: {batch_size}, {seq_len})")
    print(f"Imputed Hist shape: {x_hist.shape} (Expected: {batch_size}, {seq_len}, {num_features})")
    print("Forward pass successful!")

if __name__ == "__main__":
    test_forward()
