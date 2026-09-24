import os
import json
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset


class PhysioNetDatasetGRUD(Dataset):
    """
    PyTorch Dataset for GRU-D.
    
    For each patient, produces:
        - values:          (T, D)  Clipped and z-scored observed dynamic values (NaN -> 0 after standardization)
        - mask:            (T, D)  1 where observed, 0 where missing
        - delta:           (B, D)  hours since last observation per feature
        - static_features: (S,)    Z-scored or binary encoded static features
        - label:           (T,)    SepsisLabel at each hour
    """

    def __init__(self, data_dirs, patient_ids, config_path, max_seq_len=336):
        """
        Args:
            data_dirs: list of directories containing .psv files
            patient_ids: list of patient filenames for this split
            config_path: path to preprocessing_config.json
            max_seq_len: clip stays longer than this
        """
        self.max_seq_len = max_seq_len

        # Build file lookup: filename -> directory
        self.file_to_dir = {}
        for d in data_dirs:
            for f in os.listdir(d):
                if f.endswith('.psv'):
                    self.file_to_dir[f] = d

        # Keep only patient_ids that actually exist on disk
        self.files = [f for f in patient_ids if f in self.file_to_dir]

        # Load training statistics
        with open(config_path, 'r') as fp:
            config = json.load(fp)

        self.features = config['dynamic_features']
        self.num_features = len(self.features)

        means = config['train_means']
        stds = config['train_stds']
        self.means = np.array([means[f] for f in self.features], dtype=np.float32)
        self.stds = np.array([stds[f] for f in self.features], dtype=np.float32)

        # Clipping bounds
        self.clip_lo = np.array([config['clip_lo'][f] for f in self.features], dtype=np.float32)
        self.clip_hi = np.array([config['clip_hi'][f] for f in self.features], dtype=np.float32)

        # Static feature info
        self.static_features = config['static_features']
        self.static_encoding = config['static_encoding']
        self.static_stats = config['static_stats']

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fname = self.files[idx]
        path = os.path.join(self.file_to_dir[fname], fname)
        df = pd.read_csv(path, sep='|')

        seq_len = min(len(df), self.max_seq_len)
        raw = df[self.features].iloc[:seq_len].values.astype(np.float32)
        label = df['SepsisLabel'].iloc[:seq_len].values.astype(np.float32)

        # --- Mask (m_t) ---
        mask = (~np.isnan(raw)).astype(np.float32)

        # --- Apply Robust Clipping Bounds (from training data) ---
        # We clip only the observed (non-NaN) values
        clipped_raw = np.copy(raw)
        for d in range(self.num_features):
            obs_mask = mask[:, d] == 1
            if np.any(obs_mask):
                clipped_raw[obs_mask, d] = np.clip(raw[obs_mask, d], self.clip_lo[d], self.clip_hi[d])

        # --- Z-score using TRAINING means/stds ---
        values = (clipped_raw - self.means) / self.stds
        # Replace remaining NaNs with 0 (the population mean after standardization)
        values = np.nan_to_num(values, nan=0.0)

        # --- Time gap delta (hours since last observation, capped at 48.0 and normalized to [0, 1]) ---
        delta = np.zeros_like(raw, dtype=np.float32)
        for t in range(1, seq_len):
            for d in range(self.num_features):
                if mask[t - 1, d] == 1:
                    delta[t, d] = 1.0
                else:
                    delta[t, d] = delta[t - 1, d] + 1.0

        delta = np.minimum(delta, 48.0) / 48.0

        # --- Static Features Encoding ---
        static_vals = []
        for sf in self.static_features:
            val = df[sf].iloc[0] if sf in df.columns else np.nan
            encoding = self.static_encoding[sf]
            stats = self.static_stats[sf]
            
            if pd.isna(val):
                # Fallback to mean or default binary value
                if encoding == 'normalize':
                    processed_val = 0.0  # Mean is 0.0 after normalization
                else:
                    processed_val = 0.0
            else:
                if encoding == 'normalize':
                    processed_val = (val - stats['mean']) / stats['std']
                else:
                    processed_val = float(val)
            static_vals.append(processed_val)

        static_tensor = np.array(static_vals, dtype=np.float32)

        return {
            'values': torch.from_numpy(values),
            'mask': torch.from_numpy(mask),
            'delta': torch.from_numpy(delta),
            'static_features': torch.from_numpy(static_tensor),
            'label': torch.from_numpy(label),
            'seq_len': seq_len,
        }


def collate_fn(batch):
    """
    Pad variable-length patient stays to the longest stay in the batch.
    Returns a valid_time_mask so the loss ignores padded positions.
    """
    batch_size = len(batch)
    max_len = max(b['seq_len'] for b in batch)
    D = batch[0]['values'].shape[1]
    S = batch[0]['static_features'].shape[0]

    values = torch.zeros(batch_size, max_len, D)
    mask = torch.zeros(batch_size, max_len, D)
    delta = torch.zeros(batch_size, max_len, D)
    static_features = torch.zeros(batch_size, S)
    labels = torch.zeros(batch_size, max_len)
    valid_time_mask = torch.zeros(batch_size, max_len, dtype=torch.bool)

    for i, b in enumerate(batch):
        T = b['seq_len']
        values[i, :T] = b['values']
        mask[i, :T] = b['mask']
        delta[i, :T] = b['delta']
        static_features[i] = b['static_features']
        labels[i, :T] = b['label']
        valid_time_mask[i, :T] = True

    return values, mask, delta, static_features, labels, valid_time_mask
