import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset

class PhysioNetDataset(Dataset):
    def __init__(self, data_dir, patient_ids=None, max_seq_len=336):
        """
        Args:
            data_dir: Path to directory containing .psv files.
            patient_ids: List of filenames to include (if None, includes all).
            max_seq_len: Maximum length of ICU stay in hours (for padding).
        """
        self.data_dir = data_dir
        if patient_ids is None:
            self.files = [f for f in os.listdir(data_dir) if f.endswith('.psv')]
        else:
            self.files = patient_ids
            
        self.max_seq_len = max_seq_len
        
        # Define features. Omitting EtCO2 since it's 100% missing as analyzed earlier.
        self.features = [
            'HR', 'O2Sat', 'Temp', 'SBP', 'MAP', 'DBP', 'Resp', 'BaseExcess',
            'HCO3', 'FiO2', 'pH', 'PaCO2', 'SaO2', 'AST', 'BUN', 'Alkalinephos',
            'Calcium', 'Chloride', 'Creatinine', 'Bilirubin_direct', 'Glucose',
            'Lactate', 'Magnesium', 'Phosphate', 'Potassium', 'Bilirubin_total',
            'TroponinI', 'Hct', 'Hgb', 'PTT', 'WBC', 'Fibrinogen', 'Platelets',
            'Age', 'Gender', 'Unit1', 'Unit2', 'HospAdmTime', 'ICULOS'
        ]
        self.num_features = len(self.features)
        
        # In a real scenario, these medians should be computed on the training set only.
        # For this prototype, we'll initialize with zeros as provisional.
        self.provisional_medians = np.zeros(self.num_features)
        
    def __len__(self):
        return len(self.files)
        
    def __getitem__(self, idx):
        file_path = os.path.join(self.data_dir, self.files[idx])
        df = pd.read_csv(file_path, sep='|')
        
        # Extract features and target
        seq_len = min(len(df), self.max_seq_len)
        data = df[self.features].iloc[:seq_len].values
        target = df['SepsisLabel'].iloc[:seq_len].values
        
        # Compute Mask (m_t)
        mask = (~np.isnan(data)).astype(np.float32)
        
        # Compute Time Gap (delta_t)
        delta = np.zeros((seq_len, self.num_features), dtype=np.float32)
        for t in range(1, seq_len):
            for d in range(self.num_features):
                if mask[t-1, d] == 0:
                    delta[t, d] = delta[t-1, d] + 1
                else:
                    delta[t, d] = 1
                    
        # Fill NaNs with provisional values (e.g., 0 for now)
        data = np.nan_to_num(data)
        
        return {
            'data': torch.FloatTensor(data),
            'mask': torch.FloatTensor(mask),
            'delta': torch.FloatTensor(delta),
            'label': torch.FloatTensor(target),
            'seq_len': torch.tensor(seq_len)
        }

def collate_fn(batch):
    """
    Pads variable length sequences to the max length in the batch.
    """
    batch_size = len(batch)
    max_len = max([b['seq_len'] for b in batch])
    num_features = batch[0]['data'].shape[1]
    
    # Initialize padded tensors
    data = torch.zeros(batch_size, max_len, num_features)
    mask = torch.zeros(batch_size, max_len, num_features)
    delta = torch.zeros(batch_size, max_len, num_features)
    label = torch.zeros(batch_size, max_len)
    seq_lens = torch.zeros(batch_size, dtype=torch.long)
    
    for i, b in enumerate(batch):
        slen = b['seq_len']
        data[i, :slen, :] = b['data']
        mask[i, :slen, :] = b['mask']
        delta[i, :slen, :] = b['delta']
        label[i, :slen] = b['label']
        seq_lens[i] = slen
        
    return data, mask, delta, label, seq_lens
