import os
import json
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset


class PhysioNetDatasetGRUD(Dataset):
    """
    PyTorch Dataset for GRU-D training on the PhysioNet/CinC 2019 dataset.

    Per patient produces:
        values:          (T, D)  clipped + z-scored dynamic features; NaN -> 0
        mask:            (T, D)  1 = observed, 0 = missing
        delta:           (T, D)  normalised hours since last observation per feature
        static_features: (S,)    encoded static attributes
        label:           (T,)    SepsisLabel per hour
        seq_len:         int     number of real (non-padded) timesteps

    All statistics (means, stds, clip bounds) come exclusively from the
    training split via preprocessing_config.json — never from val/test data.
    """

    def __init__(
        self,
        data_dirs: list,
        patient_ids: list,
        config_path: str,
        max_seq_len: int = 336,
    ) -> None:
        self.max_seq_len = max_seq_len

        # Build filename -> directory lookup across all data directories
        self.file_to_dir: dict = {}
        for d in data_dirs:
            for f in os.listdir(d):
                if f.endswith(".psv"):
                    self.file_to_dir[f] = d

        # Keep only patient IDs that exist on disk
        self.files = [f for f in patient_ids if f in self.file_to_dir]

        # Load preprocessing statistics from config
        with open(config_path, "r") as fp:
            config = json.load(fp)

        self.features = config["dynamic_features"]
        self.num_features = len(self.features)

        self.means = np.array(
            [config["train_means"][f] for f in self.features], dtype=np.float32
        )
        self.stds = np.array(
            [config["train_stds"][f] for f in self.features], dtype=np.float32
        )
        self.clip_lo = np.array(
            [config["clip_lo"][f] for f in self.features], dtype=np.float32
        )
        self.clip_hi = np.array(
            [config["clip_hi"][f] for f in self.features], dtype=np.float32
        )

        self.static_features = config["static_features"]
        self.static_encoding = config["static_encoding"]
        self.static_stats = config["static_stats"]

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> dict:
        fname = self.files[idx]
        path = os.path.join(self.file_to_dir[fname], fname)
        df = pd.read_csv(path, sep="|")

        seq_len = min(len(df), self.max_seq_len)
        raw = df[self.features].iloc[:seq_len].values.astype(np.float32)  # (T, D)
        label = df["SepsisLabel"].iloc[:seq_len].values.astype(np.float32)  # (T,)

        # ── Observation mask ──────────────────────────────────────────
        mask = (~np.isnan(raw)).astype(np.float32)  # (T, D)  1=observed

        # ── Clipping (observed values only, vectorised) ───────────────
        clipped = np.where(
            mask.astype(bool),
            np.clip(raw, self.clip_lo, self.clip_hi),
            raw,
        )

        # ── Z-score normalisation ─────────────────────────────────────
        values = (clipped - self.means) / self.stds
        values = np.nan_to_num(values, nan=0.0)  # NaN -> population mean (0)

        # ── Time gap delta (vectorised, no nested loops) ──────────────
        # delta[t, d] = hours since feature d was last observed at time t
        # Capped at 48 hours, then normalised to [0, 1]
        T, D = seq_len, self.num_features
        delta = np.zeros((T, D), dtype=np.float32)
        for t in range(1, T):
            # Where observed at t-1: reset to 1 (one hour gap)
            # Where missing at t-1: accumulate previous gap + 1
            delta[t] = np.where(mask[t - 1] == 1, 1.0, delta[t - 1] + 1.0)
        delta = np.minimum(delta, 48.0) / 48.0

        # ── Static feature encoding ───────────────────────────────────
        static_vals = []
        for sf in self.static_features:
            val = df[sf].iloc[0] if sf in df.columns else np.nan
            encoding = self.static_encoding[sf]
            stats = self.static_stats[sf]
            if pd.isna(val):
                processed = 0.0  # mean after normalisation; 0 for binary
            elif encoding == "normalize":
                std = stats["std"] if stats["std"] > 0 else 1.0
                processed = (float(val) - stats["mean"]) / std
            else:
                processed = float(val)
            static_vals.append(processed)

        return {
            "values": torch.from_numpy(values),
            "mask": torch.from_numpy(mask),
            "delta": torch.from_numpy(delta),
            "static_features": torch.from_numpy(
                np.array(static_vals, dtype=np.float32)
            ),
            "label": torch.from_numpy(label),
            "seq_len": seq_len,
        }


def collate_fn(batch: list) -> tuple:
    """
    Pad variable-length patient stays to the longest stay in the batch.
    Returns a boolean valid_time_mask — True for real timesteps, False for padding.
    """
    batch_size = len(batch)
    max_len = max(b["seq_len"] for b in batch)
    D = batch[0]["values"].shape[1]
    S = batch[0]["static_features"].shape[0]

    values = torch.zeros(batch_size, max_len, D)
    mask = torch.zeros(batch_size, max_len, D)
    delta = torch.zeros(batch_size, max_len, D)
    static_features = torch.zeros(batch_size, S)
    labels = torch.zeros(batch_size, max_len)
    valid_time_mask = torch.zeros(batch_size, max_len, dtype=torch.bool)

    for i, b in enumerate(batch):
        T = b["seq_len"]
        values[i, :T] = b["values"]
        mask[i, :T] = b["mask"]
        delta[i, :T] = b["delta"]
        static_features[i] = b["static_features"]
        labels[i, :T] = b["label"]
        valid_time_mask[i, :T] = True

    return values, mask, delta, static_features, labels, valid_time_mask