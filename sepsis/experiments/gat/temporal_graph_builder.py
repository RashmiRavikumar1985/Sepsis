"""
temporal_graph_builder.py  -  GAT-2 (Temporal GAT)  [OPTIMISED]

Node ordering (DETERMINISTIC):
    node_id(t, f) = t * F + f      F = 35

Two edge types:
    A. Clinical/Spatial: (f_i, t) -> (f_j, t) for every valid t
    B. Temporal (past->current ONLY): (f, t-1) -> (f, t) for consecutive valid t

SPEED NOTES:
  - build_temporal_edge_index_fast: fully vectorised (no Python loops over T)
  - build_batched_fast: validation-free path for the training loop
  - validate_no_future_leakage / validate_no_padded_edges: call from tests ONLY

Clinical edges loaded from edges.csv (training patients ONLY).
"""

import os
import pandas as pd
import numpy as np
import torch
from typing import Tuple, Dict, List, Optional

F = 35  # fixed dynamic features


def load_clinical_edges(edges_csv_path: str) -> Tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(edges_csv_path)
    sources = df['source'].values.astype(np.int64)
    targets = df['target'].values.astype(np.int64)
    assert sources.max() < F and targets.max() < F, "Edge index out of [0, F)"
    assert (sources >= 0).all() and (targets >= 0).all(), "Negative edge indices"
    return sources, targets


# ─────────────────────────────────────────────────────────
# Fast vectorised builder  —  NO Python loop over timesteps
# ─────────────────────────────────────────────────────────
def build_temporal_edge_index_fast(
    T: int,
    clinical_sources: np.ndarray,
    clinical_targets: np.ndarray,
    add_self_loops: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Fully vectorised (numpy broadcasting).
    Returns raw numpy arrays (src, dst, etype) - caller converts to torch.
    Edge types: 0=clinical, 1=temporal, 2=self-loop
    """
    E_c = len(clinical_sources)
    t_idx = np.arange(T, dtype=np.int64)
    f_idx = np.arange(F, dtype=np.int64)

    # A. Clinical edges: broadcast [T,1] + [1,E_c] -> [T,E_c] -> ravel
    offsets  = (t_idx * F)[:, None]                           # [T, 1]
    clin_src = (clinical_sources[None, :] + offsets).ravel()  # [T*E_c]
    clin_dst = (clinical_targets[None, :] + offsets).ravel()
    clin_e   = np.zeros(T * E_c, dtype=np.int64)

    # B. Temporal edges: [T-1,1]*F + [1,F] then +F for dst
    if T > 1:
        t_pairs  = np.arange(T - 1, dtype=np.int64)
        temp_src = (t_pairs[:, None] * F + f_idx[None, :]).ravel()       # [(T-1)*F]
        temp_dst = ((t_pairs[:, None] + 1) * F + f_idx[None, :]).ravel()
        temp_e   = np.ones((T - 1) * F, dtype=np.int64)
    else:
        temp_src = temp_dst = temp_e = np.empty(0, dtype=np.int64)

    # C. Self-loops
    if add_self_loops:
        sn    = np.arange(T * F, dtype=np.int64)
        sl_e  = np.full(T * F, 2, dtype=np.int64)
    else:
        sn = sl_e = np.empty(0, dtype=np.int64)

    src   = np.concatenate([clin_src,  temp_src, sn])
    dst   = np.concatenate([clin_dst,  temp_dst, sn])
    etype = np.concatenate([clin_e,    temp_e,   sl_e])
    return src, dst, etype


# ─────────────────────────────────────────────────────────
# Batched fast builder  (training loop)
# ─────────────────────────────────────────────────────────
def build_batched_fast(
    valid_seq_lens: List[int],
    clinical_sources: np.ndarray,
    clinical_targets: np.ndarray,
    T_pad: int,
    add_self_loops: bool = True,
    device: torch.device = None,
) -> Tuple[torch.Tensor, Dict]:
    """
    Fast, validation-free batched edge_index builder.
    Patient i's nodes are offset by i * T_pad * F.
    Returns (edge_index [2,E], info dict).
    """
    all_src, all_dst = [], []
    c_clin = c_temp = c_self = 0

    for i, vlen in enumerate(valid_seq_lens):
        if vlen == 0:
            continue
        offset = i * T_pad * F
        s, d, e = build_temporal_edge_index_fast(
            vlen, clinical_sources, clinical_targets, add_self_loops
        )
        all_src.append(s + offset)
        all_dst.append(d + offset)
        c_clin += int((e == 0).sum())
        c_temp += int((e == 1).sum())
        c_self += int((e == 2).sum())

    if all_src:
        src_np = np.concatenate(all_src)
        dst_np = np.concatenate(all_dst)
        edge_index = torch.tensor(np.stack([src_np, dst_np], axis=0), dtype=torch.long)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)

    if device is not None:
        edge_index = edge_index.to(device)

    info = dict(
        batch_size=len(valid_seq_lens), T_pad=T_pad,
        n_nodes_total=len(valid_seq_lens) * T_pad * F,
        n_clinical_edges=c_clin, n_temporal_edges=c_temp,
        n_self_loops=c_self,
        n_total_edges=int(edge_index.size(1)) if edge_index.numel() > 0 else 0,
        F=F,
    )
    return edge_index, info


# ─────────────────────────────────────────────────────────
# Original API wrappers (keep tests / evaluate working)
# ─────────────────────────────────────────────────────────
def build_temporal_edge_index(
    T: int,
    clinical_sources: np.ndarray,
    clinical_targets: np.ndarray,
    valid_timesteps: Optional[np.ndarray] = None,
    add_self_loops: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
    if valid_timesteps is not None and not np.all(valid_timesteps):
        valid_t = np.where(valid_timesteps)[0]
        T = int(valid_t.max()) + 1
    s, d, e = build_temporal_edge_index_fast(T, clinical_sources, clinical_targets, add_self_loops)
    edge_index = torch.tensor(np.stack([s, d], axis=0), dtype=torch.long)
    edge_type  = torch.tensor(e, dtype=torch.long)
    info = dict(
        n_nodes=T * F,
        n_clinical_edges=int((e == 0).sum()),
        n_temporal_edges=int((e == 1).sum()),
        n_self_loops=int((e == 2).sum()),
        n_total_edges=int(len(s)),
        T=T, F=F,
    )
    return edge_index, edge_type, info


def build_batched_temporal_edge_index(
    valid_seq_lens: List[int],
    clinical_sources: np.ndarray,
    clinical_targets: np.ndarray,
    T_pad: int,
    add_self_loops: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
    edge_index, info = build_batched_fast(
        valid_seq_lens, clinical_sources, clinical_targets, T_pad, add_self_loops
    )
    etypes = []
    for vlen in valid_seq_lens:
        if vlen > 0:
            _, _, e = build_temporal_edge_index_fast(
                vlen, clinical_sources, clinical_targets, add_self_loops
            )
            etypes.append(e)
    edge_type = torch.tensor(
        np.concatenate(etypes) if etypes else np.empty(0, np.int64), dtype=torch.long
    )
    return edge_index, edge_type, info


# ─────────────────────────────────────────────────────────
# Validation  (call from tests ONLY, not training hot-path)
# ─────────────────────────────────────────────────────────
def validate_no_future_leakage(edge_index: torch.Tensor, F: int = 35):
    src_t = edge_index[0] // F
    dst_t = edge_index[1] // F
    bad = src_t > dst_t
    if bad.any():
        raise AssertionError(
            f"FUTURE LEAKAGE: {bad.sum().item()} edges have src_t > dst_t\n"
            f"Sample: {edge_index[:, bad][:, :5]}"
        )


def validate_no_padded_edges(edge_index: torch.Tensor, valid_seq_len: int, F: int = 35):
    src_t = edge_index[0] // F
    dst_t = edge_index[1] // F
    if (src_t >= valid_seq_len).any():
        raise AssertionError(f"Edge source in padded region (t >= {valid_seq_len})")
    if (dst_t >= valid_seq_len).any():
        raise AssertionError(f"Edge dest in padded region (t >= {valid_seq_len})")


if __name__ == "__main__":
    import time
    edges_csv = os.path.join(os.path.dirname(__file__), "edges.csv")
    src, dst = load_clinical_edges(edges_csv)

    valid_lens = [48] * 8
    t0 = time.perf_counter()
    for _ in range(200):
        ei, info = build_batched_fast(valid_lens, src, dst, T_pad=48)
    t1 = time.perf_counter()
    print(f"Fast build (B=8, T=48): {(t1-t0)/200*1000:.2f}ms per batch")
    print(f"Total edges: {info['n_total_edges']}")

    ei, et, info = build_temporal_edge_index(6, src, dst)
    validate_no_future_leakage(ei)
    validate_no_padded_edges(ei, 6)
    print("Anti-leakage PASSED")
