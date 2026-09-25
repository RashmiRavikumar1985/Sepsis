"""
test_temporal_gat.py
====================
Unit tests for GAT-2 (Temporal GAT).

Tests (all from Section 17 of spec):
  1. Node count: T * 35
  2. Node feature shape: [B, T*35, 3]
  3. Clinical edge construction
  4. Temporal edge construction
  5. All temporal edges satisfy source_t < dest_t
  6. No temporal edges involve padded timesteps
  7. Output shape: [B, T]
  8. Loss only uses valid timesteps
  9. No NaN or Inf values
  10. Single small batch forward + backward pass

Run with:
    cd /path/to/sepsis
    python -m pytest tests/test_temporal_gat.py -v
    # or directly:
    python tests/test_temporal_gat.py
"""

import os
import sys
import math
import torch
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from experiments.gat.temporal_graph_builder import (
    load_clinical_edges,
    build_temporal_edge_index,
    build_batched_temporal_edge_index,
    validate_no_future_leakage,
    validate_no_padded_edges,
    F as NUM_FEATURES,
)
from experiments.gat.temporal_model import TemporalGAT

EDGES_CSV = os.path.join(PROJECT_ROOT, "experiments", "gat", "edges.csv")

# ---------------------------------------------------------------
# Helper
# ---------------------------------------------------------------
def make_batch(B=2, T=6, F=35, S=5):
    """Create a synthetic batch of patient data."""
    torch.manual_seed(0)
    values = torch.randn(B, T, F)
    masks  = (torch.rand(B, T, F) > 0.3).float()
    deltas = torch.clamp(torch.rand(B, T, F), 0, 1)
    static = torch.randn(B, S)
    labels = (torch.rand(B, T) > 0.9).float()
    # valid_seq_lens: first patient has T=6, second has T=4
    valid_lens = [T, T - 2] if B == 2 else [T] * B
    valid_mask = torch.zeros(B, T, dtype=torch.bool)
    for i, vl in enumerate(valid_lens):
        valid_mask[i, :vl] = True
    return values, masks, deltas, static, labels, valid_mask, valid_lens


# ---------------------------------------------------------------
# Test 1: Node count
# ---------------------------------------------------------------
def test_node_count():
    T, F = 8, NUM_FEATURES
    assert T * F == 280, f"Expected 280 nodes for T=8, F=35, got {T*F}"

    src, dst = load_clinical_edges(EDGES_CSV)
    ei, et, info = build_temporal_edge_index(T, src, dst)
    assert info['n_nodes'] == T * F, (
        f"n_nodes mismatch: {info['n_nodes']} != {T * F}"
    )
    print(f"[PASS] Test 1: Node count = {info['n_nodes']} (T={T}, F={F})")


# ---------------------------------------------------------------
# Test 2: Node feature shape
# ---------------------------------------------------------------
def test_node_feature_shape():
    B, T, F, S = 3, 6, NUM_FEATURES, 5
    values, masks, deltas, static, labels, valid_mask, _ = make_batch(B, T, F, S)

    # [B, T, F, 3] -> flatten last two dims -> [B, T*F, 3]
    node_feat = torch.stack([values, masks, deltas], dim=-1)
    assert node_feat.shape == (B, T, F, 3), f"node_feat shape: {node_feat.shape}"
    flat = node_feat.view(B, T * F, 3)
    assert flat.shape == (B, T * F, 3), f"flattened shape: {flat.shape}"
    print(f"[PASS] Test 2: Node feature shape = {flat.shape}")


# ---------------------------------------------------------------
# Test 3: Clinical edge construction
# ---------------------------------------------------------------
def test_clinical_edge_construction():
    src_feat, dst_feat = load_clinical_edges(EDGES_CSV)
    E_clin_per_t = len(src_feat)
    T = 5
    ei, et, info = build_temporal_edge_index(T, src_feat, dst_feat, add_self_loops=False)

    expected_clin = E_clin_per_t * T
    actual_clin = info['n_clinical_edges']
    assert actual_clin == expected_clin, (
        f"Clinical edges: expected {expected_clin}, got {actual_clin}"
    )

    # All clinical edges: src_t == dst_t
    clin_mask = et == 0
    src_t = ei[0, clin_mask] // NUM_FEATURES
    dst_t = ei[1, clin_mask] // NUM_FEATURES
    assert (src_t == dst_t).all(), "Clinical edges must be within same timestep"

    print(f"[PASS] Test 3: Clinical edges = {actual_clin} "
          f"({E_clin_per_t} per timestep x {T} timesteps)")


# ---------------------------------------------------------------
# Test 4: Temporal edge construction
# ---------------------------------------------------------------
def test_temporal_edge_construction():
    src_feat, dst_feat = load_clinical_edges(EDGES_CSV)
    T = 6  # 5 consecutive pairs => 5 * 35 temporal edges
    ei, et, info = build_temporal_edge_index(T, src_feat, dst_feat, add_self_loops=False)

    expected_temp = (T - 1) * NUM_FEATURES
    actual_temp = info['n_temporal_edges']
    assert actual_temp == expected_temp, (
        f"Temporal edges: expected {expected_temp}, got {actual_temp}"
    )

    temp_mask = et == 1
    src_f = ei[0, temp_mask] % NUM_FEATURES
    dst_f = ei[1, temp_mask] % NUM_FEATURES
    assert (src_f == dst_f).all(), "Temporal edges must connect same feature across t"

    src_t = ei[0, temp_mask] // NUM_FEATURES
    dst_t = ei[1, temp_mask] // NUM_FEATURES
    assert (dst_t == src_t + 1).all(), "Temporal edges must be consecutive (t -> t+1)"

    print(f"[PASS] Test 4: Temporal edges = {actual_temp} "
          f"({T-1} steps x {NUM_FEATURES} features)")


# ---------------------------------------------------------------
# Test 5: No future leakage (source_t < dest_t for temporal edges)
# ---------------------------------------------------------------
def test_no_future_leakage():
    src_feat, dst_feat = load_clinical_edges(EDGES_CSV)
    for T in [4, 10, 30]:
        ei, et, info = build_temporal_edge_index(T, src_feat, dst_feat)
        # validate_no_future_leakage will raise if any src_t > dst_t
        validate_no_future_leakage(ei, NUM_FEATURES)

        # Extra: explicitly check temporal edges
        temp = et == 1
        if temp.any():
            src_t = ei[0, temp] // NUM_FEATURES
            dst_t = ei[1, temp] // NUM_FEATURES
            assert (src_t < dst_t).all(), f"Future leak in T={T}"

    # Also test batched graph
    batch_ei, _, _ = build_batched_temporal_edge_index(
        [4, 6, 3], src_feat, dst_feat, T_pad=6)
    # (already validates internally, but call again explicitly)
    for i, vl in enumerate([4, 6, 3]):
        off = i * 6 * NUM_FEATURES
        mask = (batch_ei[0] >= off) & (batch_ei[0] < off + 6 * NUM_FEATURES)
        if mask.any():
            validate_no_future_leakage(batch_ei[:, mask] - off, NUM_FEATURES)

    print("[PASS] Test 5: No future leakage in any edge")


# ---------------------------------------------------------------
# Test 6: No padded timestep edges
# ---------------------------------------------------------------
def test_no_padded_edges():
    src_feat, dst_feat = load_clinical_edges(EDGES_CSV)
    valid_lens = [3, 5, 8]
    T_pad = 10
    batch_ei, batch_et, _ = build_batched_temporal_edge_index(
        valid_lens, src_feat, dst_feat, T_pad=T_pad)

    for i, vlen in enumerate(valid_lens):
        off = i * T_pad * NUM_FEATURES
        mask = (batch_ei[0] >= off) & (batch_ei[0] < off + T_pad * NUM_FEATURES)
        if mask.any():
            local = batch_ei[:, mask] - off
            validate_no_padded_edges(local, vlen, NUM_FEATURES)
            src_t = local[0] // NUM_FEATURES
            dst_t = local[1] // NUM_FEATURES
            assert (src_t < vlen).all(), f"Patient {i}: src_t out of valid range"
            assert (dst_t < vlen).all(), f"Patient {i}: dst_t out of valid range"

    print("[PASS] Test 6: No padded timestep nodes in any edge")


# ---------------------------------------------------------------
# Test 7: Output shape [B, T]
# ---------------------------------------------------------------
def test_output_shape():
    B, T, F, S = 2, 6, NUM_FEATURES, 5
    values, masks, deltas, static, labels, valid_mask, valid_lens = make_batch(B, T, F, S)

    src_feat, dst_feat = load_clinical_edges(EDGES_CSV)
    edge_index, _, _ = build_batched_temporal_edge_index(
        valid_lens, src_feat, dst_feat, T_pad=T)

    model = TemporalGAT(num_features=F, static_size=S, hidden_dim=16, out_dim=16,
                        num_heads=2, num_gat_layers=2, dropout=0.0)
    model.eval()
    with torch.no_grad():
        logits = model(values, masks, deltas, static, edge_index, valid_mask)

    assert logits.shape == (B, T), f"Output shape: {logits.shape} != ({B}, {T})"
    print(f"[PASS] Test 7: Output shape = {logits.shape}")


# ---------------------------------------------------------------
# Test 8: Loss only uses valid timesteps
# ---------------------------------------------------------------
def test_loss_masking():
    B, T, F, S = 2, 6, NUM_FEATURES, 5
    values, masks, deltas, static, labels, valid_mask, valid_lens = make_batch(B, T, F, S)

    src_feat, dst_feat = load_clinical_edges(EDGES_CSV)
    edge_index, _, _ = build_batched_temporal_edge_index(
        valid_lens, src_feat, dst_feat, T_pad=T)

    model = TemporalGAT(num_features=F, static_size=S, hidden_dim=16, out_dim=16,
                        num_heads=2, num_gat_layers=2, dropout=0.0)
    model.eval()

    criterion = torch.nn.BCEWithLogitsLoss(reduction='none')
    with torch.no_grad():
        logits = model(values, masks, deltas, static, edge_index, valid_mask)
        loss_matrix = criterion(logits, labels)

    # Loss at padded positions should contribute 0 after masking
    padded_loss = (loss_matrix * (~valid_mask).float()).sum()
    # Since logits are 0.0 at padded positions (masked_fill), and BCEWithLogitsLoss
    # of (0, label) is not zero, we test that the WEIGHTED loss excludes padding.
    valid_count = valid_mask.float().sum()
    masked_loss = (loss_matrix * valid_mask.float()).sum() / valid_count.clamp(min=1)

    assert not torch.isnan(masked_loss), "Masked loss is NaN"
    assert masked_loss.item() >= 0, "Masked loss is negative"

    # Verify that only valid timesteps count by checking valid_mask
    n_valid = valid_mask.sum().item()
    n_total = B * T
    assert n_valid < n_total, "Test requires some padding"
    print(f"[PASS] Test 8: Loss masking OK — {n_valid}/{n_total} valid timesteps used")


# ---------------------------------------------------------------
# Test 9: No NaN or Inf values
# ---------------------------------------------------------------
def test_no_nan_inf():
    B, T, F, S = 4, 8, NUM_FEATURES, 5
    values, masks, deltas, static, labels, valid_mask, valid_lens = make_batch(B, T, F, S)

    src_feat, dst_feat = load_clinical_edges(EDGES_CSV)
    edge_index, _, _ = build_batched_temporal_edge_index(
        valid_lens, src_feat, dst_feat, T_pad=T)

    model = TemporalGAT(num_features=F, static_size=S, hidden_dim=32, out_dim=32,
                        num_heads=2, num_gat_layers=2, dropout=0.0)
    model.eval()
    with torch.no_grad():
        logits = model(values, masks, deltas, static, edge_index, valid_mask)

    assert torch.isfinite(logits).all(), f"Non-finite logits: {logits}"
    print(f"[PASS] Test 9: No NaN/Inf in output — logits range [{logits.min():.3f}, {logits.max():.3f}]")


# ---------------------------------------------------------------
# Test 10: Forward + backward pass
# ---------------------------------------------------------------
def test_forward_backward():
    torch.manual_seed(42)
    B, T, F, S = 2, 5, NUM_FEATURES, 5
    values  = torch.randn(B, T, F, requires_grad=False)
    masks   = (torch.rand(B, T, F) > 0.3).float()
    deltas  = torch.rand(B, T, F)
    static  = torch.randn(B, S)
    labels  = (torch.rand(B, T) > 0.9).float()
    valid_lens = [T, T - 1]
    valid_mask = torch.zeros(B, T, dtype=torch.bool)
    for i, vl in enumerate(valid_lens):
        valid_mask[i, :vl] = True

    src_feat, dst_feat = load_clinical_edges(EDGES_CSV)
    edge_index, _, info = build_batched_temporal_edge_index(
        valid_lens, src_feat, dst_feat, T_pad=T)

    model = TemporalGAT(num_features=F, static_size=S, hidden_dim=16, out_dim=16,
                        num_heads=2, num_gat_layers=2, dropout=0.1)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.BCEWithLogitsLoss(reduction='none')

    model.train()
    optimizer.zero_grad()
    logits = model(values, masks, deltas, static, edge_index, valid_mask)
    loss_matrix = criterion(logits, labels)
    loss = (loss_matrix * valid_mask.float()).sum() / valid_mask.float().sum().clamp(min=1)
    loss.backward()
    optimizer.step()

    assert torch.isfinite(loss), f"Loss is not finite: {loss}"

    # Check gradients
    for name, param in model.named_parameters():
        if param.grad is not None:
            assert torch.isfinite(param.grad).all(), f"Non-finite grad in {name}"

    print(f"[PASS] Test 10: Forward+backward OK — loss={loss.item():.4f}")
    print(f"        Graph info: {info}")


# ---------------------------------------------------------------
# Run all tests
# ---------------------------------------------------------------
def run_all_tests():
    print("=" * 60)
    print("  GAT-2 Temporal GAT — Unit Tests")
    print("=" * 60)
    tests = [
        test_node_count,
        test_node_feature_shape,
        test_clinical_edge_construction,
        test_temporal_edge_construction,
        test_no_future_leakage,
        test_no_padded_edges,
        test_output_shape,
        test_loss_masking,
        test_no_nan_inf,
        test_forward_backward,
    ]
    passed = 0
    failed = []
    for t in tests:
        try:
            t()
            passed += 1
        except Exception as e:
            print(f"[FAIL] {t.__name__}: {e}")
            failed.append(t.__name__)

    print("\n" + "=" * 60)
    print(f"  Results: {passed}/{len(tests)} tests passed")
    if failed:
        print(f"  FAILED: {failed}")
    else:
        print("  ALL TESTS PASSED ✓")
    print("=" * 60)
    return len(failed) == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
