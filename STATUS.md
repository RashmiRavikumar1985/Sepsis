# FedSepsis-KG — Project Status

## Phase Overview

| Phase | Description | Status |
|-------|-------------|--------|
| Phase 1 | GRU-D temporal baseline | ✅ Complete |
| Phase 2 | GAT graph baseline | ✅ Fixed — Ready to Train |
| Phase 3 | GRU-D + GAT combined | ⏳ Not started |
| Phase 4 | Federated Learning | ⏳ Not started |

---

## Phase 1 — GRU-D

**Architecture**: GRU-D with Causal Multi-Head Attention (8 heads)

**Results**:
| Run | AUROC | AUPRC |
|-----|-------|-------|
| Baseline GRU-D (original) | 0.8436 | 0.1125 |
| Attention GRU-D (full dataset) | training... | training... |

**Key files**:
- `src/model_grud.py` — GRUDCell + CausalMultiHeadAttention + GRUD
- `src/train_grud.py` — training script
- `src/dataset_grud.py` — PhysioNetDatasetGRUD + collate_fn
- `src/config_grud.json` — hyperparameters (hidden=128, heads=8, AdamW, cosine annealing)
- `checkpoints/best_grud_full.pt` — best checkpoint
- `artifacts/baseline_grud.pt` — canonical copy for Phase 3

**Fixes applied**:
- ✅ PROJECT_ROOT derived from `__file__` — no hardcoded paths
- ✅ `valid_mask` applied to output logits (`masked_fill`)
- ✅ GRU-D cell state not updated on padded timesteps
- ✅ Vectorised delta computation (no nested loops)
- ✅ All hyperparameters in `config_grud.json`
- ✅ Cosine annealing + warmup scheduler
- ✅ AdamW with weight decay

---

## Phase 2 — GAT

**Architecture**: 2-layer GATConv + mean node pooling + static fusion

**Results (old broken implementation)**:
| Model | AUROC | AUPRC |
|-------|-------|-------|
| Old GAT (broken) | 0.7112 | 0.0712 |

**Graph**: 35 nodes (clinical features), 244 edges (Pearson correlation > 0.2)

**Key files**:
- `experiments/gat/model.py` — GATBaseline (fixed)
- `experiments/gat/train.py` — training script (fixed)
- `experiments/gat/evaluate.py` — evaluation script (fixed)
- `experiments/gat/config_gat.json` — hyperparameters (hidden=64, out=128, heads=4, AdamW)
- `experiments/gat/graph_builder.py` — builds edges.csv from training correlations
- `experiments/gat/graph.json` / `edges.csv` / `nodes.csv` — graph topology

**Fixes applied**:
- ✅ Created `config_gat.json` — own config, no longer borrows from transformer
- ✅ `valid_mask` applied to output logits in `model.py`
- ✅ Increased model capacity: hidden=64, out=128, heads=4 (was 32/64/2)
- ✅ Added LayerNorm after each GATConv layer
- ✅ Added deeper fusion network (2-layer MLP)
- ✅ Xavier weight initialisation
- ✅ `train.py` loads `config_gat.json`, two-candidate data path fallback
- ✅ `evaluate.py` derives `num_nodes`/`static_size` from preprocessing_config
- ✅ `evaluate.py` two-candidate data path fallback
- ✅ `torch_geometric` 2.8.0 installed

**To train**:
```bash
cd f:\Sepsis\sepsis.1\sepsis\experiments\gat
python train.py          # full dataset
python train.py --subset 2000   # fast debug
```

---

## Phase 3 — Combined GRU-D + GAT

**Architecture**:
```
GRU-D ──→ h_t ──┐
                ├──→ [h_t ║ g_t] → classifier
GAT   ──→ g_t ──┘
```

**Status**: Waiting for Phase 1 and Phase 2 training results.

**Inputs required**:
- `artifacts/baseline_grud.pt` — from Phase 1
- `artifacts/baseline_gat.pt` — from Phase 2

---

## Phase 4 — Federated Learning

**Status**: Not started. Depends on Phase 3.

---

## Baseline Results Summary

| Model | Test AUROC | Test AUPRC | Status |
|-------|-----------|------------|--------|
| GRU-D (original baseline) | 0.8436 | 0.1125 | ✅ Done |
| Transformer (tuned) | 0.9463 | 0.3649 | ✅ Done |
| GAT (old broken) | 0.7112 | 0.0712 | ❌ Broken |
| GRU-D + Attention (full) | TBD | TBD | 🔄 Training |
| GAT (fixed) | TBD | TBD | ⏳ Ready to train |
| GRU-D + GAT combined | TBD | TBD | ⏳ Phase 3 |

---

## Dependencies

- PyTorch 2.5.1+cu124
- torch_geometric 2.8.0
- torch_scatter / torch_sparse / torch_cluster (cu124 builds)
- scikit-learn, pandas, numpy, tqdm, matplotlib
