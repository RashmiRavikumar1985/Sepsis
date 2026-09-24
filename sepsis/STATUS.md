# FedSepsis-KG — Implementation Status

## 1. Executive Summary

This repository includes a robust baseline for early sepsis prediction using GRU-D, as well as the implemented **Temporal Transformer** (corrected causal baseline + 128-d tuned experiment) and **Graph Attention Network (GAT) Baseline**. The data preprocessing pipeline, EDA suite, and model training loops are fully operational. The 128-d tuned Transformer experiment is now complete — it matches GRU-D on AUPRC while remaining fully causal. Transformer tuning is now closed. Next milestone: **GAT-1** redesign. The Medical Knowledge Graph (SNOMED-CT mapped) and Federated Learning simulation remain future phases.

## 2. Current Project Architecture

```text
[CURRENT] EHR (PhysioNet PSV files)
            ↓ 
[CURRENT] Preprocessing (Clipping, Z-score)
            ↓ 
[CURRENT] Temporal Encoder (GRU-D / Transformer)
            |
            |     [CURRENT] Provisional Feature Graph (GAT)
            |            ↓
            +---->[FUTURE] Fusion
                         ↓
[CURRENT] Local Sepsis Predictor (Linear + Sigmoid)
                         ↓
                  [FUTURE] Federated Client (FedAvg/FedProx)
                         ↓
                  [FUTURE] Adaptive Aggregation & Personalization
                         ↓
                  [FUTURE] Explainable Sepsis Risk
```

## 3. Dataset Status

*   **Dataset:** PhysioNet / Computing in Cardiology Challenge 2019 Sepsis Dataset
*   **Location:** `training/training_setA/` and `training/training_setB/`
*   **Availability:** **YES**
*   **Patients/Files:** 40,336 `.psv` files
*   **Splits:** Train: 28,235 | Val: 6,050 | Test: 6,051
*   **Features:** 35 dynamic, 5 static

## 4. Model Inventory

| Model | File | Status | Executable | Trained | Evaluated |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **GRU-D** | `src/model_grud.py` | COMPLETED | YES | YES | YES |
| **Transformer (64-d causal baseline)** | `experiments/transformer/model.py` | COMPLETED | YES | YES | YES |
| **Transformer (128-d tuned causal)** | `experiments/transformer/model.py` | COMPLETED | YES | YES | YES |
| **GAT Baseline** | `experiments/gat/model.py` | COMPLETED | YES | YES | YES |
| **Medical KG** | N/A | NOT IMPLEMENTED | NO | NO | NO |

## 5. Performance Comparison — Final Results

| # | Model | Test AUPRC | Test AUROC | Best Val AUPRC | Best Val AUROC | Best Epoch | Notes |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 1 | **GRU-D** | 0.1125 | 0.8436 | — | — | — | Recurrent, explicit missingness handling |
| 2 | **Transformer 64-d (non-causal, historical)** | 0.3649 | 0.9463 | — | — | — | ⚠ Leaked future labels. Historical only. |
| 3 | **Transformer 64-d (corrected causal)** | 0.1047 | 0.8325 | — | — | — | Official causal baseline |
| 4 | **Transformer 128-d (tuned causal)** | **0.1120** | **0.8305** | 0.0952 | 0.8186 | 13 | AdamW + ReduceLROnPlateau. Matches GRU-D AUPRC. |
| 5 | **GAT (provisional graph)** | 0.0712 | 0.7112 | — | — | — | Provisional correlation graph, not a real KG |

## 6. Transformer Experiment Details

### 6.1 Shared Architecture (all Transformer experiments)
- Input: 35 dynamic features + 5 static features
- Channels: values + observation masks + time deltas = **105 input channels**
- ICULOS retained as a dynamic feature
- Causal attention mask: **REQUIRED — never remove**
  - Implementation: `torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device), diagonal=1)`
  - `True` = blocked (future), `False` = allowed (past/present)
- Sequence padding mask: **REQUIRED — never remove**
  - Implementation: `key_padding_mask = ~valid_mask`
  - Both masks are `dtype=torch.bool` — no type mismatch warning
- Same preprocessing artifacts (`preprocessing_config.json`)
- Same train/validation/test split (`splits.json`)
- Same evaluation protocol (val AUPRC for model selection, test evaluated once)

### 6.2 64-d Corrected Causal Baseline (`config.json`)
| Hyperparameter | Value |
| :--- | :--- |
| d_model | 64 |
| n_heads | 4 |
| num_layers | 2 |
| dim_feedforward | 128 |
| dropout | 0.3 |
| optimizer | Adam |
| learning_rate | 1e-3 |
| weight_decay | 1e-4 |
| epochs | 20 |
| patience | 5 |
| scheduler | None |

- **Checkpoint:** `sepsis.1/artifacts/baseline_transformer.pt` ← DO NOT OVERWRITE
- **Test AUROC:** 0.8325 | **Test AUPRC:** 0.1047

### 6.3 128-d Tuned Causal Experiment (`config_tuned.json`) — COMPLETED
| Hyperparameter | Value |
| :--- | :--- |
| d_model | 128 |
| n_heads | 4 |
| num_layers | 2 |
| dim_feedforward | 256 |
| dropout | 0.20 |
| optimizer | AdamW |
| learning_rate | 3e-4 |
| weight_decay | 1e-4 |
| epochs | 30 |
| patience | 7 |
| scheduler | ReduceLROnPlateau (factor=0.5, patience=2, mode=max on val AUPRC) |

- **Checkpoint:** `sepsis.1/artifacts/tuned_transformer.pt`
- **Metrics:** `sepsis.1/artifacts/tuned_metrics.json`
- **Best epoch:** 13 (early stopping triggered)
- **Test AUROC:** 0.8305 | **Test AUPRC:** 0.1120
- **Trained from scratch** — 64-d checkpoint architecturally incompatible

### 6.4 Key Findings
- The 128-d tuned model **matches GRU-D on AUPRC** (0.1120 vs 0.1125 — delta 0.0005)
- AUROC is flat across all three causal models (0.830–0.844 band) — consistent with correctly enforced causal masking
- The 64-d baseline was leaving ~7% AUPRC on the table relative to GRU-D; the tuned model recovered that gap
- Val AUPRC (0.0952) < Test AUPRC (0.1120): natural variance on a highly imbalanced dataset (1.8% sepsis), not overfitting

### 6.5 Transformer Tuning — CLOSED
No further Transformer architecture changes. The three-way comparison is complete:

| Model | Test AUROC | Test AUPRC |
| :--- | :--- | :--- |
| GRU-D baseline | 0.8436 | 0.1125 |
| Transformer 64-d corrected causal | 0.8325 | 0.1047 |
| Transformer 128-d tuned causal | 0.8305 | 0.1120 |

### 6.6 Artifact Path Note
`train.py` computes `project_root` as `sepsis.1/` (four `dirname` levels up from the script). As a result, checkpoints and metrics are saved to `sepsis.1/artifacts/` and `sepsis.1/experiments/results/transformer/`, not `sepsis/artifacts/`. This is consistent across all runs — do not move these files.

### 6.7 How to Run
```powershell
cd f:\Sepsis\sepsis.1\sepsis\experiments\transformer

# 64-d corrected causal baseline
python train.py

# 128-d tuned experiment (already complete — re-runs from scratch)
python train.py --config config_tuned.json
```

## 7. FedSepsis-KG Component Status

| Component | Status | Evidence | Next Action |
| :--- | :--- | :--- | :--- |
| **Dataset & Preprocessing** | [COMPLETED] | `preprocess.py` | None |
| **EDA Pipeline** | [COMPLETED] | `experiments/eda/` | None |
| **GRU-D Baseline** | [COMPLETED] | `src/train_grud.py` | None |
| **Transformer 64-d (corrected causal)** | [COMPLETED] | `experiments/transformer/` | None |
| **Transformer 128-d (tuned causal)** | [COMPLETED] | `sepsis.1/artifacts/tuned_metrics.json` | None — tuning closed |
| **Provisional Graph (GAT)** | [COMPLETED & FIXED] | `experiments/gat/` | Move to GAT-1 redesign |
| **Common Evaluator** | [COMPLETED] | `src/evaluator.py` | None |
| **Knowledge Graph** | [NOT IMPLEMENTED] | No files found | Design SNOMED-CT extraction |
| **Federated Learning (FL)** | [NOT IMPLEMENTED] | No files found | Write Flower FL simulation |

## 8. Research Progress

*   **Phase 1 & 2 (Dataset, Preproc, EDA):** 100%
*   **Phase 3 (Baselines — GRU-D):** 100%
*   **Phase 4 (Transformer — corrected causal + tuned):** 100%
*   **Phase 5 (GAT / Provisional Graph):** 100%
*   **Phase 6 (True Medical KG & Fusion):** 0%
*   **Phase 7 (Federated Learning):** 0%

**OVERALL PROJECT PROGRESS:** ~55%

## 9. Next Milestone — GAT-1

The provisional GAT used a correlation-based feature graph and scored 0.0712 AUPRC / 0.7112 AUROC — well below the Transformer and GRU-D. GAT-1 requires a properly designed medical knowledge graph (SNOMED-CT mapped clinical relationships) before the graph architecture can be meaningfully evaluated.

Planned GAT-1 work:
1. Extract SNOMED-CT clinical relationships for the 35 EHR features
2. Build a medically grounded node/edge graph
3. Re-evaluate GATBaseline on the proper graph
4. Compare against Transformer 128-d tuned as the primary causal baseline
