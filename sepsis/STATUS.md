# FedSepsis-KG — Implementation Status

## 1. Executive Summary

This repository now includes a robust baseline for early sepsis prediction using GRU-D and RITS, as well as the newly implemented **Temporal Transformer** and **Graph Attention Network (GAT) Baseline**. The data preprocessing pipeline, Exploratory Data Analysis (EDA) suite, and model training loops are fully operational. The initial Phase of the "FedSepsis-KG" project (Transformer and Graph components) has been successfully scaffolded and smoke-tested. The Medical Knowledge Graph (SNOMED-CT mapped) and Federated Learning simulation are the next major milestones.

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
| **Transformer** | `experiments/transformer/model.py` | COMPLETED | YES | YES | YES |
| **GAT-0 (Static)** | `experiments/gat/model.py` | COMPLETED | YES | YES | YES |
| **GAT-2 (Temporal GAT)** | `experiments/gat/temporal_model.py` | COMPLETED | YES | YES | YES |
| **Medical KG** | N/A | MISSING | NO | NO | NO |

## 5. Performance Comparison

> ⚠️ Transformer 0.3649 AUPRC used future label leakage — **disqualified**. Valid causal baselines: 0.10–0.11 band.

| # | Model | Test AUPRC | Test AUROC | Test F1 | Notes |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 1 | **GRU-D** | **0.1125** | **0.8436** | — | Best overall |
| 2 | **Transformer 64-d (causal)** | 0.1047 | 0.8325 | — | Corrected causal mask |
| 3 | **GAT-2 (Temporal GAT)** | **0.0848** | **0.8041** | 0.1010 | Dense temporal GAT, max_seq_len=72, beats GAT-0 by +19% AUPRC |
| 4 | **GAT-0 (Static Correlation)** | 0.0712 | 0.7112 | 0.1124 | Provisional correlation graph |
| — | ~~Transformer (non-causal)~~ | ~~0.3649~~ | ~~0.9463~~ | — | ⚠ Future leakage — invalid |

## 6. FedSepsis-KG Component Status

| Component | Status | Evidence | Next Action |
| :--- | :--- | :--- | :--- |
| **Dataset & Preprocessing** | [COMPLETED] | `preprocess.py` | None |
| **EDA Pipeline** | [COMPLETED] | `experiments/eda/` | Analyze output distributions |
| **GRU-D Baseline** | [COMPLETED] | `train_grud.py` | None |
| **Temporal Transformer** | [COMPLETED & FIXED] | `experiments/transformer/` | Full dataset GPU training |
| **GAT-0 (Static Correlation)** | [COMPLETED] | `experiments/gat/` | None — superseded by GAT-2 |
| **GAT-2 (Temporal GAT)** | [COMPLETED] | `experiments/gat/temporal_model.py` | Improve with medical KG edges |
| **Common Evaluator** | [COMPLETED] | `src/evaluator.py` | Standardized evaluation metric |
| **Knowledge Graph** | [NOT IMPLEMENTED] | No files found | Design SNOMED CT extraction |
| **Federated Learning (FL)** | [NOT IMPLEMENTED] | No files found | Write Flower FL simulation |

## 7. Research Progress

*   **Phase 1 & 2 (Dataset, Preproc, EDA):** 100%
*   **Phase 3 (Baselines):** 100%
*   **Phase 4 (Transformer):** 100%
*   **Phase 5 (GAT / Temporal GAT — GAT-2):** 100%  ← GAT-2 AUROC=0.8041 / AUPRC=0.0848
*   **Phase 6 (True Medical KG & Fusion):** 0%
*   **Phase 7 (Federated Learning):** 0%

**OVERALL PROJECT PROGRESS:** 55%
