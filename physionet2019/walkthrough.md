# Walkthrough: PhysioNet 2019 Sepsis Prediction

## 1. Dataset Analysis

We analyzed all **40,336** patient `.psv` files across `training_setA` and `training_setB`.

### Key Findings
| Issue | Detail |
|---|---|
| **Extreme Sparsity** | Most lab features are 85–100% missing (e.g., `EtCO2` = 100%, `TroponinI` = 99.8%) |
| **Severe Class Imbalance** | Only **2.2%** of hourly rows are sepsis-positive (`SepsisLabel=1`) |
| **Constant Column** | `EtCO2` has zero variance across the entire dataset |

---

## 2. Approach 1: RITS/BRITS (On Hold)

We started building a RITS (Recurrent Imputation for Time Series) architecture in PyTorch. Files created but **paused** in favor of the GRU-D approach:

| File | Status |
|---|---|
| [dataset.py](file:///c:/Users/rashm/Downloads/physionet2019/src/dataset.py) | ✅ Created (RITS-specific loader) |
| [models.py](file:///c:/Users/rashm/Downloads/physionet2019/src/models.py) | ✅ Created (RITS + SepsisClassifier) |
| Training script | ❌ Not started |

---

## 3. Approach 2: GRU-D Pipeline (Active)

### 3a. Preprocessing ([preprocess.py](file:///c:/Users/rashm/Downloads/physionet2019/src/preprocess.py))

Reads all 40,336 files in parallel and produces two artifact files:

- **[preprocessing_config.json](file:///c:/Users/rashm/Downloads/physionet2019/artifacts/preprocessing_config.json)**: Training-only means, stds, clipping bounds, filtered feature list
- **[splits.json](file:///c:/Users/rashm/Downloads/physionet2019/artifacts/splits.json)**: Patient-level train/val/test split IDs (kept separate to prevent data leakage)

#### 4 Critical Issues Fixed
1. **Data Leakage Prevention** — Splits saved separately; stats computed strictly from training patients only
2. **Auto-removal of unusable features** — `EtCO2` (100% missing) automatically filtered out; 35 dynamic features kept
3. **Robust Outlier Clipping** — 1st/99th percentile bounds computed from 2,000 sampled training patients
4. **Static Feature Handling** — `Age`, `Gender`, `Unit1`, `Unit2`, `HospAdmTime` extracted with encoding strategy (z-score or binary)

#### Split Summary
| Split | Patients |
|---|---|
| Train | 28,235 |
| Validation | 6,050 |
| Test | 6,051 |

- **Positive class weight**: 54.54

### 3b. Dataset Loader ([dataset_grud.py](file:///c:/Users/rashm/Downloads/physionet2019/src/dataset_grud.py))

- Reads `.psv` files per patient
- Applies robust clipping → z-scoring (training stats only) → NaN → 0
- Computes **mask** ($m_t$: 1 if observed) and **delta** ($\Delta_t$: hours since last observation)
- Encodes static features (normalized or binary)
- Custom `collate_fn` pads variable-length stays and returns `valid_time_mask`

### 3c. Model ([model_grud.py](file:///c:/Users/rashm/Downloads/physionet2019/src/model_grud.py))

Implements the GRU-D architecture (Che et al. 2018):

- **Learned Input Decay**: $\gamma_x = \exp(-\text{ReLU}(W_{\gamma_x} \Delta_t + b))$
- **Learned Hidden-State Decay**: $\gamma_h = \exp(-\text{ReLU}(W_{\gamma_h} \Delta_t + b))$
- **Imputed Input**: $\hat{x}_t = m_t \cdot x_t + (1 - m_t) \cdot (\gamma_x \cdot x_{\text{last}})$
- **GRU Update**: Standard GRU gates on `concat(x_hat, mask)` with decayed hidden state
- **Static Feature Fusion**: Static features concatenated with GRU hidden state at every timestep before the classification head
- **Output**: Per-hour logits → sigmoid for probability

### 3d. Training Script ([train_grud.py](file:///c:/Users/rashm/Downloads/physionet2019/src/train_grud.py))

- **Loss**: Masked BCEWithLogitsLoss (only non-padded hours count), with `pos_weight=54.54`
- **Optimizer**: Adam (lr=1e-3)
- **Scheduler**: ReduceLROnPlateau on AUPRC
- **Early Stopping**: Patience=5 on validation AUPRC
- **Gradient Clipping**: max_norm=1.0
- **Device Support**: CUDA → Apple MPS → CPU (auto-detected)
- **`--subset` flag**: Allows progressive training on smaller patient counts

### 3e. Smoke Test ([smoke_test_grud.py](file:///c:/Users/rashm/Downloads/physionet2019/tests/smoke_test_grud.py))

Loads 20 patients, runs 1 batch through the full pipeline, verifies tensor shapes and gradient flow. ✅ Passed.

---

## 4. Training Results So Far

| Experiment | Patients | Best Val AUPRC | Best Val AUROC | Checkpoint |
|---|---|---|---|---|
| Subset 5,000 | 5,000 train / 1,000 val | **0.1020** | 0.8146 | `best_grud_5000.pt` ✅ |
| Subset 10,000 | 10,000 train / 2,000 val | **0.1036** | 0.8329 | `best_grud_subset_10000.pt` ✅ |
| Subset 20,000 | 20,000 train / 4,000 val | **0.0952** | 0.8200 | `best_grud_subset_20000.pt` ✅ |
| Full | 28,235 train / 6,050 val | Pending | — | `best_grud_full.pt` |

---

## 5. Path Compatibility Status

- **Relative Paths**: ✅ All hardcoded paths in `train_grud.py`, `preprocess.py`, and `smoke_test_grud.py` have been replaced with dynamic `PROJECT_ROOT` cross-platform paths. Works seamlessly on Windows, Mac, and Linux.

### What your friend needs to send back:
Only the **`checkpoints/`** folder containing the trained `.pt` model files. No need to zip the entire project — the training data and code are identical; only the model weights differ.

---

## Project Structure

```
physionet2019/
├── training/
│   ├── training_setA/       # ~20k .psv files
│   └── training_setB/       # ~20k .psv files
├── artifacts/
│   ├── preprocessing_config.json   # Training stats, clipping bounds, feature list
│   └── splits.json                 # Train/Val/Test patient IDs (separate file)
├── checkpoints/
│   └── best_grud_5000.pt           # Best model from 5k subset run
├── src/
│   ├── __init__.py
│   ├── preprocess.py               # Parallel preprocessing pipeline
│   ├── dataset_grud.py             # PyTorch Dataset for GRU-D
│   ├── model_grud.py               # GRU-D architecture
│   ├── train_grud.py               # Training loop with early stopping
│   ├── dataset.py                  # (RITS - on hold)
│   └── models.py                   # (RITS - on hold)
└── tests/
    └── smoke_test_grud.py          # End-to-end shape & gradient test
```
