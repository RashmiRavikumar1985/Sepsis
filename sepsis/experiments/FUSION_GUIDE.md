# Multi-Modal Fusion with Variable Context Windows (336h vs. 72h)
**FedSepsis-KG: Fusing GRU-D (336h), Transformer (336h), and Locked Temporal GAT (72h)**

---

## 1. Executive Summary & Design Grounding

### The Finalized Backbone Configuration
The multi-modal fusion architecture combines three distinct pretrained backbones, each operating at its optimal temporal context window:

* **GRU-D:** **336-hour** context window (Pretrained recurrent baseline)
* **Transformer:** **336-hour** context window (Pretrained self-attention baseline)
* **Temporal GAT (GAT-2):** **72-hour** context window (**Locked/Frozen** pretrained model)

---

### Purpose of the Multi-Scale Windows

Rather than forcing all models into a single arbitrary window length, the multi-scale setup leverages complementary clinical inductive biases:

1. **Longitudinal Stays & Observation Decay (336h Window):**
   - **GRU-D & Transformer** are tasked with capturing longer temporal and longitudinal patterns over the entire ICU admission.
   - They model time-since-last-observation intervals ($\Delta t$), missingness dynamics, long-range physiological drifts, and extended ward trajectories up to 336 hours.
2. **Acute Physiological Dynamics & Graph Interactions (72h Window):**
   - **Temporal GAT** is focused on short-term acute physiological states and dense cross-variable relationships within the critical first 72 hours of admission.
   - Clinical relationships across laboratory tests and vital signs (e.g., rapid co-variation between lactate, mean arterial pressure, and heart rate during early onset) are densest and most predictive during this acute window.

---

### The 72h Temporal GAT is Locked / Frozen

> [!IMPORTANT]
> **Temporal GAT (72h) is a locked pretrained model.** 
> * **Split Origin of AUROC 0.8041:** The 72-hour configuration achieved a **Test AUROC of 0.8041** (with **Test AUPRC 0.0848** and **Test F1 0.1010**) on the **held-out TEST split** (`splits["test"]`), as recorded in [experiments/results/gat_temporal/metrics.json](file:///Users/mouneshkumaran/Sepsis/Sepsis/sepsis/experiments/results/gat_temporal/metrics.json) via `evaluate_temporal.py`.
> * **Split Segregation for Fusion:** The **Test split** remains strictly sequestered and untouched during fusion training. All fusion parameters ($w_{\text{fusion}}, b$) and decision thresholds ($\tau^*$) are learned exclusively on the **Validation split** (`splits["val"]`).
> * **No retraining or window modification:** The fusion stage must not modify Temporal GAT's architecture, weights, or its 72h context window.
> * **Frozen backbones:** In the initial fusion stage, all three backbones (GRU-D, Transformer, and Temporal GAT) are kept strictly frozen (`requires_grad = False`).
> * **Goal of fusion:** The fusion stage focuses purely on learning how to optimally combine the output representations of the three frozen models on `splits["val"]`.

---

### Temporal Alignment & Masking Principles (No Artificial Data Generation)

Because Temporal GAT's context is 72 hours while GRU-D and Transformer process up to 336 hours, temporal alignment must be handled without corrupting clinical semantics:

* **Strict Temporal Validity:** Temporal GAT predictions contribute only where its 72h context is valid ($1 \le t \le 72$).
* **No Artificial Data Generation:** **Do NOT repeat, carry forward, or zero-pad Temporal GAT's predictions from 72h through 336h.** Artificially creating GAT predictions beyond its 72-hour context manufactures synthetic clinical data.
* **Dynamic Temporal Masking:**
  - **Hours 1–72 (Tri-Modal Phase):** All three models are valid. Output logits are combined using normalized fusion weights across GRU-D, Transformer, and Temporal GAT.
  - **Hours 73–336 (Bi-Modal Phase):** Temporal GAT is strictly masked out. The fusion layer routes predictions exclusively through GRU-D and Transformer, re-normalizing weights across the two active longitudinal models.

---

### Timestep Indexing: 0-Indexed Tensor Slices vs. 1-Indexed Clinical Hours

To eliminate off-by-one errors when implementing tensor operations, the relationship between Python array indexing and clinical timeline definitions is formalized below:

| Concept | Python Tensor Slice (0-Indexed) | Clinical Timeline (1-Indexed) | Active Backbones |
|---|---|---|---|
| **Acute Window** | `t = 0 .. T_short - 1` (slice `[:T_short]`, $T_{\text{short}} \le 72$) | Hospital Hours $1 \dots 72$ | **Tri-Modal:** GRU-D + Transformer + Temporal GAT |
| **Longitudinal Extension** | `t = T_short .. T_long - 1` (slice `[T_short:]`) | Hospital Hours $73 \dots 336$ | **Bi-Modal:** GRU-D + Transformer (GAT masked out) |

* **Slice `[:72]`:** Selects at most 72 hourly steps spanning indices `0` through `71`. Index `0` represents Hospital Hour 1 (admission hour).
* **Slice `[72:]`:** Selects steps starting from index `72` up to `T_long - 1`. Index `72` represents Hospital Hour 73.
* When a patient stay is shorter than 72 hours (e.g. $T = 39$), $T_{\text{short}} = \min(39, 72) = 39$. Both slices dynamically respect the patient's stay length, and `[T_short:]` is an empty slice `[39:39]` requiring no bi-modal branch evaluation.

---

### Mandatory Pre-Fusion Verification: Model Interfaces & Actual Output Shapes

> [!NOTE]
> All three pretrained models have been verified in the codebase to confirm their exact input arguments, output tensor shapes, and calling semantics:
> 
> 1. **Verified Actual Output Shapes:**
>    - `GRUD.forward()`: Returns a 2D tensor `(B, T_long)` of hourly logits produced by unrolling `GRUDCell` and applying `self.fc(h_fused).squeeze(-1)` across $T_{\text{long}}$.
>    - `TemporalTransformer.forward()`: Returns a 2D tensor `(B, T_long)` of hourly logits. *Crucial implementation detail:* `TemporalTransformer.forward()` already calls `.squeeze(-1)` internally on line 116 (`self.classifier(fused_rep).squeeze(-1)`). **Do not call `.squeeze(-1)` again** on the returned tensor in fusion code.
>    - `TemporalGAT.forward()`: Returns a 2D tensor `(B, T_short)` of hourly logits where $T_{\text{short}} \le 72$, similarly squeezed internally on line 331.
> 2. **Logit vs. Probability Semantics:** All backbones output unnormalized logits. Sigmoidal activation is applied externally or handled numerically via `BCEWithLogitsLoss`.
> 3. **Input Signatures:**
>    - `GRUD.forward(values, mask, delta, static_features)`: Recurrent model; does not accept `valid_mask` directly.
>    - `TemporalTransformer.forward(values, mask, delta, static_features, valid_mask)`: Requires boolean `valid_mask` for causal and padding attention masks.
>    - `TemporalGAT.forward(values, masks, deltas, static_features, edge_index=None, valid_mask=None)`: Requires graph adjacency baked via `set_adjacency()` prior to inference.
> 4. **Masked Evaluation:** Loss calculation and evaluation metrics must only evaluate timesteps where `valid_mask == True` to avoid scoring post-discharge padding.

---

## 2. Multi-Window Late Fusion Architecture

The primary implementation is **Late Fusion**, which combines the hourly logit predictions of each frozen backbone. Mid Fusion (embedding concatenation) is retained as an optional future extension.

```
Patient Data
├── 336h → GRU-D (Frozen Backbone)
├── 336h → Transformer (Frozen Backbone)
└── 72h  → Locked Temporal GAT (Frozen Backbone)
     ↓
Dynamic Temporal Masking:
 • Hours 1..72:   Tri-Modal Fusion (w_gru + w_trans + w_gat)
 • Hours 73..336: Bi-Modal Fusion  (w_gru + w_trans; GAT strictly masked)
     ↓
Fusion Head
     ↓
Final Sepsis Risk
```

---

## 3. Project Structure & Files

All baseline checkpoints remain untouched and reproducible. The fusion implementation resides in `experiments/fusion/`:

```
sepsis/
├── src/
│   ├── dataset_grud.py        # [Untouched] PhysioNetDatasetGRUD & collate_fn
│   ├── model_grud.py          # [Untouched] Pre-trained GRU-D (336h)
├── experiments/
│   ├── transformer/model.py   # [Untouched] Pre-trained Transformer (336h)
│   ├── gat/temporal_model.py  # [Untouched] Pre-trained Temporal GAT (72h)
│   ├── results/
│   │   ├── baseline/best_model.pt       # Pre-trained GRU-D checkpoint (336h)
│   │   ├── transformer/best_model.pt    # Pre-trained Transformer checkpoint (336h)
│   │   └── gat_temporal/best_model.pt   # LOCKED Pre-trained Temporal GAT checkpoint (72h)
│   └── fusion/                          # <── NEW FUSION MODULE
│       ├── __init__.py
│       ├── dataset.py         # Multi-window collation (slices 336h and 72h)
│       ├── model.py           # LateFusionModel with dynamic temporal masking
│       ├── train.py           # Optimizes fusion weights on VALIDATION split only
│       └── evaluate.py        # Independent threshold tuning & final TEST evaluation
```

---

## 4. Implementation Code

### File 1: `experiments/fusion/dataset.py`
*Extracts dual context windows (336h and 72h) dynamically per batch without duplicating data on disk.*

```python
"""
experiments/fusion/dataset.py
Multi-window collation for fusing 336h and 72h models.
"""

import torch
from src.dataset_grud import collate_fn

def multi_window_collate_fn(batch, short_limit=72):
    """
    Collate variable-length patient stays into dual context windows:
      - Long window: capped at dataset max (336h)
      - Short window: sliced to min(T_batch, short_limit) (72h)
    
    Returns:
      dict with:
        'long':  (values, mask, delta, static, labels, valid_mask)
        'short': (values, mask, delta, static, labels, valid_mask)
    """
    # 1. Collate full sequence (up to 336h)
    values, mask, delta, static, labels, valid_mask = collate_fn(batch)
    B, T, D = values.shape

    # 2. Slice short window strictly up to min(T, short_limit) (72h)
    T_short = min(T, short_limit)
    v_short = values[:, :T_short, :]
    m_short = mask[:, :T_short, :]
    d_short = delta[:, :T_short, :]
    l_short = labels[:, :T_short]
    vm_short = valid_mask[:, :T_short]

    return {
        'long': {
            'values': values, 'mask': mask, 'delta': delta,
            'static': static, 'labels': labels, 'valid_mask': valid_mask
        },
        'short': {
            'values': v_short, 'mask': m_short, 'delta': d_short,
            'static': static, 'labels': l_short, 'valid_mask': vm_short
        }
    }
```

---

### File 2: `experiments/fusion/model.py`
*Freezes all 3 pre-trained backbones and implements Late Fusion with proper temporal masking.*

```python
"""
experiments/fusion/model.py
Multi-Window Late Fusion with Dynamic Temporal Masking.
Backbones are strictly FROZEN. Only fusion parameters are trainable.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model_grud import GRUD
from experiments.transformer.model import TemporalTransformer
from experiments.gat.temporal_model import TemporalGAT
from experiments.gat.temporal_graph_builder import load_clinical_edges

class MultiWindowLateFusion(nn.Module):
    def __init__(
        self,
        grud_path: str,
        trans_path: str,
        gat_path: str,
        edges_csv: str,
        device: torch.device = torch.device("cpu")
    ):
        super().__init__()
        self.device = device

        # ── 1. Load GRU-D Backbone (336h) ──
        self.grud = GRUD(input_size=35, static_size=5, hidden_size=64, dropout=0.0).to(device)
        self.grud.load_state_dict(torch.load(grud_path, map_location=device))

        # ── 2. Load Transformer Backbone (336h) ──
        self.transformer = TemporalTransformer(
            input_size=35, static_size=5, d_model=64, n_heads=4,
            num_layers=2, dim_feedforward=128, dropout=0.0
        ).to(device)
        self.transformer.load_state_dict(torch.load(trans_path, map_location=device))

        # ── 3. Load Locked Temporal GAT Backbone (72h) ──
        self.gat = TemporalGAT(
            num_features=35, static_size=5, hidden_dim=64, out_dim=64,
            num_heads=2, num_gat_layers=2, dropout=0.0
        ).to(device)
        src_edges, dst_edges = load_clinical_edges(edges_csv)
        self.gat.set_adjacency(src_edges, dst_edges, add_self_loops=True)
        self.gat.load_state_dict(torch.load(gat_path, map_location=device))

        # ── 4. Freeze All Backbones Completely ──
        for model in [self.grud, self.transformer, self.gat]:
            for p in model.parameters():
                p.requires_grad = False
            model.eval()

        # ── 5. Trainable Late Fusion Parameters ──
        # Log-weights for [GRU-D, Transformer, Temporal GAT]
        self.fusion_weights = nn.Parameter(torch.tensor([0.33, 0.33, 0.34], dtype=torch.float32))
        self.bias = nn.Parameter(torch.zeros(1, dtype=torch.float32))

    def forward(self, batch_data):
        long_data  = batch_data['long']
        short_data = batch_data['short']

        v_l = long_data['values'].to(self.device)
        m_l = long_data['mask'].to(self.device)
        d_l = long_data['delta'].to(self.device)
        s_l = long_data['static'].to(self.device)
        vm_l = long_data['valid_mask'].to(self.device)

        v_s = short_data['values'].to(self.device)
        m_s = short_data['mask'].to(self.device)
        d_s = short_data['delta'].to(self.device)
        s_s = short_data['static'].to(self.device)
        vm_s = short_data['valid_mask'].to(self.device)

        # Forward pass through frozen backbones
        with torch.no_grad():
            logit_gru = self.grud(v_l, m_l, d_l, s_l)                                          # (B, T_long)
            logit_trans = self.transformer(v_l, m_l, d_l, s_l, valid_mask=vm_l)               # (B, T_long) - already 2D
            logit_gat = self.gat(v_s, m_s, d_s, s_s, valid_mask=vm_s)                         # (B, T_short <= 72) - already 2D

        B, T_long = logit_gru.shape
        T_short = logit_gat.shape[1]

        # ── Dynamic Temporal Masking ──
        # Phase 1 (t <= 72h): Tri-modal combination (all 3 models valid)
        w_tri = F.softmax(self.fusion_weights, dim=0)  # [w_gru, w_trans, w_gat]
        tri_logits = (
            w_tri[0] * logit_gru[:, :T_short] +
            w_tri[1] * logit_trans[:, :T_short] +
            w_tri[2] * logit_gat[:, :T_short]
        )

        # Phase 2 (t > 72h): Bi-modal combination (GRU-D + Transformer only)
        # GAT is strictly masked out. No artificial or repeated predictions are fabricated.
        if T_long > T_short:
            w_bi = F.softmax(self.fusion_weights[:2], dim=0)  # Re-normalize between [GRU-D, Transformer]
            bi_logits = (
                w_bi[0] * logit_gru[:, T_short:] +
                w_bi[1] * logit_trans[:, T_short:]
            )
            fused_logits = torch.cat([tri_logits, bi_logits], dim=1) + self.bias
        else:
            fused_logits = tri_logits + self.bias

        return fused_logits
```

---

### File 3: `experiments/fusion/train.py`
*Learns fusion weights strictly on the **Validation Split**. The Test Split remains completely untouched.*

```python
"""
experiments/fusion/train.py
Learns late fusion parameters on the validation split only.
"""

import os
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.dataset_grud import PhysioNetDatasetGRUD
from experiments.fusion.dataset import multi_window_collate_fn
from experiments.fusion.model import MultiWindowLateFusion

def train_fusion():
    sepsis_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    splits_path = os.path.join(sepsis_root, "artifacts", "splits.json")
    prep_cfg_path = os.path.join(sepsis_root, "artifacts", "preprocessing_config.json")
    edges_csv = os.path.join(sepsis_root, "experiments", "gat", "edges.csv")

    results_dir = os.path.join(sepsis_root, "experiments", "results", "fusion_late")
    os.makedirs(results_dir, exist_ok=True)

    with open(splits_path) as f: splits = json.load(f)
    with open(prep_cfg_path) as f: prep_cfg = json.load(f)

    # Checkpoint paths (336h GRU-D, 336h Transformer, LOCKED 72h GAT-2)
    grud_pt  = os.path.join(sepsis_root, "experiments", "results", "baseline", "best_model.pt")
    trans_pt = os.path.join(sepsis_root, "experiments", "results", "transformer", "best_model.pt")
    gat_pt   = os.path.join(sepsis_root, "experiments", "results", "gat_temporal", "best_model.pt")

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Calibrating Late Fusion on Validation Split ({device})...")

    model = MultiWindowLateFusion(
        grud_path=grud_pt, trans_path=trans_pt, gat_path=gat_pt,
        edges_csv=edges_csv, device=device
    ).to(device)

    # ONLY optimize fusion parameters (backbones are frozen)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    assert len(trainable_params) == 2, f"Expected 2 parameter tensors (weights, bias), got {len(trainable_params)}"

    data_dirs = [os.path.join(os.path.dirname(sepsis_root), "data")]
    val_ds = PhysioNetDatasetGRUD(data_dirs, splits["val"], prep_cfg_path, max_seq_len=336)
    val_loader = DataLoader(
        val_ds, batch_size=64, shuffle=True,
        collate_fn=lambda b: multi_window_collate_fn(b, short_limit=72)
    )

    optimizer = torch.optim.Adam(trainable_params, lr=0.01)
    pos_weight = torch.tensor([prep_cfg.get("class_weight", 54.54)]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction="none")

    # Fast calibration: 5 epochs on validation split
    model.train()
    for epoch in range(5):
        total_loss = 0.0
        for batch in val_loader:
            labels = batch['long']['labels'].to(device)
            mask = batch['long']['valid_mask'].to(device)

            optimizer.zero_grad()
            logits = model(batch)
            loss_all = criterion(logits, labels)
            loss = (loss_all * mask.float()).sum() / mask.float().sum().clamp(min=1)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1}/5 - Val Loss: {total_loss / len(val_loader):.4f}")

    w = torch.softmax(model.fusion_weights, dim=0).detach().cpu().numpy()
    print(f"Calibrated Tri-Modal Weights -> GRU-D: {w[0]:.3f} | Trans: {w[1]:.3f} | GAT-2 (72h): {w[2]:.3f}")

    save_path = os.path.join(results_dir, "best_model.pt")
    torch.save(model.state_dict(), save_path)
    print(f"Fusion weights saved to {save_path}")

if __name__ == "__main__":
    train_fusion()
```

---

### File 4: `experiments/fusion/evaluate.py`
*Performs **Independent Threshold Tuning** on the Validation Set, then evaluates on the untouched **Test Set**.*

```python
"""
experiments/fusion/evaluate.py
Two-step protocol:
  1. Optimize decision threshold on Validation set (maximizing F1).
  2. Evaluate calibrated model + optimal threshold on Held-Out Test Set.
"""

import os
import json
import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_fscore_support

from src.dataset_grud import PhysioNetDatasetGRUD
from experiments.fusion.dataset import multi_window_collate_fn
from experiments.fusion.model import MultiWindowLateFusion

def collect_predictions(model, dataloader, device):
    all_probs, all_labels = [], []
    with torch.no_grad():
        for batch in dataloader:
            labels = batch['long']['labels'].to(device)
            mask = batch['long']['valid_mask'].to(device)

            logits = model(batch)
            probs = torch.sigmoid(logits)

            all_probs.extend(probs[mask].cpu().numpy())
            all_labels.extend(labels[mask].cpu().numpy())
    return np.array(all_probs), np.array(all_labels)

def tune_threshold_on_val(val_probs, val_labels):
    best_thresh = 0.5
    best_f1 = -1.0
    for thresh in np.linspace(0.05, 0.95, 91):
        preds = (val_probs >= thresh).astype(int)
        _, _, f1, _ = precision_recall_fscore_support(val_labels, preds, average="binary", zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh
    return float(best_thresh), float(best_f1)

def evaluate_pipeline():
    sepsis_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    splits_path = os.path.join(sepsis_root, "artifacts", "splits.json")
    prep_cfg_path = os.path.join(sepsis_root, "artifacts", "preprocessing_config.json")
    edges_csv = os.path.join(sepsis_root, "experiments", "gat", "edges.csv")

    results_dir = os.path.join(sepsis_root, "experiments", "results", "fusion_late")
    fusion_pt = os.path.join(results_dir, "best_model.pt")

    with open(splits_path) as f: splits = json.load(f)
    with open(prep_cfg_path) as f: prep_cfg = json.load(f)

    grud_pt  = os.path.join(sepsis_root, "experiments", "results", "baseline", "best_model.pt")
    trans_pt = os.path.join(sepsis_root, "experiments", "results", "transformer", "best_model.pt")
    gat_pt   = os.path.join(sepsis_root, "experiments", "results", "gat_temporal", "best_model.pt")

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = MultiWindowLateFusion(
        grud_path=grud_pt, trans_path=trans_pt, gat_path=gat_pt,
        edges_csv=edges_csv, device=device
    ).to(device)
    model.load_state_dict(torch.load(fusion_pt, map_location=device))
    model.eval()

    data_dirs = [os.path.join(os.path.dirname(sepsis_root), "data")]

    # ── STEP 1: Threshold Optimization on Validation Set ──
    print("Step 1: Finding optimal threshold on Validation Split...")
    val_ds = PhysioNetDatasetGRUD(data_dirs, splits["val"], prep_cfg_path, max_seq_len=336)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, collate_fn=lambda b: multi_window_collate_fn(b, short_limit=72))
    val_probs, val_labels = collect_predictions(model, val_loader, device)
    best_thresh, val_f1 = tune_threshold_on_val(val_probs, val_labels)
    print(f"Optimal Validation Threshold: {best_thresh:.3f} (Val F1: {val_f1:.4f})")

    # ── STEP 2: Evaluation on Untouched Test Set ──
    print("Step 2: Evaluating on Untouched Test Split with fixed threshold...")
    test_ds = PhysioNetDatasetGRUD(data_dirs, splits["test"], prep_cfg_path, max_seq_len=336)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, collate_fn=lambda b: multi_window_collate_fn(b, short_limit=72))
    test_probs, test_labels = collect_predictions(model, test_loader, device)

    test_auroc = roc_auc_score(test_labels, test_probs)
    test_auprc = average_precision_score(test_labels, test_probs)
    test_preds = (test_probs >= best_thresh).astype(int)
    prec, rec, f1, _ = precision_recall_fscore_support(test_labels, test_preds, average="binary", zero_division=0)

    print(f"Test AUROC     : {test_auroc:.4f}")
    print(f"Test AUPRC     : {test_auprc:.4f}")
    print(f"Test Precision : {prec:.4f}")
    print(f"Test Recall    : {rec:.4f}")
    print(f"Test F1        : {f1:.4f} (at threshold {best_thresh:.3f})")

    metrics = {
        "model": "LateFusion (GRU-D 336h + Trans 336h + Locked GAT-2 72h)",
        "optimal_threshold": best_thresh,
        "Test AUROC": float(test_auroc),
        "Test AUPRC": float(test_auprc),
        "Test Precision": float(prec),
        "Test Recall": float(rec),
        "Test F1": float(f1)
    }
    with open(os.path.join(results_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)

if __name__ == "__main__":
    evaluate_pipeline()
```

---

## 5. Summary of Architectural Guardrails

| Requirement | Implementation Safeguard |
|---|---|
| **Backbone Integrity** | Pre-trained weights for GRU-D, Transformer, and GAT-2 are strictly frozen (`requires_grad = False`). |
| **Temporal Alignment** | For $t \le 72$h, tri-modal fusion applies. For $t > 72$h, GAT-2 is masked out and weights re-normalize over GRU-D and Transformer. |
| **No Fabricated Outputs** | Hours 73–336 never receive repeated, forward-filled, or zero-padded GAT outputs. |
| **Dataset Isolation** | Fusion weights are calibrated on the **Validation split** only. The **Test split** is evaluated strictly once. |
| **Threshold Calibration** | Threshold optimization is detached from training; optimal $\tau^*$ is found via sweep on Validation split before test inference. |
| **Interface Adaptation** | All sub-models are wrapped to accommodate their existing arguments (`valid_mask`, graph adjacency) and hourly logit shapes `(B, T)`. |

---

## 6. Future Extension: Mid Fusion (Optional)

Once Late Fusion is established, **Mid Fusion** can be implemented as a subsequent experiment without changing the underlying backbones:
1. Extract the pre-classification representations:
   - GRU-D hidden state sequence: $\mathbf{h}_{\text{gru}} \in \mathbb{R}^{B \times T_{\text{long}} \times 64}$
   - Transformer representation sequence: $\mathbf{h}_{\text{trans}} \in \mathbb{R}^{B \times T_{\text{long}} \times 32}$
   - Temporal GAT representation sequence: $\mathbf{h}_{\text{gat}} \in \mathbb{R}^{B \times T_{\text{short}} \times 64}$
2. Apply temporal masking: for $t > 72$, replace the missing GAT embedding slice with learned projection or pass through a bi-modal branch.
3. Train a lightweight 2-layer MLP head ($160 \to 64 \to 1$) on the concatenated representations using the same validation-split protocol.
