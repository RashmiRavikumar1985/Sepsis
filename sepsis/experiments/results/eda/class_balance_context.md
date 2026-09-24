# Class Imbalance Analysis

## Why Class Imbalance Matters
In early sepsis prediction, the vast majority of hours are non-septic (negative). Training a model naively on this data will cause the model to always predict '0' and achieve high accuracy, but fail to detect any actual sepsis cases. 

To counteract this, we must penalize false negatives more heavily by using a `pos_weight` in the binary cross-entropy loss. We must also evaluate the model using Area Under the Precision-Recall Curve (AUPRC) rather than just AUROC, as AUROC can be misleadingly high on imbalanced datasets.

## Statistics
- **Total Samples (Hours)**: 1552210
- **Positive Samples**: 27916 (1.80%)
- **Negative Samples**: 1524294 (98.20%)
- **Patients with Sepsis**: 2932
- **Patients without Sepsis**: 37404
- **Calculated `pos_weight`**: 54.60
