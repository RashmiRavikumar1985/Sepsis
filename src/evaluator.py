import os
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    average_precision_score, 
    roc_auc_score, 
    precision_score, 
    recall_score, 
    f1_score, 
    roc_curve, 
    precision_recall_curve
)

def compute_metrics(y_true, y_pred_prob, threshold=0.5):
    """
    Computes standard evaluation metrics for time-series sepsis prediction.
    y_true: 1D array of ground truth labels (valid unpadded sequence steps)
    y_pred_prob: 1D array of predicted probabilities
    """
    auprc = average_precision_score(y_true, y_pred_prob)
    auroc = roc_auc_score(y_true, y_pred_prob)
    
    bin_preds = (np.array(y_pred_prob) >= threshold).astype(int)
    precision = precision_score(y_true, bin_preds, zero_division=0)
    recall = recall_score(y_true, bin_preds, zero_division=0)
    f1 = f1_score(y_true, bin_preds, zero_division=0)
    
    return {
        "AUPRC": float(auprc),
        "AUROC": float(auroc),
        "Precision": float(precision),
        "Recall": float(recall),
        "F1": float(f1)
    }

def save_evaluation_plots(y_true, y_pred_prob, output_dir, model_name="Model"):
    """
    Generates and saves PR and ROC curves to output_dir.
    """
    os.makedirs(output_dir, exist_ok=True)
    metrics = compute_metrics(y_true, y_pred_prob)
    
    # PR Curve
    precision_vals, recall_vals, _ = precision_recall_curve(y_true, y_pred_prob)
    plt.figure(figsize=(8, 6))
    plt.plot(recall_vals, precision_vals, label=f'AUPRC = {metrics["AUPRC"]:.4f}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve ({model_name})')
    plt.legend()
    plt.savefig(os.path.join(output_dir, "pr_curve.png"))
    plt.close()
    
    # ROC Curve
    fpr, tpr, _ = roc_curve(y_true, y_pred_prob)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'AUROC = {metrics["AUROC"]:.4f}')
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve ({model_name})')
    plt.legend()
    plt.savefig(os.path.join(output_dir, "roc_curve.png"))
    plt.close()
    
    return metrics
