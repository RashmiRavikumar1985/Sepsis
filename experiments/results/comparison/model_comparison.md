# Experimental Model Comparison

| Model | Test AUPRC | Test AUROC | Test Precision | Test Recall | Test F1 |
| :--- | ---: | ---: | ---: | ---: | ---: |
| GRU-D | 0.1307 | 0.8464 | 0.1788 | 0.2351 | 0.2031 |
| Transformer (d=64) | 0.3649 | 0.9463 | 0.0928 | 0.7200 | 0.1644 |
| Transformer (tuned d=128) | 0.1120 | 0.8305 | N/A | N/A | N/A |
| GAT (Baseline) | 0.1083 | 0.7983 | 0.1519 | 0.2554 | 0.1905 |

_GRU-D Precision/Recall/F1 at F1-optimal threshold._  
_Transformer/GAT Precision/Recall/F1 at threshold=0.5._  
