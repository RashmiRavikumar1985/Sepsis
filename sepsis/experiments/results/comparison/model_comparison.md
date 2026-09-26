# Experimental Model Comparison

| Model                       |   Test AUPRC |   Test AUROC |   Test Precision |   Test Recall |   Test F1 |
|:----------------------------|-------------:|-------------:|-----------------:|--------------:|----------:|
| GRU-D                       |       0.1307 |       0.8464 |           0.1788 |        0.2351 |    0.2031 |
| Transformer (baseline d=64) |       0.3649 |       0.9463 |           0.0928 |        0.7200 |    0.1644 |
| Transformer (tuned d=128)   |       0.1120 |       0.8305 |         nan      |      nan      |  nan      |
| GAT (Graph Baseline)        |       0.0905 |       0.7815 |           0.1336 |        0.3007 |    0.1850 |

_GRU-D Precision/Recall/F1 computed at F1-optimal threshold._  
_Transformer Precision/Recall/F1 computed at threshold=0.5._  
_GAT Precision/Recall/F1 computed at threshold=0.5._  
