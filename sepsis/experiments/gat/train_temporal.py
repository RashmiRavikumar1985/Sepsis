"""
train_temporal.py  -  GAT-2 (Temporal GAT) Training  [DENSE / FAST]

Usage:
    python experiments/gat/train_temporal.py                         # default (72h)
    python experiments/gat/train_temporal.py --device cpu
    python experiments/gat/train_temporal.py --device mps
    python experiments/gat/train_temporal.py --seq-len 120           # try 120h, isolated dir
    python experiments/gat/train_temporal.py --seq-len 120 --run-name gat2_120h

--run-name controls the output directory so existing results are NEVER overwritten:
    results dir : experiments/results/<run-name>/
    plots dir   : experiments/gat/plots_<run-name>/

Speed design:
  - Dense 35x35 clinical GAT (matmul, no scatter) — fast on ANY device
  - Causal temporal mixing — simple gated shift, no scatter
  - No edge_index needed at forward time (adjacency baked into model)
  - num_workers=0 on MPS (multiprocessing unsupported), 2 on CPU/CUDA
  - pin_memory only on CUDA
"""

import os, sys, json, argparse, torch, torch.nn as nn, numpy as np, random
from torch.utils.data import DataLoader
from sklearn.metrics import average_precision_score, roc_auc_score
from tqdm import tqdm

try:
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt; HAS_MPL=True
except Exception: HAS_MPL=False

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from src.dataset_grud import PhysioNetDatasetGRUD, collate_fn
from experiments.gat.temporal_graph_builder import load_clinical_edges
from experiments.gat.temporal_model import TemporalGAT

GAT2_CONFIG = {
    "hidden_dim": 64, "out_dim": 64, "num_heads": 2, "num_gat_layers": 2,
    "dropout": 0.3, "batch_size": 32, "max_seq_len": 72,
    "epochs": 30, "patience": 5, "learning_rate": 0.001, "weight_decay": 1e-4,
    "seed": 42, "add_self_loops": True, "model": "TemporalGAT", "version": "GAT-2",
}


def set_seed(s=42):
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)
    np.random.seed(s); random.seed(s)
    torch.backends.cudnn.deterministic = True


def pick_device(user_choice: str) -> torch.device:
    if user_choice == "auto":
        if torch.cuda.is_available():   return torch.device("cuda")
        if torch.backends.mps.is_available(): return torch.device("mps")
        return torch.device("cpu")
    return torch.device(user_choice)


def find_data_dirs(project_root, sepsis_root):
    for dirs in [
        [os.path.join(project_root, "physionet2019","training","training_setA"),
         os.path.join(project_root, "physionet2019","training","training_setB")],
        [os.path.join(project_root,"training","training_setA"),
         os.path.join(project_root,"training","training_setB")],
        [os.path.join(sepsis_root,"training","training_setA"),
         os.path.join(sepsis_root,"training","training_setB")],
    ]:
        if os.path.exists(dirs[0]): return dirs
    raise FileNotFoundError("Cannot find training_setA / training_setB")


@torch.no_grad()
def evaluate_epoch(model, loader, criterion, device):
    model.eval()
    total_loss=total_valid=0; all_preds=[]; all_labels=[]

    for values,mask,delta,static_feat,labels,valid_mask in loader:
        values=values.to(device,non_blocking=True); mask=mask.to(device,non_blocking=True)
        delta=delta.to(device,non_blocking=True); static_feat=static_feat.to(device,non_blocking=True)
        labels=labels.to(device,non_blocking=True); valid_mask=valid_mask.to(device,non_blocking=True)

        logits = model(values,mask,delta,static_feat,valid_mask=valid_mask)
        loss_m = criterion(logits,labels)
        vf = valid_mask.float(); nv = int(vf.sum().item())
        loss = (loss_m*vf).sum()/vf.sum().clamp(min=1)
        total_loss+=loss.item()*nv; total_valid+=nv
        probs=torch.sigmoid(logits)
        all_preds.append(probs[valid_mask].cpu()); all_labels.append(labels[valid_mask].cpu())

    all_preds=torch.cat(all_preds).numpy(); all_labels=torch.cat(all_labels).numpy()
    avg_loss = total_loss/max(total_valid,1)
    if len(np.unique(all_labels))>1:
        auprc=average_precision_score(all_labels,all_preds)
        auroc=roc_auc_score(all_labels,all_preds)
    else: auprc=auroc=0.0
    return avg_loss, auroc, auprc


def train_temporal_gat(device_choice="auto", seq_len=None, run_name=None):
    print("="*65); print("  GAT-2: Temporal GAT — Training  [DENSE / FAST]"); print("="*65)

    sepsis_root   = project_root
    prep_cfg_path = os.path.join(sepsis_root,"artifacts","preprocessing_config.json")
    splits_path   = os.path.join(sepsis_root,"artifacts","splits.json")
    edges_csv     = os.path.join(sepsis_root,"experiments","gat","edges.csv")

    # run_name isolates results so existing runs are never overwritten
    _run = run_name if run_name else "gat_temporal"
    results_dir = os.path.join(sepsis_root,"experiments","results",_run)
    plots_dir   = os.path.join(sepsis_root,"experiments","gat",f"plots_{_run}")
    os.makedirs(results_dir,exist_ok=True); os.makedirs(plots_dir,exist_ok=True)
    print(f"Run name     : {_run}")
    print(f"Results dir  : {results_dir}")

    with open(prep_cfg_path) as f: prep_cfg=json.load(f)
    with open(splits_path)   as f: splits=json.load(f)

    F=len(prep_cfg["dynamic_features"]); S=len(prep_cfg["static_features"])
    assert F==35 and S==5 and "ICULOS" in prep_cfg["dynamic_features"]
    pos_weight=prep_cfg.get("class_weight",54.54)
    cfg=GAT2_CONFIG.copy()
    if seq_len is not None:
        cfg["max_seq_len"] = int(seq_len)   # CLI override
    max_seq_len=cfg["max_seq_len"]
    set_seed(cfg["seed"])

    device=pick_device(device_choice)
    # Device-specific settings
    use_workers = 0 if device.type=="mps" else 2   # MPS can't use multiprocessing
    use_pin     = device.type=="cuda"               # pin_memory only for CUDA
    print(f"Device       : {device}  (workers={use_workers}, pin_memory={use_pin})")
    print(f"max_seq_len  : {max_seq_len}h  → max {max_seq_len*35:,} nodes/patient")
    print(f"batch_size   : {cfg['batch_size']}")

    data_dirs=find_data_dirs(os.path.dirname(sepsis_root),sepsis_root)

    train_ds=PhysioNetDatasetGRUD(data_dirs,splits["train"],prep_cfg_path,max_seq_len)
    val_ds  =PhysioNetDatasetGRUD(data_dirs,splits["val"],  prep_cfg_path,max_seq_len)

    train_loader=DataLoader(train_ds,batch_size=cfg["batch_size"],shuffle=True,
                            collate_fn=collate_fn,num_workers=use_workers,pin_memory=use_pin)
    val_loader  =DataLoader(val_ds,  batch_size=cfg["batch_size"],shuffle=False,
                            collate_fn=collate_fn,num_workers=use_workers,pin_memory=use_pin)

    # Load clinical graph and bake into model
    clinical_src,clinical_dst=load_clinical_edges(edges_csv)
    print(f"Clinical edges per timestep: {len(clinical_src)}")

    model=TemporalGAT(num_features=F,static_size=S,
                      hidden_dim=cfg["hidden_dim"],out_dim=cfg["out_dim"],
                      num_heads=cfg["num_heads"],num_gat_layers=cfg["num_gat_layers"],
                      dropout=cfg["dropout"]).to(device)

    # KEY: bake adjacency into model (no edge_index needed at forward time)
    model.set_adjacency(clinical_src,clinical_dst,cfg["add_self_loops"])
    print(f"Clinical adjacency baked into model layers ✓")

    n_params=sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters   : {n_params:,}")

    criterion=nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight,device=device),reduction='none')
    optimizer=torch.optim.Adam(model.parameters(),lr=cfg["learning_rate"],weight_decay=cfg["weight_decay"])

    with open(os.path.join(results_dir,"run_config.json"),"w") as f:
        json.dump({**cfg,"num_features":F,"static_size":S,"pos_weight":pos_weight,
                   "n_params":n_params,"device":str(device)},f,indent=2)

    best_auprc=0.0; patience_counter=0
    train_losses=[]; val_losses=[]; val_aurocs=[]; val_auprcs=[]

    for epoch in range(cfg["epochs"]):
        model.train()
        epoch_loss=epoch_valid=0

        pbar=tqdm(train_loader,desc=f"Ep {epoch+1:02d}/{cfg['epochs']}",
                  leave=False,dynamic_ncols=True)
        for values,mask,delta,static_feat,labels,valid_mask in pbar:
            values=values.to(device,non_blocking=True)
            mask=mask.to(device,non_blocking=True)
            delta=delta.to(device,non_blocking=True)
            static_feat=static_feat.to(device,non_blocking=True)
            labels=labels.to(device,non_blocking=True)
            valid_mask=valid_mask.to(device,non_blocking=True)

            # No edge_index needed — adjacency is baked in
            optimizer.zero_grad()
            logits=model(values,mask,delta,static_feat,valid_mask=valid_mask)
            loss_m=criterion(logits,labels)
            vf=valid_mask.float()
            loss=(loss_m*vf).sum()/vf.sum().clamp(min=1)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
            optimizer.step()

            nv=int(vf.sum().item()); epoch_loss+=loss.item()*nv; epoch_valid+=nv
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_train_loss=epoch_loss/max(epoch_valid,1); train_losses.append(avg_train_loss)

        val_loss,val_auroc,val_auprc=evaluate_epoch(model,val_loader,criterion,device)
        val_losses.append(val_loss); val_aurocs.append(val_auroc); val_auprcs.append(val_auprc)

        flag=" ← best" if val_auprc>best_auprc else ""
        print(f"Ep {epoch+1:02d} | train={avg_train_loss:.4f} | val={val_loss:.4f}"
              f" | AUROC={val_auroc:.4f} | AUPRC={val_auprc:.4f}{flag}")

        if val_auprc>best_auprc:
            best_auprc=val_auprc
            torch.save(model.state_dict(),os.path.join(results_dir,"best_model.pt"))
            patience_counter=0
        else:
            patience_counter+=1
            if patience_counter>=cfg["patience"]:
                print(f"Early stopping at epoch {epoch+1}"); break

    if HAS_MPL:
        plt.figure(figsize=(10,4)); plt.plot(train_losses,label="Train"); plt.plot(val_losses,label="Val")
        plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title("GAT-2 Loss")
        plt.legend(); plt.tight_layout(); plt.savefig(os.path.join(plots_dir,"loss_curve.png")); plt.close()

        plt.figure(figsize=(10,4)); plt.plot(val_auprcs,label="AUPRC"); plt.plot(val_aurocs,label="AUROC")
        plt.xlabel("Epoch"); plt.ylabel("Score"); plt.title("GAT-2 Val Metrics")
        plt.legend(); plt.tight_layout(); plt.savefig(os.path.join(plots_dir,"val_metrics.png")); plt.close()

    print(f"\nBest Val AUPRC: {best_auprc:.4f}")
    print(f"Saved: {results_dir}/best_model.pt")
    return best_auprc


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--device", default="auto",
                        choices=["auto","cpu","mps","cuda"],
                        help="Device to train on (default: auto-detect best)")
    parser.add_argument("--seq-len", type=int, default=None,
                        help="Override max_seq_len (e.g. 120 or 168). Default: use config (72).")
    parser.add_argument("--run-name", type=str, default=None,
                        help="Name for this run's output directory. Default: gat_temporal.\n"
                             "E.g. --run-name gat2_120h  saves to experiments/results/gat2_120h/")
    args=parser.parse_args()
    train_temporal_gat(args.device, args.seq_len, args.run_name)
