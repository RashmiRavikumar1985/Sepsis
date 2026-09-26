"""Rebuild Graph F only using cached Spearman/MI from full run."""
import os, sys, json
import numpy as np
import pandas as pd

sys.path.insert(0, ".")

import importlib.util
spec = importlib.util.spec_from_file_location(
    "graph_builder_v2", "experiments/gat/graph_builder_v2.py"
)
gb = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gb)

SEPSIS_ROOT  = os.path.abspath(".")
PROJECT_ROOT = os.path.dirname(SEPSIS_ROOT)

with open("artifacts/preprocessing_config.json") as f:
    prep_cfg = json.load(f)
with open("artifacts/splits.json") as f:
    splits = json.load(f)

all_features = prep_cfg["dynamic_features"]
train_ids    = splits["train"]
CAND_1 = [
    os.path.join(PROJECT_ROOT, "physionet2019", "training", "training_setA"),
    os.path.join(PROJECT_ROOT, "physionet2019", "training", "training_setB"),
]
CAND_2 = [
    os.path.join(PROJECT_ROOT, "training", "training_setA"),
    os.path.join(PROJECT_ROOT, "training", "training_setB"),
]
data_dirs = CAND_1 if os.path.exists(CAND_1[0]) else CAND_2

print("Loading all training patients...")
df = gb.load_data(all_features, train_ids, data_dirs)
print(f"  Total timesteps: {len(df):,}")

corr_features = [f for f in all_features if f != "ICULOS"]
iculos_idx    = all_features.index("ICULOS")
n_all         = len(all_features)

print("Computing Spearman matrix...")
rho_corr = gb.compute_spearman_matrix(df, corr_features)
print(f"  Spearman range: [{rho_corr.min():.4f}, {rho_corr.max():.4f}]")

print("Computing MI matrix...")
mi_corr = gb.compute_mi_matrix(df, corr_features, n_bins=10)

# Expand to 35x35
rho_full = np.zeros((n_all, n_all))
mi_full  = np.zeros((n_all, n_all))
corr_idx = [all_features.index(f) for f in corr_features]
for ii, ci in enumerate(corr_idx):
    for jj, cj in enumerate(corr_idx):
        rho_full[ci, cj] = rho_corr[ii, jj]
        mi_full[ci, cj]  = mi_corr[ii, jj]

# Clinical prior set
clinical_prior_set = set()
for a, b in gb.CLINICAL_PRIOR_EDGES:
    if a in all_features and b in all_features:
        clinical_prior_set.add((a, b))
        clinical_prior_set.add((b, a))

# Clinical weight
strong_vals  = rho_full[rho_full >= 0.5]
clinical_w   = float(strong_vals.mean()) if len(strong_vals) > 0 else 0.5
print(f"Clinical prior weight: {clinical_w:.4f}")
print(f"Clinical prior pairs : {len(clinical_prior_set)//2}")

# Graph F mask: Spearman >= 0.30
mask_F = rho_full >= 0.30
mask_F[iculos_idx, :] = False
mask_F[:, iculos_idx] = False
np.fill_diagonal(mask_F, False)

nodes_df = pd.DataFrame([
    {"node_id": i, "feature_name": f}
    for i, f in enumerate(all_features)
])

edges_df = gb.build_edges(
    features          = all_features,
    rho               = rho_full,
    mi                = mi_full,
    clinical_prior_set= clinical_prior_set,
    mask              = mask_F,
    include_mi        = False,
    include_clinical  = True,
    clinical_weight   = clinical_w,
)

out_dir = "experiments/gat/graphs/graph_F"
summary = gb.save_graph(edges_df, nodes_df, out_dir, "F",
                        "Spearman |rho| >= 0.30 + clinical priors (v2)")

print(f"\n=== Graph F (rebuilt) ===")
print(f"  Unique edges     : {summary['n_edges_unique']}")
print(f"  Statistical      : {summary['n_statistical_edges']}")
print(f"  Clinical prior   : {summary['n_clinical_edges']}")
print(f"  Weight range     : [{summary['weight_min']:.4f}, {summary['weight_max']:.4f}]")

# Verify no isolated nodes
connected = set(edges_df.source.tolist() + edges_df.target.tolist())
isolated  = set(range(n_all)) - connected
print(f"  Isolated nodes   : {len(isolated)}")
for n in sorted(isolated):
    print(f"    [{n}] {all_features[n]}")

if len(isolated) == 0:
    print("  All 35 nodes connected!")
