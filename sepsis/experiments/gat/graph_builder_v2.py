"""
graph_builder_v2.py — GAT-v2 clinical feature graph construction.

Builds 7 candidate graphs using:
  - Spearman rank correlation (all training patients, pairwise complete)
  - Mutual Information (supplementary signal)
  - Clinical physiological priors

Graph variants:
  A: |ρ| >= 0.20  (matches current baseline for comparison)
  B: |ρ| >= 0.30
  C: |ρ| >= 0.35
  D: adaptive top-k per node (k=5)
  E: Spearman + MI
  F: Spearman (B) + clinical priors
  G: Spearman (B) + MI + clinical priors

Edge schema (provenance-aware):
  {source, target, spearman, mi, clinical_prior, weight}

ICULOS: kept as input node but excluded from correlation-derived edges.

Saves per graph:
  graphs/graph_X/edges.csv   — edge list with provenance
  graphs/graph_X/nodes.csv   — node list
  graphs/graph_X/summary.json — stats

Usage:
    python experiments/gat/graph_builder_v2.py
"""

import os
import sys
import json
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import mutual_info_score
from sklearn.preprocessing import KBinsDiscretizer

# ── Paths ──────────────────────────────────────────────────────────────
_HERE        = os.path.dirname(os.path.abspath(__file__))
SEPSIS_ROOT  = os.path.dirname(os.path.dirname(_HERE))
PROJECT_ROOT = os.path.dirname(SEPSIS_ROOT)
GRAPHS_DIR   = os.path.join(_HERE, "graphs")

PREPROCESS_CFG = os.path.join(SEPSIS_ROOT, "artifacts", "preprocessing_config.json")
SPLITS_PATH    = os.path.join(SEPSIS_ROOT, "artifacts", "splits.json")

# Data path candidates
CAND_1 = [
    os.path.join(PROJECT_ROOT, "physionet2019", "training", "training_setA"),
    os.path.join(PROJECT_ROOT, "physionet2019", "training", "training_setB"),
]
CAND_2 = [
    os.path.join(PROJECT_ROOT, "training", "training_setA"),
    os.path.join(PROJECT_ROOT, "training", "training_setB"),
]

# ── Clinical physiological prior edges ────────────────────────────────
# Source: established physiology, NOT Sepsis-3 definition.
# These are relationships where one variable directly affects another
# through known physiological mechanisms.
CLINICAL_PRIOR_EDGES = [
    # Hemodynamic coupling
    ("MAP",  "SBP"),
    ("MAP",  "DBP"),
    ("HR",   "MAP"),
    ("SBP",  "DBP"),
    ("HR",   "Resp"),

    # Respiratory chain
    ("Resp", "O2Sat"),
    ("Resp", "FiO2"),
    ("Resp", "PaCO2"),
    ("O2Sat","SaO2"),
    ("SaO2", "FiO2"),
    ("FiO2", "PaCO2"),
    ("EtCO2","PaCO2"),

    # Acid-base balance
    ("pH",        "HCO3"),
    ("pH",        "PaCO2"),
    ("pH",        "BaseExcess"),
    ("HCO3",      "BaseExcess"),
    ("PaCO2",     "BaseExcess"),
    ("Lactate",   "pH"),
    ("Lactate",   "HCO3"),

    # Temperature — systemic inflammation
    ("Temp",      "HR"),
    ("Temp",      "Resp"),
    ("Temp",      "WBC"),
    ("Temp",      "Lactate"),

    # Electrolytes / metabolic
    ("Calcium",   "Phosphate"),
    ("Calcium",   "Magnesium"),
    ("Calcium",   "Chloride"),
    ("Magnesium", "Potassium"),
    ("Potassium", "Chloride"),
    ("Potassium", "pH"),
    ("Glucose",   "Lactate"),
    ("Glucose",   "Potassium"),
    ("Glucose",   "pH"),

    # Renal function
    ("Creatinine","BUN"),
    ("Creatinine","Potassium"),
    ("Creatinine","Phosphate"),

    # Liver / bilirubin
    ("AST",              "Bilirubin_direct"),
    ("AST",              "Bilirubin_total"),
    ("Bilirubin_direct", "Bilirubin_total"),
    ("AST",              "Alkalinephos"),
    ("Alkalinephos",     "Bilirubin_total"),

    # Hematology / oxygen delivery
    ("Hct",      "Hgb"),
    ("Hgb",      "Lactate"),
    ("WBC",      "Platelets"),
    ("WBC",      "Fibrinogen"),
    ("WBC",      "Lactate"),
    ("Platelets","Fibrinogen"),
    ("TroponinI","HR"),
    ("TroponinI","Lactate"),
    ("PTT",      "Fibrinogen"),
]


def load_data(features: list, train_ids: list, data_dirs: list) -> pd.DataFrame:
    """Load all training patients and return concatenated DataFrame of raw values."""
    file_to_dir = {}
    for d in data_dirs:
        if os.path.exists(d):
            for f in os.listdir(d):
                if f.endswith(".psv"):
                    file_to_dir[f] = d

    all_dfs = []
    missing = 0
    for pid in train_ids:
        if pid not in file_to_dir:
            missing += 1
            continue
        path = os.path.join(file_to_dir[pid], pid)
        try:
            df = pd.read_csv(path, sep="|")[features]
            all_dfs.append(df)
        except Exception:
            missing += 1

    print(f"  Loaded {len(all_dfs):,} patients ({missing} missing)")
    return pd.concat(all_dfs, ignore_index=True)


def compute_spearman_matrix(df: pd.DataFrame, features: list) -> np.ndarray:
    """
    Compute pairwise Spearman |ρ| using only co-observed (both non-NaN) timesteps.
    Returns (N, N) matrix of absolute Spearman correlations.
    """
    n = len(features)
    rho = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            xi = df[features[i]].values
            xj = df[features[j]].values
            # Pairwise complete: both must be observed
            valid = ~(np.isnan(xi) | np.isnan(xj))
            if valid.sum() < 30:  # need at least 30 co-observed points
                rho[i, j] = 0.0
            else:
                r, _ = stats.spearmanr(xi[valid], xj[valid])
                rho[i, j] = abs(r) if not np.isnan(r) else 0.0
            rho[j, i] = rho[i, j]
    return rho


def compute_mi_matrix(df: pd.DataFrame, features: list, n_bins: int = 10) -> np.ndarray:
    """
    Compute pairwise Mutual Information using equal-frequency binning.
    Returns (N, N) matrix normalized to [0, 1].
    """
    n = len(features)
    mi = np.zeros((n, n))

    # Discretize each feature using equal-frequency bins on non-NaN values
    disc_data = {}
    for feat in features:
        vals = df[feat].dropna().values.reshape(-1, 1)
        if len(vals) < 30:
            disc_data[feat] = None
            continue
        kbd = KBinsDiscretizer(n_bins=n_bins, encode="ordinal", strategy="quantile")
        disc_data[feat] = kbd.fit_transform(vals).ravel()

    for i in range(n):
        for j in range(i + 1, n):
            xi_full = df[features[i]].values
            xj_full = df[features[j]].values
            # Co-observed mask
            valid = ~(np.isnan(xi_full) | np.isnan(xj_full))
            if valid.sum() < 30:
                mi[i, j] = 0.0
            else:
                # Bin co-observed values
                xi_v = xi_full[valid].reshape(-1, 1)
                xj_v = xj_full[valid].reshape(-1, 1)
                kbd_i = KBinsDiscretizer(n_bins=n_bins, encode="ordinal", strategy="quantile")
                kbd_j = KBinsDiscretizer(n_bins=n_bins, encode="ordinal", strategy="quantile")
                xi_d = kbd_i.fit_transform(xi_v).ravel().astype(int)
                xj_d = kbd_j.fit_transform(xj_v).ravel().astype(int)
                mi_val = mutual_info_score(xi_d, xj_d)
                mi[i, j] = mi_val
            mi[j, i] = mi[i, j]

    # Normalize MI to [0, 1]
    max_mi = mi.max()
    if max_mi > 0:
        mi = mi / max_mi
    return mi


def build_edges(
    features: list,
    rho: np.ndarray,
    mi: np.ndarray,
    clinical_prior_set: set,
    mask: np.ndarray,       # boolean (N, N) — True = include edge
    include_mi: bool = False,
    include_clinical: bool = False,
    clinical_weight: float = None,
) -> pd.DataFrame:
    """
    Build edge DataFrame from mask with provenance columns.

    clinical_weight: if None, uses mean of strong statistical edges (rho >= 0.5)
    """
    n = len(features)
    feat_idx = {f: i for i, f in enumerate(features)}

    # Compute clinical prior weight from data
    if clinical_weight is None:
        strong = rho[rho >= 0.5]
        clinical_weight = float(strong.mean()) if len(strong) > 0 else 0.5

    rows = []
    seen = set()

    for i in range(n):
        for j in range(i + 1, n):
            fi, fj = features[i], features[j]
            is_clinical = (fi, fj) in clinical_prior_set or (fj, fi) in clinical_prior_set
            stat_include = bool(mask[i, j])

            if not stat_include and not (include_clinical and is_clinical):
                continue

            key = (min(i, j), max(i, j))
            if key in seen:
                continue
            seen.add(key)

            spearman_val = float(rho[i, j])
            mi_val       = float(mi[i, j]) if include_mi else 0.0
            clinical_val = 1 if is_clinical else 0

            # Weight: combined normalized
            if is_clinical and include_clinical:
                # Blend: statistical weight if available, clinical_weight if not
                stat_w = spearman_val if stat_include else 0.0
                w = max(stat_w, clinical_weight)
            else:
                w = spearman_val

            # Add both directions (undirected → bidirectional for PyG)
            for src, tgt in [(i, j), (j, i)]:
                rows.append({
                    "source"        : src,
                    "target"        : tgt,
                    "spearman"      : spearman_val,
                    "mi"            : mi_val,
                    "clinical_prior": clinical_val,
                    "weight"        : w,
                })

    return pd.DataFrame(rows)


def save_graph(edges_df: pd.DataFrame, nodes_df: pd.DataFrame,
               out_dir: str, graph_id: str, description: str) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    edges_df.to_csv(os.path.join(out_dir, "edges.csv"), index=False)
    nodes_df.to_csv(os.path.join(out_dir, "nodes.csv"), index=False)

    unique_edges = len(edges_df) // 2
    summary = {
        "graph_id"          : graph_id,
        "description"       : description,
        "n_nodes"           : len(nodes_df),
        "n_edges_unique"    : unique_edges,
        "n_edges_total"     : len(edges_df),
        "weight_mean"       : float(edges_df["weight"].mean()),
        "weight_min"        : float(edges_df["weight"].min()),
        "weight_max"        : float(edges_df["weight"].max()),
        "n_clinical_edges"  : int((edges_df["clinical_prior"] == 1).sum()) // 2,
        "n_statistical_edges": int((edges_df["clinical_prior"] == 0).sum()) // 2,
    }
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=4)
    return summary


def main():
    print("=" * 65)
    print("GAT-v2 Graph Builder")
    print("=" * 65)

    # ── Load config ───────────────────────────────────────────────────
    with open(PREPROCESS_CFG) as f:
        prep_cfg = json.load(f)
    with open(SPLITS_PATH) as f:
        splits = json.load(f)

    all_features = prep_cfg["dynamic_features"]   # 35 features including ICULOS
    train_ids    = splits["train"]
    data_dirs    = CAND_1 if os.path.exists(CAND_1[0]) else CAND_2

    print(f"Features     : {len(all_features)} (including ICULOS)")
    print(f"Train patients: {len(train_ids):,}")
    print(f"Data dirs    : {data_dirs[0]}")

    # ── Load all training data ────────────────────────────────────────
    print("\nLoading all training patients...")
    df = load_data(all_features, train_ids, data_dirs)
    print(f"  Total timesteps: {len(df):,}")

    # ── ICULOS exclusion from correlation ─────────────────────────────
    # Keep ICULOS as a node but exclude from correlation computation
    corr_features = [f for f in all_features if f != "ICULOS"]
    iculos_idx    = all_features.index("ICULOS")
    n_corr        = len(corr_features)
    print(f"\nCorrelation features: {n_corr} (ICULOS excluded from edges)")

    # ── Compute Spearman matrix ───────────────────────────────────────
    print("\nComputing Spearman correlation matrix (pairwise complete)...")
    rho_corr = compute_spearman_matrix(df, corr_features)
    print(f"  Spearman range: [{rho_corr.min():.4f}, {rho_corr.max():.4f}]")

    # Expand to full 35x35 matrix (ICULOS row/col = 0)
    n_all = len(all_features)
    rho_full = np.zeros((n_all, n_all))
    corr_idx = [all_features.index(f) for f in corr_features]
    for ii, ci in enumerate(corr_idx):
        for jj, cj in enumerate(corr_idx):
            rho_full[ci, cj] = rho_corr[ii, jj]

    # ── Compute MI matrix ─────────────────────────────────────────────
    print("\nComputing Mutual Information matrix...")
    mi_corr = compute_mi_matrix(df, corr_features, n_bins=10)
    print(f"  MI range (normalized): [{mi_corr.min():.4f}, {mi_corr.max():.4f}]")

    mi_full = np.zeros((n_all, n_all))
    for ii, ci in enumerate(corr_idx):
        for jj, cj in enumerate(corr_idx):
            mi_full[ci, cj] = mi_corr[ii, jj]

    # ── Clinical prior set (feature name pairs) ───────────────────────
    clinical_prior_set = set()
    for a, b in CLINICAL_PRIOR_EDGES:
        if a in all_features and b in all_features:
            clinical_prior_set.add((a, b))
            clinical_prior_set.add((b, a))

    feat_idx = {f: i for i, f in enumerate(all_features)}
    print(f"\nClinical prior pairs: {len(clinical_prior_set)//2}")

    # ── Node DataFrame ────────────────────────────────────────────────
    nodes_df = pd.DataFrame([
        {"node_id": i, "feature_name": f}
        for i, f in enumerate(all_features)
    ])

    # ── Clinical prior weight (mean of strong statistical edges) ──────
    strong_vals = rho_full[rho_full >= 0.5]
    clinical_w  = float(strong_vals.mean()) if len(strong_vals) > 0 else 0.5
    print(f"Clinical prior weight (mean strong rho): {clinical_w:.4f}")

    # ── Graph D: adaptive top-k (k=5) ─────────────────────────────────
    # For each node, keep its top-5 strongest Spearman connections
    k = 5
    topk_mask = np.zeros((n_all, n_all), dtype=bool)
    for i in range(n_all):
        if i == iculos_idx:
            continue
        row = rho_full[i].copy()
        row[i] = 0.0  # exclude self
        row[iculos_idx] = 0.0  # exclude ICULOS
        top_indices = np.argsort(row)[-k:]
        topk_mask[i, top_indices] = True
    # Make symmetric
    topk_mask = topk_mask | topk_mask.T

    # ── MI threshold: top 30% of non-zero MI values ───────────────────
    mi_nonzero = mi_full[mi_full > 0]
    mi_threshold = float(np.percentile(mi_nonzero, 70)) if len(mi_nonzero) > 0 else 0.1
    print(f"MI threshold (top 30%): {mi_threshold:.4f}")

    # ── Build all 7 graphs ─────────────────────────────────────────────
    GRAPHS = [
        {
            "id"         : "A",
            "desc"       : "|ρ| >= 0.20 — current baseline (Spearman)",
            "mask"       : rho_full >= 0.20,
            "inc_mi"     : False,
            "inc_clin"   : False,
        },
        {
            "id"         : "B",
            "desc"       : "|ρ| >= 0.30 — stricter threshold (Spearman)",
            "mask"       : rho_full >= 0.30,
            "inc_mi"     : False,
            "inc_clin"   : False,
        },
        {
            "id"         : "C",
            "desc"       : "|ρ| >= 0.35 — aggressive pruning (Spearman)",
            "mask"       : rho_full >= 0.35,
            "inc_mi"     : False,
            "inc_clin"   : False,
        },
        {
            "id"         : "D",
            "desc"       : "Adaptive top-k (k=5 per node, Spearman)",
            "mask"       : topk_mask,
            "inc_mi"     : False,
            "inc_clin"   : False,
        },
        {
            "id"         : "E",
            "desc"       : "Spearman |ρ| >= 0.30 + MI (top 30%)",
            "mask"       : (rho_full >= 0.30) | (mi_full >= mi_threshold),
            "inc_mi"     : True,
            "inc_clin"   : False,
        },
        {
            "id"         : "F",
            "desc"       : "Spearman |ρ| >= 0.30 + clinical priors",
            "mask"       : rho_full >= 0.30,
            "inc_mi"     : False,
            "inc_clin"   : True,
        },
        {
            "id"         : "G",
            "desc"       : "Spearman |ρ| >= 0.30 + MI + clinical priors",
            "mask"       : (rho_full >= 0.30) | (mi_full >= mi_threshold),
            "inc_mi"     : True,
            "inc_clin"   : True,
        },
    ]

    # Remove ICULOS from all masks
    for g in GRAPHS:
        g["mask"][iculos_idx, :] = False
        g["mask"][:, iculos_idx] = False
        # Remove self-loops
        np.fill_diagonal(g["mask"], False)

    print("\n" + "=" * 65)
    print("Building graphs...")
    print("=" * 65)

    all_summaries = []
    for g in GRAPHS:
        edges_df = build_edges(
            features        = all_features,
            rho             = rho_full,
            mi              = mi_full,
            clinical_prior_set = clinical_prior_set,
            mask            = g["mask"],
            include_mi      = g["inc_mi"],
            include_clinical= g["inc_clin"],
            clinical_weight = clinical_w,
        )
        out_dir = os.path.join(GRAPHS_DIR, f"graph_{g['id']}")
        summary = save_graph(edges_df, nodes_df, out_dir, g["id"], g["desc"])
        all_summaries.append(summary)

        print(f"\n  Graph {g['id']}: {g['desc']}")
        print(f"    Unique edges   : {summary['n_edges_unique']}")
        print(f"    Statistical    : {summary['n_statistical_edges']}")
        print(f"    Clinical prior : {summary['n_clinical_edges']}")
        print(f"    Weight range   : [{summary['weight_min']:.4f}, {summary['weight_max']:.4f}]")
        print(f"    Saved → {out_dir}")

    # ── Master summary ─────────────────────────────────────────────────
    master_path = os.path.join(GRAPHS_DIR, "graph_summary.json")
    with open(master_path, "w") as f:
        json.dump(all_summaries, f, indent=4)

    print("\n" + "=" * 65)
    print("SUMMARY TABLE")
    print("=" * 65)
    print(f"{'Graph':<8} {'Description':<45} {'Edges':>6} {'Clin':>5}")
    print("-" * 65)
    for s in all_summaries:
        print(f"  {s['graph_id']:<6} {s['description'][:44]:<45} {s['n_edges_unique']:>6} {s['n_clinical_edges']:>5}")

    print(f"\nMaster summary → {master_path}")
    print("Done.")


if __name__ == "__main__":
    main()
