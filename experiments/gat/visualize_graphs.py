"""
visualize_graphs.py — Visualize candidate GAT graphs A, B, F side by side.

Saves to experiments/gat/plots/graph_analysis/
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx

_HERE   = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(_HERE, "plots", "graph_analysis")
os.makedirs(OUT_DIR, exist_ok=True)

# ── Feature categories ────────────────────────────────────────────────
CATEGORIES = {
    "Vitals"    : ["HR", "O2Sat", "Temp", "SBP", "MAP", "DBP", "Resp", "EtCO2"],
    "ABG"       : ["BaseExcess", "HCO3", "FiO2", "pH", "PaCO2", "SaO2"],
    "Metabolic" : ["Glucose", "Lactate", "Calcium", "Magnesium", "Phosphate",
                   "Potassium", "Chloride", "BUN", "Creatinine"],
    "Liver"     : ["AST", "Alkalinephos", "Bilirubin_direct", "Bilirubin_total"],
    "Hematology": ["Hct", "Hgb", "PTT", "WBC", "Fibrinogen", "Platelets", "TroponinI"],
    "Time"      : ["ICULOS"],
}
CAT_COLORS = {
    "Vitals"    : "#E74C3C",
    "ABG"       : "#3498DB",
    "Metabolic" : "#2ECC71",
    "Liver"     : "#F39C12",
    "Hematology": "#9B59B6",
    "Time"      : "#95A5A6",
}
def get_color(feat):
    for cat, feats in CATEGORIES.items():
        if feat in feats:
            return CAT_COLORS[cat]
    return "#CCCCCC"

def load_graph(graph_id):
    graph_dir = os.path.join(_HERE, "graphs", f"graph_{graph_id}")
    edges = pd.read_csv(os.path.join(graph_dir, "edges.csv"))
    nodes = pd.read_csv(os.path.join(graph_dir, "nodes.csv"))
    with open(os.path.join(graph_dir, "summary.json")) as f:
        summary = json.load(f)
    return edges, nodes, summary

def build_nx(edges, nodes):
    G = nx.Graph()
    feat_names = nodes["feature_name"].tolist()
    for _, row in nodes.iterrows():
        G.add_node(int(row["node_id"]), label=row["feature_name"])
    seen = set()
    for _, row in edges.iterrows():
        u, v = int(row["source"]), int(row["target"])
        key = (min(u,v), max(u,v))
        if key not in seen:
            G.add_edge(u, v,
                       weight=float(row["weight"]),
                       clinical=int(row["clinical_prior"]))
            seen.add(key)
    return G, feat_names

# ── Load graphs ───────────────────────────────────────────────────────
graphs_to_show = ["A", "B", "F"]
data = {}
for gid in graphs_to_show:
    edges, nodes, summary = load_graph(gid)
    G, feat_names = build_nx(edges, nodes)
    data[gid] = {"G": G, "feat_names": feat_names,
                 "edges": edges, "summary": summary}

# Use a consistent spring layout seed across all graphs
# Compute shared layout from the densest graph (A)
pos_base = nx.spring_layout(data["A"]["G"], seed=42, k=2.8)
# For B and F, use same positions (subset of nodes)
pos = {gid: nx.spring_layout(data[gid]["G"], seed=42, k=2.8)
       for gid in graphs_to_show}

# ── PLOT 1: Side-by-side comparison ───────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(24, 10))
fig.suptitle("GAT Candidate Graph Comparison\n"
             "Red edges = clinical priors | Gray edges = statistical only | "
             "Edge thickness = correlation strength",
             fontsize=13, fontweight="bold")

for ax, gid in zip(axes, graphs_to_show):
    G         = data[gid]["G"]
    feat_names = data[gid]["feat_names"]
    summary   = data[gid]["summary"]
    p         = pos[gid]

    node_colors = [get_color(feat_names[n]) for n in G.nodes()]

    # Draw edges — clinical in red, statistical in gray
    stat_edges    = [(u,v) for u,v,d in G.edges(data=True) if d["clinical"] == 0]
    clinical_edges= [(u,v) for u,v,d in G.edges(data=True) if d["clinical"] == 1]
    stat_w        = [G[u][v]["weight"] for u,v in stat_edges]
    clin_w        = [G[u][v]["weight"] for u,v in clinical_edges]

    nx.draw_networkx_edges(G, p, edgelist=stat_edges,
                           width=[0.5 + 2.5*w for w in stat_w],
                           edge_color="#AAAAAA", alpha=0.6, ax=ax)
    nx.draw_networkx_edges(G, p, edgelist=clinical_edges,
                           width=[1.0 + 3.0*w for w in clin_w],
                           edge_color="#E74C3C", alpha=0.8, ax=ax)
    nx.draw_networkx_nodes(G, p, node_color=node_colors,
                           node_size=600, ax=ax)
    nx.draw_networkx_labels(G, p,
                            labels={n: feat_names[n] for n in G.nodes()},
                            font_size=6, font_weight="bold", ax=ax)

    ax.set_title(
        f"Graph {gid}: {summary['description']}\n"
        f"{summary['n_edges_unique']} edges "
        f"({summary['n_statistical_edges']} stat + "
        f"{summary['n_clinical_edges']} clinical)",
        fontsize=9, fontweight="bold"
    )
    ax.axis("off")

# Category legend
legend_patches = [
    plt.Line2D([0],[0], marker='o', color='w', markerfacecolor=c,
               markersize=10, label=cat)
    for cat, c in CAT_COLORS.items()
]
legend_patches += [
    plt.Line2D([0],[0], color='#E74C3C', lw=2, label='Clinical prior edge'),
    plt.Line2D([0],[0], color='#AAAAAA', lw=2, label='Statistical edge'),
]
fig.legend(handles=legend_patches, loc="lower center",
           ncol=8, fontsize=9, framealpha=0.9,
           bbox_to_anchor=(0.5, -0.02))

plt.tight_layout(rect=[0, 0.04, 1, 1])
p1 = os.path.join(OUT_DIR, "graph_ABF_comparison.png")
plt.savefig(p1, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved → {p1}")

# ── PLOT 2: Graph F detailed — full size ──────────────────────────────
G_F = data["F"]["G"]
fn_F = data["F"]["feat_names"]
pos_F = nx.spring_layout(G_F, seed=42, k=3.2)

fig, ax = plt.subplots(figsize=(16, 13))

stat_edges_F    = [(u,v) for u,v,d in G_F.edges(data=True) if d["clinical"] == 0]
clinical_edges_F= [(u,v) for u,v,d in G_F.edges(data=True) if d["clinical"] == 1]
stat_w_F        = [G_F[u][v]["weight"] for u,v in stat_edges_F]
clin_w_F        = [G_F[u][v]["weight"] for u,v in clinical_edges_F]

nx.draw_networkx_edges(G_F, pos_F, edgelist=stat_edges_F,
                       width=[0.8 + 3.0*w for w in stat_w_F],
                       edge_color="#999999", alpha=0.6, ax=ax)
nx.draw_networkx_edges(G_F, pos_F, edgelist=clinical_edges_F,
                       width=[1.5 + 3.5*w for w in clin_w_F],
                       edge_color="#E74C3C", alpha=0.85, ax=ax)

# Label edge weights on clinical edges
for u, v in clinical_edges_F:
    x = (pos_F[u][0] + pos_F[v][0]) / 2
    y = (pos_F[u][1] + pos_F[v][1]) / 2
    w = G_F[u][v]["weight"]
    ax.text(x, y, f"{w:.2f}", fontsize=5.5, color="#C0392B",
            ha="center", va="center")

node_colors_F = [get_color(fn_F[n]) for n in G_F.nodes()]
nx.draw_networkx_nodes(G_F, pos_F, node_color=node_colors_F,
                       node_size=900, ax=ax)
nx.draw_networkx_labels(G_F, pos_F,
                        labels={n: fn_F[n] for n in G_F.nodes()},
                        font_size=7.5, font_weight="bold", ax=ax)

summary_F = data["F"]["summary"]
ax.set_title(
    f"Graph F — {summary_F['description']}\n"
    f"{summary_F['n_edges_unique']} edges "
    f"({summary_F['n_statistical_edges']} statistical + "
    f"{summary_F['n_clinical_edges']} clinical prior)\n"
    f"Red edges = physiological priors | Labels = Spearman weight",
    fontsize=12, fontweight="bold"
)
ax.axis("off")

cat_patches = [
    plt.Line2D([0],[0], marker='o', color='w', markerfacecolor=c,
               markersize=11, label=cat)
    for cat, c in CAT_COLORS.items()
]
cat_patches += [
    plt.Line2D([0],[0], color='#E74C3C', lw=2.5, label='Clinical prior edge'),
    plt.Line2D([0],[0], color='#999999', lw=2, label='Statistical edge (|ρ|≥0.30)'),
]
ax.legend(handles=cat_patches, loc="upper left", fontsize=9, framealpha=0.9)

plt.tight_layout()
p2 = os.path.join(OUT_DIR, "graph_F_detailed.png")
plt.savefig(p2, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved → {p2}")

# ── PLOT 3: Edge weight heatmap for all 3 ────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(21, 7))
feat_names_all = data["A"]["feat_names"]
n = len(feat_names_all)

for ax, gid in zip(axes, graphs_to_show):
    edges_df = data[gid]["edges"]
    adj = np.zeros((n, n))
    for _, row in edges_df.iterrows():
        u, v = int(row["source"]), int(row["target"])
        # Color clinical edges differently
        if row["clinical_prior"] == 1:
            adj[u][v] = row["weight"] + 1.0  # offset to distinguish
        else:
            adj[u][v] = row["weight"]

    # Use two-part colormap: gray for stat, red for clinical
    im = ax.imshow(adj, cmap="YlOrRd", vmin=0, vmax=2.0, aspect="auto")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(feat_names_all, rotation=90, fontsize=5)
    ax.set_yticklabels(feat_names_all, fontsize=5)
    summary = data[gid]["summary"]
    ax.set_title(f"Graph {gid}\n{summary['n_edges_unique']} edges",
                 fontsize=10, fontweight="bold")

plt.colorbar(im, ax=axes[-1], label="Weight (>1.0 = clinical prior)")
plt.suptitle("Adjacency Matrix Comparison — Graphs A, B, F",
             fontsize=13, fontweight="bold")
plt.tight_layout()
p3 = os.path.join(OUT_DIR, "graph_ABF_adjacency.png")
plt.savefig(p3, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved → {p3}")

print(f"\nAll plots saved to:\n  {OUT_DIR}")
