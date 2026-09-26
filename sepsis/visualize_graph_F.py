"""
Visualize Graph F — full graph, degree analysis, adjacency heatmap.
Saves to experiments/gat/plots/graph_F/
"""
import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx

OUT_DIR = "experiments/gat/plots/graph_F"
os.makedirs(OUT_DIR, exist_ok=True)

edges = pd.read_csv("experiments/gat/graphs/graph_F/edges.csv")
nodes = pd.read_csv("experiments/gat/graphs/graph_F/nodes.csv")
feat_names = nodes["feature_name"].tolist()
n = len(feat_names)

# ── Categories ────────────────────────────────────────────────────────
CATEGORIES = {
    "Vitals"    : ["HR","O2Sat","Temp","SBP","MAP","DBP","Resp","EtCO2"],
    "ABG"       : ["BaseExcess","HCO3","FiO2","pH","PaCO2","SaO2"],
    "Metabolic" : ["Glucose","Lactate","Calcium","Magnesium","Phosphate",
                   "Potassium","Chloride","BUN","Creatinine"],
    "Liver"     : ["AST","Alkalinephos","Bilirubin_direct","Bilirubin_total"],
    "Hematology": ["Hct","Hgb","PTT","WBC","Fibrinogen","Platelets","TroponinI"],
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
def cat_color(feat):
    for cat, feats in CATEGORIES.items():
        if feat in feats:
            return CAT_COLORS[cat]
    return "#CCCCCC"

# ── Build NetworkX ────────────────────────────────────────────────────
G = nx.Graph()
for _, row in nodes.iterrows():
    G.add_node(int(row.node_id), label=row.feature_name)

seen = set()
for _, row in edges.iterrows():
    u, v = int(row.source), int(row.target)
    key  = (min(u,v), max(u,v))
    if key not in seen:
        G.add_edge(u, v, weight=float(row.weight),
                   clinical=int(row.clinical_prior),
                   spearman=float(row.spearman))
        seen.add(key)

stat_edges     = [(u,v) for u,v,d in G.edges(data=True) if d["clinical"]==0]
clinical_edges = [(u,v) for u,v,d in G.edges(data=True) if d["clinical"]==1]
node_colors    = [cat_color(feat_names[n]) for n in G.nodes()]

# ── Layout ────────────────────────────────────────────────────────────
pos = nx.spring_layout(G, seed=42, k=3.0, weight="weight")

# ── PLOT 1: Full graph ────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(18, 14))

nx.draw_networkx_edges(G, pos, edgelist=stat_edges,
    width=[1.0 + 3.0*G[u][v]["weight"] for u,v in stat_edges],
    edge_color="#AAAAAA", alpha=0.6, ax=ax)
nx.draw_networkx_edges(G, pos, edgelist=clinical_edges,
    width=[1.5 + 3.5*G[u][v]["weight"] for u,v in clinical_edges],
    edge_color="#E74C3C", alpha=0.85, ax=ax)

# Clinical edge weight labels
for u, v in clinical_edges:
    x = (pos[u][0]+pos[v][0])/2
    y = (pos[u][1]+pos[v][1])/2
    w = G[u][v]["spearman"]
    label = f"{w:.2f}" if w > 0 else "prior"
    ax.text(x, y, label, fontsize=5, color="#C0392B", ha="center", va="center")

nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=900, ax=ax)
nx.draw_networkx_labels(G, pos,
    labels={nd: feat_names[nd] for nd in G.nodes()},
    font_size=7.5, font_weight="bold", ax=ax)

legend_patches = [
    plt.Line2D([0],[0],marker='o',color='w',markerfacecolor=c,markersize=12,label=cat)
    for cat,c in CAT_COLORS.items()
] + [
    plt.Line2D([0],[0],color='#E74C3C',lw=2.5,label='Clinical prior edge'),
    plt.Line2D([0],[0],color='#AAAAAA',lw=2,label='Statistical edge (|ρ|≥0.30)'),
]
ax.legend(handles=legend_patches, loc="upper left", fontsize=9, framealpha=0.9)
ax.set_title(
    f"Graph F — Spearman |ρ|≥0.30 + Clinical Physiological Priors\n"
    f"67 edges (18 statistical + 49 clinical) | All 34 clinical features connected\n"
    f"Built from 28,235 training patients | Red = physiological prior | Gray = statistical",
    fontsize=12, fontweight="bold"
)
ax.axis("off")
plt.tight_layout()
p1 = os.path.join(OUT_DIR, "graph_F_full.png")
plt.savefig(p1, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved → {p1}")

# ── PLOT 2: Degree + centrality ───────────────────────────────────────
degree      = dict(G.degree())
betweenness = nx.betweenness_centrality(G, weight="weight")

sorted_deg  = sorted(degree.items(), key=lambda x: -x[1])
feats_d     = [feat_names[nd] for nd,_ in sorted_deg]
vals_d      = [v for _,v in sorted_deg]
colors_d    = [cat_color(f) for f in feats_d]

sorted_btw  = sorted(betweenness.items(), key=lambda x: -x[1])
feats_b     = [feat_names[nd] for nd,_ in sorted_btw]
vals_b      = [v for _,v in sorted_btw]
colors_b    = [cat_color(f) for f in feats_b]

fig, axes = plt.subplots(1, 2, figsize=(18, 10))

axes[0].barh(feats_d[::-1], vals_d[::-1], color=colors_d[::-1])
axes[0].set_xlabel("Degree (connections)", fontsize=11)
axes[0].set_title("Node Degree — Graph F", fontweight="bold", fontsize=12)
axes[0].grid(axis="x", alpha=0.3)
axes[0].tick_params(axis='y', labelsize=8)

axes[1].barh(feats_b[::-1], vals_b[::-1], color=colors_b[::-1])
axes[1].set_xlabel("Betweenness Centrality", fontsize=11)
axes[1].set_title("Betweenness Centrality — Graph F\n(bridge nodes between communities)",
                  fontweight="bold", fontsize=12)
axes[1].grid(axis="x", alpha=0.3)
axes[1].tick_params(axis='y', labelsize=8)

cat_legend = [
    plt.Line2D([0],[0],marker='o',color='w',markerfacecolor=c,markersize=11,label=cat)
    for cat,c in CAT_COLORS.items()
]
fig.legend(handles=cat_legend, loc="lower center", ncol=6,
           fontsize=9, framealpha=0.9, bbox_to_anchor=(0.5,-0.01))
plt.suptitle("Graph F — Node Importance Metrics", fontsize=13, fontweight="bold")
plt.tight_layout(rect=[0,0.04,1,1])
p2 = os.path.join(OUT_DIR, "graph_F_node_importance.png")
plt.savefig(p2, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved → {p2}")

# ── PLOT 3: Adjacency heatmap ─────────────────────────────────────────
adj = np.zeros((n, n))
for _, row in edges.iterrows():
    u, v = int(row.source), int(row.target)
    # Clinical edges shown at +1.0 offset to distinguish
    adj[u][v] = row.weight + (1.0 if row.clinical_prior == 1 else 0.0)

fig, ax = plt.subplots(figsize=(14, 12))
im = ax.imshow(adj, cmap="YlOrRd", vmin=0, vmax=2.0, aspect="auto")
ax.set_xticks(range(n))
ax.set_yticks(range(n))
ax.set_xticklabels(feat_names, rotation=90, fontsize=6)
ax.set_yticklabels(feat_names, fontsize=6)
plt.colorbar(im, ax=ax,
             label="Weight  (>1.0 = clinical prior offset,  0 = no edge)")
ax.set_title("Graph F — Adjacency Matrix\n"
             "Orange/red = clinical prior edges | Yellow = statistical edges",
             fontsize=12, fontweight="bold")
plt.tight_layout()
p3 = os.path.join(OUT_DIR, "graph_F_adjacency.png")
plt.savefig(p3, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved → {p3}")

# ── Print degree table ────────────────────────────────────────────────
print("\n=== DEGREE TABLE — Graph F ===")
print(f"{'Feature':<25} {'Degree':>6}  {'Category'}")
print("-"*50)
for nd, deg in sorted_deg:
    cat = next((c for c,fs in CATEGORIES.items() if feat_names[nd] in fs), "?")
    print(f"  {feat_names[nd]:<23} {deg:>6}  {cat}")

print(f"\n  Total nodes    : {G.number_of_nodes()}")
print(f"  Total edges    : {G.number_of_edges()}")
print(f"  Graph density  : {nx.density(G):.4f}")
print(f"  Is connected   : {nx.is_connected(G)}")
print(f"  Avg degree     : {np.mean(vals_d):.2f}")
print(f"\nAll plots → {OUT_DIR}")
