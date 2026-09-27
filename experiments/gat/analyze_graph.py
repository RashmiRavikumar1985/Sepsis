"""
analyze_graph.py — Full analysis and visualization of the GAT clinical feature graph.

Saves all plots to experiments/gat/plots/graph_analysis/
"""

import os
import json
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors

import networkx as nx

# ── Paths ──────────────────────────────────────────────────────────────
_HERE    = os.path.dirname(os.path.abspath(__file__))
EDGES    = os.path.join(_HERE, "edges.csv")
NODES    = os.path.join(_HERE, "nodes.csv")
OUT_DIR  = os.path.join(_HERE, "plots", "graph_analysis")
os.makedirs(OUT_DIR, exist_ok=True)

edges_df = pd.read_csv(EDGES)
nodes_df = pd.read_csv(NODES)
feat_names = nodes_df["feature_name"].tolist()

# ── Build NetworkX graph ───────────────────────────────────────────────
G = nx.Graph()
for _, row in nodes_df.iterrows():
    G.add_node(int(row["node_id"]), label=row["feature_name"])

# Only add one direction (undirected)
seen = set()
for _, row in edges_df.iterrows():
    u, v = int(row["source"]), int(row["target"])
    if (min(u,v), max(u,v)) not in seen:
        G.add_edge(u, v, weight=float(row["weight"]))
        seen.add((min(u,v), max(u,v)))

print(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

# ── Clinical feature categories for coloring ──────────────────────────
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
feat_to_cat = {}
for cat, feats in CATEGORIES.items():
    for f in feats:
        feat_to_cat[f] = cat

node_colors = [CAT_COLORS.get(feat_to_cat.get(feat_names[n], "Time"), "#95A5A6")
               for n in G.nodes()]
node_labels = {n: feat_names[n] for n in G.nodes()}

# ── Compute layout ─────────────────────────────────────────────────────
pos = nx.spring_layout(G, seed=42, k=2.5, weight="weight")

# ── PLOT 1: Full graph colored by category ────────────────────────────
fig, ax = plt.subplots(figsize=(18, 14))

edge_weights = [G[u][v]["weight"] for u, v in G.edges()]
edge_alphas  = [0.2 + 0.8 * (w - 0.2) / (1.0 - 0.2) for w in edge_weights]
edge_widths  = [0.5 + 3.0 * (w - 0.2) / (1.0 - 0.2) for w in edge_weights]

for (u, v), alpha, width in zip(G.edges(), edge_alphas, edge_widths):
    x = [pos[u][0], pos[v][0]]
    y = [pos[u][1], pos[v][1]]
    ax.plot(x, y, color="#AAAAAA", alpha=alpha, linewidth=width, zorder=1)

nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=800, ax=ax)
nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=7, font_weight="bold", ax=ax)

# Legend
legend_patches = [plt.Line2D([0],[0], marker='o', color='w',
                              markerfacecolor=c, markersize=12, label=cat)
                  for cat, c in CAT_COLORS.items()]
ax.legend(handles=legend_patches, loc="upper left", fontsize=10, framealpha=0.9)
ax.set_title("GAT Clinical Feature Graph — Full View\n"
             "Edge thickness = correlation strength | Colors = feature category",
             fontsize=14, fontweight="bold")
ax.axis("off")
plt.tight_layout()
p1 = os.path.join(OUT_DIR, "graph_full.png")
plt.savefig(p1, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved → {p1}")

# ── PLOT 2: Strong edges only (weight >= 0.5) ─────────────────────────
strong_edges = [(u, v) for u, v, d in G.edges(data=True) if d["weight"] >= 0.5]
G_strong = G.edge_subgraph(strong_edges)

fig, ax = plt.subplots(figsize=(14, 10))
pos_s = nx.spring_layout(G_strong, seed=42, k=3.0, weight="weight")
s_colors = [CAT_COLORS.get(feat_to_cat.get(feat_names[n], "Time"), "#95A5A6")
            for n in G_strong.nodes()]
s_weights = [G_strong[u][v]["weight"] for u, v in G_strong.edges()]
s_widths  = [1.0 + 4.0 * (w - 0.5) / 0.5 for w in s_weights]

for (u, v), width, w in zip(G_strong.edges(), s_widths, s_weights):
    x = [pos_s[u][0], pos_s[v][0]]
    y = [pos_s[u][1], pos_s[v][1]]
    ax.plot(x, y, color="#555555", alpha=0.7, linewidth=width, zorder=1)
    mx, my = (x[0]+x[1])/2, (y[0]+y[1])/2
    ax.text(mx, my, f"{w:.2f}", fontsize=6, color="#333333",
            ha="center", va="center", zorder=4)

nx.draw_networkx_nodes(G_strong, pos_s, node_color=s_colors, node_size=1200, ax=ax)
nx.draw_networkx_labels(G_strong, pos_s, labels={n: feat_names[n] for n in G_strong.nodes()},
                        font_size=8, font_weight="bold", ax=ax)
ax.legend(handles=legend_patches, loc="upper left", fontsize=10, framealpha=0.9)
ax.set_title("GAT Graph — Strong Edges Only (weight >= 0.5)\nEdge labels = correlation weight",
             fontsize=13, fontweight="bold")
ax.axis("off")
plt.tight_layout()
p2 = os.path.join(OUT_DIR, "graph_strong_edges.png")
plt.savefig(p2, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved → {p2}")

# ── PLOT 3: Correlation heatmap ────────────────────────────────────────
adj = np.zeros((35, 35))
for _, row in edges_df.iterrows():
    u, v = int(row["source"]), int(row["target"])
    adj[u][v] = row["weight"]

fig, ax = plt.subplots(figsize=(14, 12))
im = ax.imshow(adj, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
ax.set_xticks(range(35))
ax.set_yticks(range(35))
ax.set_xticklabels(feat_names, rotation=90, fontsize=7)
ax.set_yticklabels(feat_names, fontsize=7)
plt.colorbar(im, ax=ax, label="Correlation Weight")
ax.set_title("GAT Graph — Adjacency Matrix (Correlation Weights)",
             fontsize=13, fontweight="bold")
plt.tight_layout()
p3 = os.path.join(OUT_DIR, "graph_adjacency_heatmap.png")
plt.savefig(p3, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved → {p3}")

# ── PLOT 4: Node degree + betweenness centrality ──────────────────────
degree      = dict(G.degree())
betweenness = nx.betweenness_centrality(G, weight="weight")

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Degree bar
sorted_deg = sorted(degree.items(), key=lambda x: -x[1])
feat_d  = [feat_names[n] for n, _ in sorted_deg]
vals_d  = [v for _, v in sorted_deg]
colors_d = [CAT_COLORS.get(feat_to_cat.get(f, "Time"), "#95A5A6") for f in feat_d]
axes[0].barh(feat_d[::-1], vals_d[::-1], color=colors_d[::-1])
axes[0].set_xlabel("Degree (number of connections)")
axes[0].set_title("Node Degree", fontweight="bold")
axes[0].grid(axis="x", alpha=0.3)

# Betweenness bar
sorted_btw = sorted(betweenness.items(), key=lambda x: -x[1])
feat_b  = [feat_names[n] for n, _ in sorted_btw]
vals_b  = [v for _, v in sorted_btw]
colors_b = [CAT_COLORS.get(feat_to_cat.get(f, "Time"), "#95A5A6") for f in feat_b]
axes[1].barh(feat_b[::-1], vals_b[::-1], color=colors_b[::-1])
axes[1].set_xlabel("Betweenness Centrality")
axes[1].set_title("Node Betweenness Centrality\n(Bridge nodes between communities)",
                  fontweight="bold")
axes[1].grid(axis="x", alpha=0.3)

plt.suptitle("GAT Graph — Node Importance Metrics", fontsize=13, fontweight="bold")
plt.tight_layout()
p4 = os.path.join(OUT_DIR, "graph_node_importance.png")
plt.savefig(p4, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved → {p4}")

# ── PLOT 5: Edge weight distribution ─────────────────────────────────
unique_weights = edges_df.drop_duplicates(subset=["source","target"])[
    edges_df["source"] < edges_df["target"]]["weight"]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].hist(unique_weights, bins=20, color="#4C72B0", edgecolor="white", alpha=0.8)
axes[0].axvline(0.3, color="red",    linestyle="--", label="Proposed threshold (0.3)")
axes[0].axvline(0.2, color="orange", linestyle="--", label="Current threshold (0.2)")
axes[0].set_xlabel("Correlation Weight")
axes[0].set_ylabel("Edge Count")
axes[0].set_title("Edge Weight Distribution")
axes[0].legend()
axes[0].grid(alpha=0.3)

thresholds = np.arange(0.1, 0.9, 0.05)
edge_counts = []
for t in thresholds:
    c = ((unique_weights >= t)).sum()
    edge_counts.append(c)
axes[1].plot(thresholds, edge_counts, marker="o", color="#DD8452", lw=2)
axes[1].axvline(0.2, color="orange", linestyle="--", label="Current (0.2) → 122 edges")
axes[1].axvline(0.3, color="red",    linestyle="--", label="Proposed (0.3) → fewer edges")
axes[1].set_xlabel("Threshold")
axes[1].set_ylabel("Number of Edges")
axes[1].set_title("Edge Count vs Threshold")
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.suptitle("GAT Graph — Edge Weight Analysis", fontsize=13, fontweight="bold")
plt.tight_layout()
p5 = os.path.join(OUT_DIR, "graph_edge_analysis.png")
plt.savefig(p5, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved → {p5}")

# ── Summary stats ─────────────────────────────────────────────────────
print()
print("=== GRAPH ANALYSIS SUMMARY ===")
print(f"  Nodes            : {G.number_of_nodes()}")
print(f"  Edges (unique)   : {G.number_of_edges()}")
print(f"  Avg degree       : {np.mean(list(degree.values())):.2f}")
print(f"  Max degree       : {max(degree.values())} ({feat_names[max(degree, key=degree.get)]})")
print(f"  Graph density    : {nx.density(G):.4f}")
print(f"  Is connected     : {nx.is_connected(G)}")
print(f"  Avg clustering   : {nx.average_clustering(G, weight='weight'):.4f}")
weak_edges = (unique_weights < 0.3).sum()
print(f"  Weak edges (<0.3): {weak_edges} ({100*weak_edges/len(unique_weights):.1f}% of all edges)")
print(f"  Strong edges(>0.5): {(unique_weights >= 0.5).sum()}")
print()
print(f"  All plots → {OUT_DIR}")
