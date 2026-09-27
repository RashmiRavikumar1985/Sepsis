"""
compare_models.py — Model comparison plots, training curves, radar chart.

Reads:
  GRU-D           : experiments/results/baseline/metrics.json
  Transformer (b) : experiments/results/transformer/metrics.json
  Transformer (t) : <project_root>/artifacts/tuned_metrics.json
  GAT             : experiments/results/gat/metrics.json
  GAT training log: experiments/results/gat/training_log_full.json
  GRU-D tuned     : experiments/results/grud_tuned/metrics.json  (if present)

Writes to experiments/results/comparison/:
  model_comparison.csv / .md
  auprc_comparison.png
  auroc_comparison.png
  full_comparison.png
  radar_comparison.png
  gat_training_curve.png
  gat_training_curve_lr.png
  grud_tuned_comparison.png   (if tuned GRU-D results exist)
  graph_density_comparison.png
"""

import os
import sys
import json
import math

import numpy as np
import pandas as pd

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    HAS_MATPLOTLIB = True
except Exception:
    HAS_MATPLOTLIB = False

# ── Paths ──────────────────────────────────────────────────────────────
_HERE    = os.path.dirname(os.path.abspath(__file__))
SEPSIS   = os.path.dirname(_HERE)
SEPSIS1  = os.path.dirname(SEPSIS)
RESULTS  = os.path.join(_HERE, "results")
COMP_DIR = os.path.join(RESULTS, "comparison")
os.makedirs(COMP_DIR, exist_ok=True)

# ── Colour palette (consistent across all plots) ───────────────────────
PALETTE = {
    "GRU-D"                    : "#4C72B0",
    "GRU-D (Tuned)"            : "#1A4A8A",
    "Transformer (d=64)"       : "#DD8452",
    "Transformer (tuned d=128)": "#A05020",
    "GAT (Baseline)"           : "#55A868",
    "GAT (Tuned)"              : "#1E6B3A",
}
COLORS_LIST = list(PALETTE.values())

# ── Load metrics ───────────────────────────────────────────────────────
SOURCES = [
    {
        "model": "GRU-D",
        "path" : os.path.join(RESULTS, "baseline", "metrics.json"),
        "keys" : {"Test AUPRC": "Test AUPRC", "Test AUROC": "Test AUROC",
                  "Test Precision": "Test Precision", "Test Recall": "Test Recall",
                  "Test F1": "Test F1"},
    },
    {
        "model": "GRU-D (Tuned)",
        "path" : os.path.join(RESULTS, "grud_tuned", "metrics.json"),
        "keys" : {"Test AUPRC": "Test AUPRC", "Test AUROC": "Test AUROC",
                  "Test Precision": "Test Precision", "Test Recall": "Test Recall",
                  "Test F1": "Test F1"},
    },
    {
        "model": "Transformer (d=64)",
        "path" : os.path.join(RESULTS, "transformer", "metrics.json"),
        "keys" : {"Test AUPRC": "Test AUPRC", "Test AUROC": "Test AUROC",
                  "Test Precision": "Test Precision", "Test Recall": "Test Recall",
                  "Test F1": "Test F1"},
    },
    {
        "model": "Transformer (tuned d=128)",
        "path" : os.path.join(SEPSIS1, "artifacts", "tuned_metrics.json"),
        "keys" : {"Test AUPRC": "test_auprc", "Test AUROC": "test_auroc",
                  "Test Precision": None, "Test Recall": None, "Test F1": None},
    },
    {
        "model": "GAT (Baseline)",
        "path" : os.path.join(RESULTS, "gat", "metrics.json"),
        "keys" : {"Test AUPRC": "Test AUPRC", "Test AUROC": "Test AUROC",
                  "Test Precision": "Test Precision", "Test Recall": "Test Recall",
                  "Test F1": "Test F1"},
    },
]

rows = []
for src in SOURCES:
    if not os.path.exists(src["path"]):
        print(f"  (skipping {src['model']} — no metrics file at {src['path']})")
        continue
    with open(src["path"]) as f:
        raw = json.load(f)
    row = {"Model": src["model"]}
    for col, key in src["keys"].items():
        row[col] = raw.get(key) if key else None
    rows.append(row)

df = pd.DataFrame(rows).set_index("Model")
df.index.name = "Model"
COLS = ["Test AUPRC", "Test AUROC", "Test Precision", "Test Recall", "Test F1"]
df = df[[c for c in COLS if c in df.columns]]

print("\n=== MODEL COMPARISON ===")
print(df.to_string(float_format=lambda x: f"{x:.4f}" if x is not None else "N/A"))

# ── Save CSV & Markdown ────────────────────────────────────────────────
csv_path = os.path.join(COMP_DIR, "model_comparison.csv")
df.to_csv(csv_path)
print(f"\nCSV  → {csv_path}")

md_path = os.path.join(COMP_DIR, "model_comparison.md")
with open(md_path, "w") as f:
    f.write("# Experimental Model Comparison\n\n")
    # Build markdown table manually (no tabulate dependency)
    cols = list(df.columns)
    header = "| Model | " + " | ".join(cols) + " |"
    sep    = "| :--- | " + " | ".join(["---:" for _ in cols]) + " |"
    f.write(header + "\n" + sep + "\n")
    for model, row in df.iterrows():
        vals = []
        for c in cols:
            v = row[c]
            vals.append(f"{v:.4f}" if pd.notna(v) else "N/A")
        f.write("| " + model + " | " + " | ".join(vals) + " |\n")
    f.write("\n_GRU-D Precision/Recall/F1 at F1-optimal threshold._  \n")
    f.write("_Transformer/GAT Precision/Recall/F1 at threshold=0.5._  \n")
print(f"MD   → {md_path}")

if not HAS_MATPLOTLIB:
    print("matplotlib not available — skipping plots.")
    raise SystemExit

models   = df.index.tolist()
n_models = len(models)
colors   = [PALETTE.get(m, COLORS_LIST[i % len(COLORS_LIST)]) for i, m in enumerate(models)]

# ═══════════════════════════════════════════════════════════════════════
# PLOT 1 — AUPRC bar chart
# ═══════════════════════════════════════════════════════════════════════
auprc_vals = df["Test AUPRC"].tolist()
fig, ax = plt.subplots(figsize=(10, 5))
bars = ax.bar(models, auprc_vals, color=colors, width=0.55, zorder=3,
              edgecolor="white", linewidth=0.8)
ax.bar_label(bars, fmt="%.4f", padding=5, fontsize=10, fontweight="bold")
ax.set_ylabel("AUPRC", fontsize=12)
ax.set_title("Test AUPRC — All Models\n(higher is better; random baseline ≈ 0.018)",
             fontsize=13, fontweight="bold")
ax.set_ylim(0, max(v for v in auprc_vals if v) * 1.30)
ax.yaxis.grid(True, linestyle="--", alpha=0.5)
ax.set_axisbelow(True)
ax.axhline(0.018, color="gray", linestyle=":", lw=1.5, label="Random baseline")
ax.tick_params(axis="x", labelsize=9)
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(COMP_DIR, "auprc_comparison.png"), dpi=150)
plt.close()
print(f"Plot → {os.path.join(COMP_DIR, 'auprc_comparison.png')}")

# ═══════════════════════════════════════════════════════════════════════
# PLOT 2 — AUROC bar chart
# ═══════════════════════════════════════════════════════════════════════
auroc_vals = df["Test AUROC"].tolist()
fig, ax = plt.subplots(figsize=(10, 5))
bars = ax.bar(models, auroc_vals, color=colors, width=0.55, zorder=3,
              edgecolor="white", linewidth=0.8)
ax.bar_label(bars, fmt="%.4f", padding=5, fontsize=10, fontweight="bold")
ax.set_ylabel("AUROC", fontsize=12)
ax.set_title("Test AUROC — All Models\n(higher is better; random baseline = 0.5)",
             fontsize=13, fontweight="bold")
ax.set_ylim(0.5, 1.05)
ax.yaxis.grid(True, linestyle="--", alpha=0.5)
ax.set_axisbelow(True)
ax.axhline(0.5, color="gray", linestyle=":", lw=1.5, label="Random baseline")
ax.tick_params(axis="x", labelsize=9)
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(COMP_DIR, "auroc_comparison.png"), dpi=150)
plt.close()
print(f"Plot → {os.path.join(COMP_DIR, 'auroc_comparison.png')}")

# ═══════════════════════════════════════════════════════════════════════
# PLOT 3 — Full grouped bar chart (AUPRC, AUROC, Precision, Recall, F1)
# ═══════════════════════════════════════════════════════════════════════
metrics_to_plot = [c for c in COLS if c in df.columns]
x      = np.arange(len(metrics_to_plot))
width  = 0.80 / n_models
offsets = np.linspace(-(n_models - 1) / 2 * width, (n_models - 1) / 2 * width, n_models)

fig, ax = plt.subplots(figsize=(14, 6))
for i, (model, offset, color) in enumerate(zip(models, offsets, colors)):
    vals = [df.loc[model, m] if pd.notna(df.loc[model, m]) else 0 for m in metrics_to_plot]
    bars = ax.bar(x + offset, vals, width=width, label=model, color=color, zorder=3,
                  edgecolor="white", linewidth=0.5)
    ax.bar_label(bars, fmt="%.3f", padding=2, fontsize=6.5, rotation=90)

ax.set_xticks(x)
ax.set_xticklabels([m.replace("Test ", "") for m in metrics_to_plot], fontsize=12)
ax.set_ylabel("Score", fontsize=12)
ax.set_title("Full Metrics Comparison — All Models", fontsize=14, fontweight="bold")
ax.set_ylim(0, 1.25)
ax.yaxis.grid(True, linestyle="--", alpha=0.5)
ax.set_axisbelow(True)
ax.legend(fontsize=9, loc="upper right")
plt.tight_layout()
plt.savefig(os.path.join(COMP_DIR, "full_comparison.png"), dpi=150)
plt.close()
print(f"Plot → {os.path.join(COMP_DIR, 'full_comparison.png')}")

# ═══════════════════════════════════════════════════════════════════════
# PLOT 4 — Radar / Spider chart
# ═══════════════════════════════════════════════════════════════════════
radar_metrics = ["Test AUPRC", "Test AUROC", "Test Precision", "Test Recall", "Test F1"]
radar_metrics = [m for m in radar_metrics if m in df.columns]
N = len(radar_metrics)
angles = [n / float(N) * 2 * math.pi for n in range(N)]
angles += angles[:1]  # close the polygon

fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(polar=True))

for model, color in zip(models, colors):
    vals = []
    for m in radar_metrics:
        v = df.loc[model, m]
        vals.append(float(v) if pd.notna(v) else 0.0)
    vals += vals[:1]
    ax.plot(angles, vals, "o-", lw=2, label=model, color=color)
    ax.fill(angles, vals, alpha=0.08, color=color)

ax.set_thetagrids(
    [a * 180 / math.pi for a in angles[:-1]],
    [m.replace("Test ", "") for m in radar_metrics],
    fontsize=11
)
ax.set_ylim(0, 1)
ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=8, color="grey")
ax.set_title("Model Comparison — Radar Chart\n(all metrics normalised to [0,1])",
             size=13, fontweight="bold", pad=20)
ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15), fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(COMP_DIR, "radar_comparison.png"), dpi=150, bbox_inches="tight")
plt.close()
print(f"Plot → {os.path.join(COMP_DIR, 'radar_comparison.png')}")

# ═══════════════════════════════════════════════════════════════════════
# PLOT 5 — GRU-D vs GRU-D Tuned side-by-side (if tuned results exist)
# ═══════════════════════════════════════════════════════════════════════
if "GRU-D (Tuned)" in df.index and "GRU-D" in df.index:
    grud_models  = ["GRU-D", "GRU-D (Tuned)"]
    grud_colors  = [PALETTE["GRU-D"], PALETTE["GRU-D (Tuned)"]]
    grud_metrics = [m for m in COLS if m in df.columns]

    x_g   = np.arange(len(grud_metrics))
    w_g   = 0.35
    fig, ax = plt.subplots(figsize=(11, 5))
    for i, (m, c) in enumerate(zip(grud_models, grud_colors)):
        vals = [float(df.loc[m, met]) if pd.notna(df.loc[m, met]) else 0 for met in grud_metrics]
        bars = ax.bar(x_g + (i - 0.5) * w_g, vals, w_g, label=m, color=c,
                      zorder=3, edgecolor="white")
        ax.bar_label(bars, fmt="%.4f", padding=3, fontsize=9)

    ax.set_xticks(x_g)
    ax.set_xticklabels([m.replace("Test ", "") for m in grud_metrics], fontsize=11)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("GRU-D Baseline vs Tuned", fontsize=14, fontweight="bold")
    ax.set_ylim(0, 1.15)
    ax.yaxis.grid(True, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)
    ax.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(COMP_DIR, "grud_tuned_comparison.png"), dpi=150)
    plt.close()
    print(f"Plot → {os.path.join(COMP_DIR, 'grud_tuned_comparison.png')}")

# ═══════════════════════════════════════════════════════════════════════
# PLOT 6 — GAT training curve (Val AUROC + Val AUPRC + Train Loss)
# ═══════════════════════════════════════════════════════════════════════
gat_log_path = os.path.join(RESULTS, "gat", "training_log_full.json")
if os.path.exists(gat_log_path):
    with open(gat_log_path) as f:
        gat_log = json.load(f)

    epochs      = [e["epoch"]       for e in gat_log["epochs"]]
    train_loss  = [e["train_loss"]  for e in gat_log["epochs"]]
    val_loss    = [e["val_loss"]    for e in gat_log["epochs"]]
    val_auroc   = [e["val_auroc"]   for e in gat_log["epochs"]]
    val_auprc   = [e["val_auprc"]   for e in gat_log["epochs"]]
    lr_vals     = [e["lr"]          for e in gat_log["epochs"]]
    best_ep     = gat_log["best_epoch"]

    # — Training & validation loss + AUROC/AUPRC on twin axes ——————————
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    # Left: loss curves
    ax1 = axes[0]
    ax1.plot(epochs, train_loss, lw=2, color="#4C72B0", label="Train Loss")
    ax1.plot(epochs, val_loss,   lw=2, color="#DD8452", label="Val Loss",
             linestyle="--")
    ax1.axvline(best_ep, color="#55A868", linestyle=":", lw=1.8, label=f"Best epoch ({best_ep})")
    ax1.set_xlabel("Epoch", fontsize=11)
    ax1.set_ylabel("Loss", fontsize=11)
    ax1.set_title("GAT — Training & Validation Loss", fontsize=12, fontweight="bold")
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    # Right: AUROC + AUPRC
    ax2 = axes[1]
    ax2.plot(epochs, val_auroc, lw=2, color="#4C72B0", label="Val AUROC")
    ax2.plot(epochs, val_auprc, lw=2, color="#DD8452", label="Val AUPRC", linestyle="--")
    ax2.axvline(best_ep, color="#55A868", linestyle=":", lw=1.8, label=f"Best epoch ({best_ep})")
    ax2.set_xlabel("Epoch", fontsize=11)
    ax2.set_ylabel("Score", fontsize=11)
    ax2.set_title("GAT — Validation AUROC & AUPRC", fontsize=12, fontweight="bold")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    plt.suptitle(
        f"GAT Training Curve (Graph F, {len(epochs)} epochs)\n"
        f"Best Val AUPRC={gat_log['best_val_auprc']:.4f} @ epoch {best_ep}  |  "
        f"Test AUROC={gat_log['test_results']['test_auroc']:.4f}  |  "
        f"Test AUPRC={gat_log['test_results']['test_auprc']:.4f}",
        fontsize=11, fontweight="bold"
    )
    plt.tight_layout()
    plt.savefig(os.path.join(COMP_DIR, "gat_training_curve.png"), dpi=150)
    plt.close()
    print(f"Plot → {os.path.join(COMP_DIR, 'gat_training_curve.png')}")

    # — LR schedule plot ————————————————————————————————————————————————
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(epochs, lr_vals, lw=2, color="#9B59B6", marker="o", markersize=3)
    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("Learning Rate", fontsize=11)
    ax.set_title("GAT — Learning Rate Schedule (ReduceLROnPlateau)",
                 fontsize=12, fontweight="bold")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3, which="both")

    # Annotate LR reduction points
    for i in range(1, len(lr_vals)):
        if lr_vals[i] < lr_vals[i - 1]:
            ax.annotate(f"÷2", xy=(epochs[i], lr_vals[i]),
                        xytext=(epochs[i] + 0.5, lr_vals[i] * 1.8),
                        fontsize=8, color="#9B59B6",
                        arrowprops=dict(arrowstyle="->", color="#9B59B6", lw=1))

    plt.tight_layout()
    plt.savefig(os.path.join(COMP_DIR, "gat_training_curve_lr.png"), dpi=150)
    plt.close()
    print(f"Plot → {os.path.join(COMP_DIR, 'gat_training_curve_lr.png')}")

# ═══════════════════════════════════════════════════════════════════════
# PLOT 7 — Graph density comparison (all 7 graphs A–G)
# ═══════════════════════════════════════════════════════════════════════
graph_summary_path = os.path.join(_HERE, "gat", "graphs", "graph_summary.json")
if os.path.exists(graph_summary_path):
    with open(graph_summary_path) as f:
        graph_data = json.load(f)

    g_ids    = [g["graph_id"]         for g in graph_data]
    n_edges  = [g["n_edges_unique"]    for g in graph_data]
    w_means  = [g["weight_mean"]       for g in graph_data]
    n_clin   = [g["n_clinical_edges"]  for g in graph_data]
    n_stat   = [g["n_statistical_edges"] for g in graph_data]
    descs    = [g["description"]       for g in graph_data]

    x_g = np.arange(len(g_ids))
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # — Edge count (stacked: clinical + statistical) ——————————————————
    bars1 = axes[0].bar(g_ids, n_stat, color="#AAAAAA", label="Statistical", zorder=3)
    bars2 = axes[0].bar(g_ids, n_clin, bottom=n_stat, color="#E74C3C",
                        label="Clinical Prior", zorder=3)
    axes[0].set_ylabel("Unique Edges", fontsize=11)
    axes[0].set_title("Edge Count per Graph\n(clinical vs statistical)",
                      fontsize=11, fontweight="bold")
    axes[0].legend(fontsize=9)
    axes[0].yaxis.grid(True, alpha=0.3)
    axes[0].set_axisbelow(True)
    # Annotate total
    for idx, (s, c) in enumerate(zip(n_stat, n_clin)):
        axes[0].text(idx, s + c + 1, str(s + c), ha="center", fontsize=9, fontweight="bold")
    # Highlight selected graph F
    sel_idx = g_ids.index("F")
    axes[0].get_children()[sel_idx].set_edgecolor("gold")
    axes[0].get_children()[sel_idx].set_linewidth(2.5)

    # — Mean edge weight ——————————————————————————————————————————————
    bar_colors_w = ["gold" if g == "F" else "#55A868" for g in g_ids]
    bars3 = axes[1].bar(g_ids, w_means, color=bar_colors_w, zorder=3,
                        edgecolor="white", linewidth=0.8)
    axes[1].bar_label(bars3, fmt="%.3f", padding=3, fontsize=9)
    axes[1].set_ylabel("Mean Edge Weight", fontsize=11)
    axes[1].set_title("Mean Edge Weight per Graph\n(gold = selected Graph F)",
                      fontsize=11, fontweight="bold")
    axes[1].set_ylim(0, max(w_means) * 1.3)
    axes[1].yaxis.grid(True, alpha=0.3)
    axes[1].set_axisbelow(True)

    # — Total edges as bubble ——————————————————————————————————————————
    n_total = [g["n_edges_total"] for g in graph_data]
    bubble_colors = ["gold" if g == "F" else "#3498DB" for g in g_ids]
    axes[2].scatter(g_ids, w_means,
                    s=[t * 1.5 for t in n_total],
                    c=bubble_colors, alpha=0.8, edgecolors="white", linewidths=1.5, zorder=3)
    for i, (gid, nm, wm) in enumerate(zip(g_ids, n_total, w_means)):
        axes[2].annotate(f"{nm} edges", (gid, wm),
                         textcoords="offset points", xytext=(0, 10),
                         ha="center", fontsize=8)
    axes[2].set_ylabel("Mean Edge Weight", fontsize=11)
    axes[2].set_title("Graph Density Bubble Chart\n(bubble size = total edge count)",
                      fontsize=11, fontweight="bold")
    axes[2].yaxis.grid(True, alpha=0.3)
    axes[2].set_axisbelow(True)

    plt.suptitle("GAT Graph Variants A–G Comparison  |  ★ = Selected (Graph F)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(COMP_DIR, "graph_density_comparison.png"), dpi=150)
    plt.close()
    print(f"Plot → {os.path.join(COMP_DIR, 'graph_density_comparison.png')}")

print("\n✓ All comparison plots saved to:", COMP_DIR)
