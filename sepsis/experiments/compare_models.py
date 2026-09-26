"""
compare_models.py — Regenerate model comparison from current metric files.

Reads:
  GRU-D       : experiments/results/baseline/metrics.json   (full eval)
  Transformer : experiments/results/transformer/metrics.json
  GAT         : experiments/results/gat/metrics.json

Writes:
  experiments/results/comparison/model_comparison.csv
  experiments/results/comparison/model_comparison.md
  experiments/results/comparison/auprc_comparison.png
  experiments/results/comparison/full_comparison.png
"""

import os
import sys
import json

import pandas as pd

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    HAS_MATPLOTLIB = True
except Exception:
    HAS_MATPLOTLIB = False

# ── Paths ──────────────────────────────────────────────────────────────
_HERE       = os.path.dirname(os.path.abspath(__file__))
SEPSIS      = os.path.dirname(_HERE)                        # sepsis/
SEPSIS1     = os.path.dirname(SEPSIS)                       # sepsis.1/
RESULTS     = os.path.join(_HERE, "results")
COMP_DIR    = os.path.join(RESULTS, "comparison")
os.makedirs(COMP_DIR, exist_ok=True)

# ── Load metrics ───────────────────────────────────────────────────────
SOURCES = [
    {
        "model" : "GRU-D",
        "path"  : os.path.join(RESULTS, "baseline", "metrics.json"),
        "keys"  : {
            "Test AUPRC"     : "Test AUPRC",
            "Test AUROC"     : "Test AUROC",
            "Test Precision" : "Test Precision",
            "Test Recall"    : "Test Recall",
            "Test F1"        : "Test F1",
        },
    },
    {
        "model" : "Transformer (baseline d=64)",
        "path"  : os.path.join(RESULTS, "transformer", "metrics.json"),
        "keys"  : {
            "Test AUPRC"     : "Test AUPRC",
            "Test AUROC"     : "Test AUROC",
            "Test Precision" : "Test Precision",
            "Test Recall"    : "Test Recall",
            "Test F1"        : "Test F1",
        },
    },
    {
        "model" : "Transformer (tuned d=128)",
        "path"  : os.path.join(SEPSIS1, "artifacts", "tuned_metrics.json"),
        "keys"  : {
            "Test AUPRC"     : "test_auprc",
            "Test AUROC"     : "test_auroc",
            "Test Precision" : None,
            "Test Recall"    : None,
            "Test F1"        : None,
        },
    },
    {
        "model" : "GAT (Graph Baseline)",
        "path"  : os.path.join(RESULTS, "gat", "metrics.json"),
        "keys"  : {
            "Test AUPRC"     : "Test AUPRC",
            "Test AUROC"     : "Test AUROC",
            "Test Precision" : "Test Precision",
            "Test Recall"    : "Test Recall",
            "Test F1"        : "Test F1",
        },
    },
]

rows = []
for src in SOURCES:
    if not os.path.exists(src["path"]):
        print(f"WARNING: missing metrics for {src['model']}: {src['path']}")
        continue
    with open(src["path"]) as f:
        raw = json.load(f)
    row = {"Model": src["model"]}
    for col, key in src["keys"].items():
        row[col] = raw.get(key, None) if key is not None else None
    rows.append(row)

df = pd.DataFrame(rows).set_index("Model")
COLS = ["Test AUPRC", "Test AUROC", "Test Precision", "Test Recall", "Test F1"]
df = df[[c for c in COLS if c in df.columns]]

# ── Print to console ───────────────────────────────────────────────────
print("\n=== MODEL COMPARISON ===")
print(df.to_string(float_format=lambda x: f"{x:.4f}"))

# ── Save CSV ───────────────────────────────────────────────────────────
csv_path = os.path.join(COMP_DIR, "model_comparison.csv")
df.to_csv(csv_path)
print(f"\nCSV  → {csv_path}")

# ── Save Markdown ──────────────────────────────────────────────────────
md_path = os.path.join(COMP_DIR, "model_comparison.md")
with open(md_path, "w") as f:
    f.write("# Experimental Model Comparison\n\n")
    f.write(df.to_markdown(floatfmt=".4f"))
    f.write("\n\n")
    f.write("_GRU-D Precision/Recall/F1 computed at F1-optimal threshold._  \n")
    f.write("_Transformer Precision/Recall/F1 computed at threshold=0.5._  \n")
    f.write("_GAT Precision/Recall/F1 computed at threshold=0.5._  \n")
print(f"MD   → {md_path}")

# ── Plots ──────────────────────────────────────────────────────────────
if HAS_MATPLOTLIB:
    COLORS = ["#4C72B0", "#DD8452", "#55A868"]
    models = df.index.tolist()

    # ── Plot 1: AUPRC bar chart ────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(models, df["Test AUPRC"], color=COLORS[:len(models)], width=0.5, zorder=3)
    ax.bar_label(bars, fmt="%.4f", padding=4, fontsize=11, fontweight="bold")
    ax.set_ylabel("AUPRC", fontsize=12)
    ax.set_title("Test AUPRC — Model Comparison", fontsize=14, fontweight="bold")
    ax.set_ylim(0, max(df["Test AUPRC"]) * 1.25)
    ax.yaxis.grid(True, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)
    plt.tight_layout()
    p1 = os.path.join(COMP_DIR, "auprc_comparison.png")
    plt.savefig(p1, dpi=150)
    plt.close()
    print(f"Plot → {p1}")

    # ── Plot 2: Full metrics grouped bar chart ─────────────────────────
    metrics = ["Test AUPRC", "Test AUROC", "Test Precision", "Test Recall", "Test F1"]
    metrics = [m for m in metrics if m in df.columns]
    x      = range(len(metrics))
    n      = len(models)
    width  = 0.22
    offsets = [-(n - 1) / 2 * width + i * width for i in range(n)]

    fig, ax = plt.subplots(figsize=(12, 6))
    for i, (model, offset) in enumerate(zip(models, offsets)):
        vals = [df.loc[model, m] for m in metrics]
        bars = ax.bar(
            [xi + offset for xi in x], vals,
            width=width, label=model,
            color=COLORS[i % len(COLORS)], zorder=3,
        )
        ax.bar_label(bars, fmt="%.3f", padding=2, fontsize=7)

    ax.set_xticks(list(x))
    ax.set_xticklabels([m.replace("Test ", "") for m in metrics], fontsize=11)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Full Metrics Comparison — All Models", fontsize=14, fontweight="bold")
    ax.set_ylim(0, 1.15)
    ax.yaxis.grid(True, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)
    ax.legend(fontsize=10)
    plt.tight_layout()
    p2 = os.path.join(COMP_DIR, "full_comparison.png")
    plt.savefig(p2, dpi=150)
    plt.close()
    print(f"Plot → {p2}")

print("\nDone.")
