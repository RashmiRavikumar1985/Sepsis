import json, torch, os

print("=" * 60)
print("=== training_log_full.json ===")
with open("checkpoints/training_log_full.json") as f:
    log = json.load(f)

cfg = log.get("config", {})
print(f"lr_schedule    : {cfg.get('lr_schedule')}")
print(f"attention      : {cfg.get('attention')}")
print(f"learning_rate  : {cfg.get('learning_rate')}")
print(f"best_val_auprc : {log.get('best_val_auprc'):.4f}")
print(f"best_val_auroc : {log.get('best_val_auroc'):.4f}")
print(f"best_epoch     : {log.get('best_epoch')}")
tr = log.get("test_results", {})
if tr:
    print(f"test_auprc     : {tr.get('test_auprc'):.4f}")
    print(f"test_auroc     : {tr.get('test_auroc'):.4f}")
else:
    print("test_results   : NOT FOUND")

print()
print("Last 5 epochs:")
for e in log.get("epochs", [])[-5:]:
    print(f"  Epoch {e['epoch']:02d} | val_auprc={e['val_auprc']:.4f} | val_auroc={e['val_auroc']:.4f} | lr={e['lr']:.2e}")

print()
print("=" * 60)
print("=== best_grud_full.pt checkpoint ===")
ckpt = torch.load("checkpoints/best_grud_full.pt", map_location="cpu", weights_only=False)
ccfg = ckpt.get("config", {})
print(f"Saved epoch    : {ckpt.get('epoch')}")
print(f"val_auprc      : {ckpt.get('val_auprc'):.4f}")
print(f"val_auroc      : {ckpt.get('val_auroc'):.4f}")
print(f"lr_schedule    : {ccfg.get('lr_schedule')}")
print(f"attention      : {ccfg.get('attention')}")

print()
print("=" * 60)
print("=== experiments/results/baseline/metrics.json ===")
with open("experiments/results/baseline/metrics.json") as f:
    m = json.load(f)
for k, v in m.items():
    print(f"  {k}: {v}")

print()
print("=" * 60)
print("=== STATUS.md GRU-D original baseline ===")
print("  AUROC : 0.8436")
print("  AUPRC : 0.1125")
print("  (This was WITHOUT attention, logged before modifications)")
