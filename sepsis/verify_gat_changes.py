import ast, json

with open("experiments/gat/train.py") as f:
    src = f.read()
ast.parse(src)
print("train.py syntax : OK")

cfg = json.load(open("experiments/gat/config_gat.json"))
print(f"batch_size      : {cfg['batch_size']}   (was 16)")
print(f"learning_rate   : {cfg['learning_rate']}  (was 5e-4)")
print(f"patience        : {cfg['patience']}  (was 8)")
print(f"scheduler_pat   : {cfg['scheduler_patience']}    (was 3)")
print(f"epochs          : {cfg['epochs']}   (was 30)")

# Check evaluate_full is present
assert "evaluate_full" in src, "evaluate_full missing!"
assert "test_precision" in src, "test_precision missing!"
assert "test_recall" in src, "test_recall missing!"
assert "test_f1" in src, "test_f1 missing!"
print("evaluate_full   : present")
print("All checks passed.")
