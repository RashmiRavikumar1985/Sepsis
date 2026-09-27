import json
with open("artifacts/splits.json") as f:
    splits = json.load(f)
train = len(splits["train"])
val   = len(splits["val"])
test  = len(splits["test"])
total = train + val + test
print(f"Train patients : {train:,}")
print(f"Val patients   : {val:,}")
print(f"Test patients  : {test:,}")
print(f"Total          : {total:,}")
print()
print("Only ~6,051 patients used for evaluation = test split only.")
print("Training uses all splits combined for model development.")
