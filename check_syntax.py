import ast
with open("experiments/gat/graph_builder_v2.py", encoding="utf-8") as f:
    src = f.read()
ast.parse(src)
print("Syntax OK")

# Also check no isolated nodes will exist for Graph F
# by checking all 35 features appear in clinical priors
import sys, os
sys.path.insert(0, ".")

# Extract CLINICAL_PRIOR_EDGES by running the module
import importlib.util
spec = importlib.util.spec_from_file_location(
    "graph_builder_v2", "experiments/gat/graph_builder_v2.py"
)
gb = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gb)

import json
with open("artifacts/preprocessing_config.json") as f:
    cfg = json.load(f)
features = cfg["dynamic_features"]
feat_set  = set(features)

# Which features appear in clinical priors
in_priors = set()
for a, b in gb.CLINICAL_PRIOR_EDGES:
    in_priors.add(a)
    in_priors.add(b)

missing = feat_set - in_priors - {"ICULOS"}
print(f"Total clinical prior pairs : {len(gb.CLINICAL_PRIOR_EDGES)}")
print(f"Features covered by priors : {len(in_priors)}")
print(f"Features NOT in any prior  : {sorted(missing)}")
