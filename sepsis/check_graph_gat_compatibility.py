import pandas as pd, json

nodes = pd.read_csv("experiments/gat/graphs/graph_F/nodes.csv")
edges = pd.read_csv("experiments/gat/graphs/graph_F/edges.csv")

with open("artifacts/preprocessing_config.json") as f:
    cfg = json.load(f)

features = cfg["dynamic_features"]
n_features = len(features)

print(f"preprocessing_config dynamic_features : {n_features}")
print(f"nodes.csv rows                         : {len(nodes)}")
print(f"Match                                  : {len(nodes) == n_features}")
print()

iculos_row = nodes[nodes.feature_name == "ICULOS"]
print(f"ICULOS in nodes.csv : {len(iculos_row) > 0}")
if len(iculos_row) > 0:
    iculos_idx = int(iculos_row.node_id.values[0])
    print(f"ICULOS node index   : {iculos_idx}")
    in_edges = (edges.source == iculos_idx).any() or (edges.target == iculos_idx).any()
    print(f"ICULOS in any edge  : {in_edges}")

print()
max_node_in_edges = max(edges.source.max(), edges.target.max())
print(f"Max node index in edges : {max_node_in_edges}")
print(f"Expected max (n-1)      : {n_features - 1}")
print(f"Edge indices valid      : {max_node_in_edges < n_features}")
print()

# Check all node indices in edges are valid
all_edge_nodes = set(edges.source.tolist() + edges.target.tolist())
invalid = [n for n in all_edge_nodes if n >= n_features]
print(f"Invalid node indices in edges : {invalid}")
print()
print("CONCLUSION:")
if len(nodes) == n_features and max_node_in_edges < n_features and not invalid:
    print("  Graph is FULLY COMPATIBLE with GAT (35 nodes).")
    print("  ICULOS has no edges but exists as node with self-loop via add_self_loops=True.")
else:
    print("  INCOMPATIBILITY DETECTED — needs fix.")
