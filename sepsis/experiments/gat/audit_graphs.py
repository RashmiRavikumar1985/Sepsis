import pandas as pd

for gid in ["B", "F"]:
    edges = pd.read_csv(f"experiments/gat/graphs/graph_{gid}/edges.csv")
    nodes = pd.read_csv(f"experiments/gat/graphs/graph_{gid}/nodes.csv")

    connected = set(edges.source.tolist() + edges.target.tolist())
    all_nodes = set(nodes.node_id.tolist())
    isolated  = all_nodes - connected

    print(f"=== Graph {gid} ===")
    print(f"  Total nodes    : {len(nodes)}")
    print(f"  Connected nodes: {len(connected)}")
    print(f"  Isolated nodes : {len(isolated)}")
    for n in sorted(isolated):
        print(f"    [{n}] {nodes.iloc[n].feature_name}")

    deg = edges.groupby("source").size().reset_index(name="degree")
    deg["feature"] = deg.source.apply(lambda x: nodes.iloc[x].feature_name)
    low = deg[deg["degree"] <= 2].sort_values("degree")
    print(f"  Low-degree nodes (<=2):")
    for _, r in low.iterrows():
        print(f"    [{int(r.source)}] {r.feature:<25} degree={int(r.degree)}")
    print()
