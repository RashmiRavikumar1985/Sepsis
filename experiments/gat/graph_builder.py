import os
import sys
import json
import pandas as pd
import numpy as np

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

def build_provisional_graph():
    print("=== Constructing Provisional Clinical Feature Interaction Graph ===")
    
    # 1. Load config to get the dynamic features
    config_path = os.path.join(project_root, "artifacts", "preprocessing_config.json")
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    features = config['dynamic_features']
    print(f"Nodes (Features): {len(features)}")
    
    # We will sample training files (from splits['train'] ONLY) to compute correlation without leakage
    splits_path = os.path.join(project_root, "artifacts", "splits.json")
    with open(splits_path, 'r') as f:
        splits = json.load(f)
    
    train_files = splits['train'][:1000] # Sample 1000 training files for fast correlation calculation
    data_dirs = [
        os.path.join(project_root, "training", "training_setA"),
        os.path.join(project_root, "training", "training_setB")
    ]
    
    all_dfs = []
    for f in train_files:
        for d in data_dirs:
            p = os.path.join(d, f)
            if os.path.exists(p):
                df = pd.read_csv(p, sep='|')[features]
                all_dfs.append(df)
                break
                
    if not all_dfs:
        print("Error: Could not load any training files.")
        return
        
    combined = pd.concat(all_dfs)
    
    print("Computing Pearson Correlation Matrix...")
    corr_matrix = combined.corr().fillna(0)
    
    nodes = [{"node_id": i, "feature_name": feat} for i, feat in enumerate(features)]
    edges = []
    
    # Threshold for creating edges
    corr_threshold = 0.2
    
    for i in range(len(features)):
        for j in range(i + 1, len(features)):
            weight = abs(corr_matrix.iloc[i, j])
            if weight > corr_threshold:
                edges.append({
                    "source": i,
                    "target": j,
                    "weight": weight,
                    "type": "correlation"
                })
                # Undirected graph
                edges.append({
                    "source": j,
                    "target": i,
                    "weight": weight,
                    "type": "correlation"
                })
                
    graph_data = {
        "nodes": nodes,
        "edges": edges,
        "metadata": {
            "description": "Provisional Clinical Feature Interaction Graph based on Pearson correlation threshold > 0.2",
            "edge_count": len(edges)
        }
    }
    
    out_dir = os.path.join(project_root, "experiments", "gat")
    
    # Save JSON
    with open(os.path.join(out_dir, "graph.json"), "w") as f:
        json.dump(graph_data, f, indent=4)
        
    # Save CSVs
    pd.DataFrame(nodes).to_csv(os.path.join(out_dir, "nodes.csv"), index=False)
    pd.DataFrame(edges).to_csv(os.path.join(out_dir, "edges.csv"), index=False)
    
    print(f"Generated {len(edges)} edges among {len(nodes)} nodes.")
    print("Graph saved to experiments/gat/")

if __name__ == "__main__":
    build_provisional_graph()
