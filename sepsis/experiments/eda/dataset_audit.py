import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from utils import get_all_psv_files, get_results_dir, get_plots_dir, get_config

def run_audit():
    print("Starting Dataset Audit...")
    files = get_all_psv_files()
    if not files:
        print("No PSV files found. Exiting audit.")
        return
        
    config = get_config()
    if config:
        dyn_feats = config['dynamic_features']
        stat_feats = config['static_features']
    else:
        # Fallback if config doesn't exist
        stat_feats = ['Age', 'Gender', 'Unit1', 'Unit2', 'HospAdmTime']
        dyn_feats = [
            'HR', 'O2Sat', 'Temp', 'SBP', 'MAP', 'DBP', 'Resp', 'BaseExcess',
            'HCO3', 'FiO2', 'pH', 'PaCO2', 'SaO2', 'AST', 'BUN', 'Alkalinephos',
            'Calcium', 'Chloride', 'Creatinine', 'Bilirubin_direct', 'Glucose',
            'Lactate', 'Magnesium', 'Phosphate', 'Potassium', 'Bilirubin_total',
            'TroponinI', 'Hct', 'Hgb', 'PTT', 'WBC', 'Fibrinogen', 'Platelets',
            'EtCO2', 'ICULOS'
        ]

    stats = {
        'num_patient_files': len(files),
        'num_patients': len(files),
        'num_rows': 0,
        'seq_lengths': [],
        'num_dynamic_features': len(dyn_feats),
        'num_static_features': len(stat_feats),
        'total_observations': 0,
        'missing_observations': 0,
        'positive_sepsis_hours': 0,
        'negative_sepsis_hours': 0,
        'patients_with_sepsis': 0,
        'patients_without_sepsis': 0
    }
    
    feature_counts = {f: 0 for f in dyn_feats}
    feature_missing = {f: 0 for f in dyn_feats}

    for f in tqdm(files, desc="Auditing files"):
        df = pd.read_csv(f, sep='|')
        seq_len = len(df)
        stats['num_rows'] += seq_len
        stats['seq_lengths'].append(seq_len)
        
        has_sepsis = df['SepsisLabel'].max() > 0
        if has_sepsis:
            stats['patients_with_sepsis'] += 1
        else:
            stats['patients_without_sepsis'] += 1
            
        pos = df['SepsisLabel'].sum()
        stats['positive_sepsis_hours'] += pos
        stats['negative_sepsis_hours'] += (seq_len - pos)
        
        for feat in dyn_feats:
            if feat in df.columns:
                valid_count = df[feat].count()
                missing = seq_len - valid_count
                feature_counts[feat] += valid_count
                feature_missing[feat] += missing
                stats['total_observations'] += valid_count
                stats['missing_observations'] += missing
            else:
                feature_missing[feat] += seq_len
                stats['missing_observations'] += seq_len

    # Sequence length stats
    seq_arr = np.array(stats['seq_lengths'])
    stats['min_sequence_length'] = int(np.min(seq_arr))
    stats['max_sequence_length'] = int(np.max(seq_arr))
    stats['mean_sequence_length'] = float(np.mean(seq_arr))
    stats['median_sequence_length'] = float(np.median(seq_arr))
    stats['std_sequence_length'] = float(np.std(seq_arr))
    
    total_cells = stats['total_observations'] + stats['missing_observations']
    stats['observation_percentage'] = float(stats['total_observations'] / total_cells * 100) if total_cells > 0 else 0.0

    # Save outputs
    res_dir = get_results_dir()
    plot_dir = get_plots_dir()
    
    # JSON
    with open(os.path.join(res_dir, "dataset_summary.json"), 'w') as f:
        # Exclude raw list from json for readability
        save_stats = {k: v for k, v in stats.items() if k != 'seq_lengths'}
        json.dump(save_stats, f, indent=4)
        
    # CSV
    pd.DataFrame([save_stats]).to_csv(os.path.join(res_dir, "dataset_summary.csv"), index=False)
    
    # MD
    with open(os.path.join(res_dir, "dataset_audit.md"), 'w') as f:
        f.write("# Dataset Audit Summary\n\n")
        for k, v in save_stats.items():
            f.write(f"- **{k}**: {v}\n")
            
    # Plots
    # 1. Histogram
    plt.figure(figsize=(10, 6))
    sns.histplot(seq_arr, bins=50, kde=True)
    plt.title("Patient Sequence Length Distribution")
    plt.xlabel("Sequence Length (hours)")
    plt.ylabel("Count")
    plt.savefig(os.path.join(plot_dir, "seq_length_hist.png"))
    plt.close()
    
    # 2. Boxplot
    plt.figure(figsize=(10, 6))
    sns.boxplot(x=seq_arr)
    plt.title("Patient Sequence Length Boxplot")
    plt.xlabel("Sequence Length (hours)")
    plt.savefig(os.path.join(plot_dir, "seq_length_box.png"))
    plt.close()
    
    # 3 & 4. Features
    feats = list(feature_counts.keys())
    counts = list(feature_counts.values())
    missing = list(feature_missing.values())
    total_f = np.array(counts) + np.array(missing)
    missing_pct = np.array(missing) / (total_f + 1e-9) * 100
    
    plt.figure(figsize=(12, 8))
    sns.barplot(x=counts, y=feats, orient='h')
    plt.title("Number of Observations per Feature")
    plt.xlabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "feature_observations.png"))
    plt.close()
    
    plt.figure(figsize=(12, 8))
    sns.barplot(x=missing_pct, y=feats, orient='h')
    plt.title("Missingness Percentage per Feature")
    plt.xlabel("Missing Percentage (%)")
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "feature_missingness.png"))
    plt.close()
    
    print("Dataset Audit Completed.")

if __name__ == "__main__":
    run_audit()
