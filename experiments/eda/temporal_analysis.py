import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from utils import get_all_psv_files, get_results_dir, get_plots_dir

def run_temporal_analysis():
    print("Starting Temporal Analysis...")
    files = get_all_psv_files()
    if not files:
        print("No PSV files found.")
        return

    res_dir = get_results_dir()
    temporal_plot_dir = os.path.join(get_plots_dir(), "temporal")
    os.makedirs(temporal_plot_dir, exist_ok=True)

    # Accumulators
    first_pos_times = []
    norm_first_pos_positions = []
    
    seq_len_pos = []
    seq_len_neg = []
    
    pos_density_list = []
    
    obs_density_pos = []
    obs_density_neg = []
    
    # Pre-sepsis
    pre_windows = [6, 12, 24, 48]
    pre_stats = []

    for f in tqdm(files, desc="Analyzing temporal distributions"):
        df = pd.read_csv(f, sep='|')
        seq_len = len(df)
        labels = df['SepsisLabel'].values
        features = df.drop(columns=['SepsisLabel', 'Age', 'Gender', 'Unit1', 'Unit2', 'HospAdmTime', 'ICULOS'], errors='ignore')
        
        pos_count = np.sum(labels)
        
        obs_count = features.notna().sum().sum()
        possible_count = features.size
        obs_density = obs_count / possible_count if possible_count > 0 else 0
        
        if pos_count > 0:
            first_pos_idx = np.argmax(labels == 1)
            first_pos_times.append(first_pos_idx)
            norm_first_pos_positions.append(first_pos_idx / seq_len if seq_len > 0 else 0)
            seq_len_pos.append(seq_len)
            pos_density_list.append((seq_len, pos_count / seq_len))
            obs_density_pos.append(obs_density)
            
            # Pre-sepsis window analysis
            for w in pre_windows:
                if first_pos_idx >= w:
                    w_features = features.iloc[first_pos_idx - w : first_pos_idx]
                    w_obs = w_features.notna().sum().sum()
                    w_poss = w_features.size
                    w_var = w_features.var().mean() if w_features.shape[0] > 1 else 0
                    
                    pre_stats.append({
                        "Window": f"{w}h",
                        "Available Observations": w_obs,
                        "Missingness": (w_poss - w_obs) / w_poss if w_poss > 0 else 0,
                        "Feature Variance": w_var,
                        "Sequence Length": seq_len
                    })
                
        else:
            seq_len_neg.append(seq_len)
            obs_density_neg.append(obs_density)

    # 1. Temporal distribution plots (Part 8)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('Sepsis Temporal Distribution')
    sns.histplot(first_pos_times, bins=30, ax=axes[0], kde=True)
    axes[0].set_title('First Positive Time (Hours)')
    axes[0].set_xlabel('Hours')
    
    sns.histplot(norm_first_pos_positions, bins=30, ax=axes[1], kde=True)
    axes[1].set_title('Normalized First Positive Position')
    axes[1].set_xlabel('Position Ratio')
    plt.savefig(os.path.join(temporal_plot_dir, "sepsis_temporal_distribution.png"))
    plt.close()

    # 2. Sequence Length vs Sepsis (Part 17)
    plt.figure(figsize=(8, 6))
    sns.boxplot(data=[seq_len_neg, seq_len_pos])
    plt.xticks([0, 1], ['Negative', 'Positive'])
    plt.title('Sequence Length by Class')
    plt.ylabel('Sequence Length (Hours)')
    plt.savefig(os.path.join(temporal_plot_dir, "sequence_length_by_class.png"))
    plt.close()
    
    if pos_density_list:
        sl, pdens = zip(*pos_density_list)
        plt.figure(figsize=(8, 6))
        sns.scatterplot(x=sl, y=pdens)
        plt.title('Sequence Length vs Positive Density (Sepsis Patients)')
        plt.xlabel('Sequence Length')
        plt.ylabel('Positive Label Density')
        plt.savefig(os.path.join(temporal_plot_dir, "sequence_length_vs_positive_density.png"))
        plt.close()

    # 3. Observation Density (Part 18)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sns.histplot(obs_density_neg, color='blue', label='Negative', kde=True, ax=axes[0], stat='density', common_norm=False)
    sns.histplot(obs_density_pos, color='red', label='Positive', kde=True, ax=axes[0], stat='density', common_norm=False)
    axes[0].set_title('Observation Density Distribution by Class')
    axes[0].set_xlabel('Observation Density')
    axes[0].legend()
    
    sns.boxplot(data=[obs_density_neg, obs_density_pos], ax=axes[1])
    axes[1].set_xticks([0, 1])
    axes[1].set_xticklabels(['Negative', 'Positive'])
    axes[1].set_title('Observation Density by Class')
    plt.savefig(os.path.join(temporal_plot_dir, "observation_density_by_class.png"))
    plt.close()
    
    # 4. Pre-sepsis stats (Part 9)
    if pre_stats:
        df_pre = pd.DataFrame(pre_stats)
        df_pre.to_csv(os.path.join(res_dir, "presepsis_window_statistics.csv"), index=False)

    print("Temporal Analysis Completed.")

if __name__ == "__main__":
    run_temporal_analysis()
