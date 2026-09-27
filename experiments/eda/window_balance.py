import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from utils import get_all_psv_files, get_results_dir, get_plots_dir

def calculate_entropy(p0, p1):
    if p0 == 0 or p1 == 0:
        return 0.0
    return - (p0 * np.log(p0) + p1 * np.log(p1))

def calculate_gini(p0, p1):
    return 1 - (p0**2 + p1**2)

def run_window_balance():
    print("Starting Window Balance Analysis...")
    files = get_all_psv_files()
    if not files:
        print("No PSV files found.")
        return

    plot_dir = os.path.join(get_plots_dir(), "window_balance")
    context_plot_dir = os.path.join(get_plots_dir(), "context")
    os.makedirs(plot_dir, exist_ok=True)
    os.makedirs(context_plot_dir, exist_ok=True)

    window_sizes = [6, 12, 24, 48, 72, 168, 336]
    
    # We will accumulate stats per window size
    window_stats = {ws: {
        "total_windows": 0,
        "any_positive_windows": 0,
        "no_positive_windows": 0,
        "total_positive_timesteps": 0,
        "total_negative_timesteps": 0,
        "positive_timesteps_in_pos_windows": [], # To compute mean/median
        "exactly_one_pos_windows": 0,
        "multiple_pos_windows": 0,
        "observed_values": 0,
        "possible_values": 0,
        "missing_values": 0
    } for ws in window_sizes}

    for f in tqdm(files, desc="Analyzing temporal windows"):
        df = pd.read_csv(f, sep='|')
        labels = df['SepsisLabel'].values
        seq_len = len(df)
        
        # dynamic features calculation for observation density
        features = df.drop(columns=['SepsisLabel', 'Age', 'Gender', 'Unit1', 'Unit2', 'HospAdmTime', 'ICULOS'], errors='ignore')
        
        for ws in window_sizes:
            # Create rolling windows (or non-overlapping? "Temporal window" usually implies sliding window in EDA for sequences)
            # Let's do non-overlapping chunks or sliding windows. Sliding windows produce massive amounts.
            # We'll use sliding windows with stride 1 to capture all contexts.
            
            if seq_len < ws:
                # Sequence shorter than window size, just take the whole sequence as one window?
                # For sliding window, we usually only take windows of exactly `ws`.
                continue
                
            for i in range(0, seq_len - ws + 1):
                window_labels = labels[i:i+ws]
                window_features = features.iloc[i:i+ws]
                
                pos_count = np.sum(window_labels)
                neg_count = ws - pos_count
                
                window_stats[ws]["total_windows"] += 1
                window_stats[ws]["total_positive_timesteps"] += pos_count
                window_stats[ws]["total_negative_timesteps"] += neg_count
                
                obs_count = window_features.notna().sum().sum()
                possible_count = window_features.size
                window_stats[ws]["observed_values"] += obs_count
                window_stats[ws]["possible_values"] += possible_count
                window_stats[ws]["missing_values"] += (possible_count - obs_count)
                
                if pos_count > 0:
                    window_stats[ws]["any_positive_windows"] += 1
                    window_stats[ws]["positive_timesteps_in_pos_windows"].append(pos_count)
                    if pos_count == 1:
                        window_stats[ws]["exactly_one_pos_windows"] += 1
                    else:
                        window_stats[ws]["multiple_pos_windows"] += 1
                else:
                    window_stats[ws]["no_positive_windows"] += 1

    # Compile final results
    results = []
    for ws in window_sizes:
        stats = window_stats[ws]
        tw = stats["total_windows"]
        if tw == 0:
            continue
            
        any_pos = stats["any_positive_windows"]
        no_pos = stats["no_positive_windows"]
        pt = stats["total_positive_timesteps"]
        nt = stats["total_negative_timesteps"]
        pos_ts_list = stats["positive_timesteps_in_pos_windows"]
        
        p1_ts = pt / (pt + nt) if (pt+nt) > 0 else 0
        p0_ts = nt / (pt + nt) if (pt+nt) > 0 else 0
        entropy = calculate_entropy(p0_ts, p1_ts)
        norm_entropy = entropy / np.log(2) if p1_ts > 0 else 0
        gini = calculate_gini(p0_ts, p1_ts)
        imbalance_ratio = nt / pt if pt > 0 else float('inf')
        
        obs_density = stats["observed_values"] / stats["possible_values"] if stats["possible_values"] > 0 else 0
        missingness = stats["missing_values"] / stats["possible_values"] if stats["possible_values"] > 0 else 0

        res_dict = {
            "Window Size": ws,
            "Total Windows": tw,
            "Positive Windows": any_pos,
            "Negative Windows": no_pos,
            "Positive Window %": (any_pos / tw) * 100,
            "Negative Window %": (no_pos / tw) * 100,
            "Positive Timesteps": pt,
            "Negative Timesteps": nt,
            "Positive Timestep %": p1_ts * 100,
            "Negative Timestep %": p0_ts * 100,
            "Imbalance Ratio": imbalance_ratio,
            "Mean Pos Timesteps per Pos Window": np.mean(pos_ts_list) if pos_ts_list else 0,
            "Median Pos Timesteps per Pos Window": np.median(pos_ts_list) if pos_ts_list else 0,
            "Completely Negative Fraction": no_pos / tw,
            "Exactly One Pos Fraction": stats["exactly_one_pos_windows"] / tw,
            "Multiple Pos Fraction": stats["multiple_pos_windows"] / tw,
            "Observation Density": obs_density,
            "Missingness": missingness,
            "Entropy": entropy,
            "Normalized Entropy": norm_entropy,
            "Gini": gini
        }
        results.append(res_dict)

    df_results = pd.DataFrame(results)
    
    # PLOTS
    # 1. window_class_balance.png
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df_results, x="Window Size", y="Positive Window %", marker='o')
    plt.title("Percentage of Windows Containing Sepsis-Positive Timesteps")
    plt.xlabel("Window Size (hours)")
    plt.ylabel("Positive Window %")
    plt.grid(True)
    plt.savefig(os.path.join(plot_dir, "window_class_balance.png"))
    plt.close()
    
    # 2. window_positive_prevalence.png
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df_results, x="Window Size", y="Positive Timestep %", marker='o')
    plt.title("Positive Timestep Percentage by Window Size")
    plt.xlabel("Window Size (hours)")
    plt.ylabel("Positive Timestep %")
    plt.grid(True)
    plt.savefig(os.path.join(plot_dir, "window_positive_prevalence.png"))
    plt.close()
    
    # 3. window_imbalance_ratio.png
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df_results, x="Window Size", y="Imbalance Ratio", marker='o')
    plt.title("Imbalance Ratio (Negative/Positive) by Window Size")
    plt.xlabel("Window Size (hours)")
    plt.ylabel("Imbalance Ratio")
    plt.yscale('log')
    plt.grid(True)
    plt.savefig(os.path.join(plot_dir, "window_imbalance_ratio.png"))
    plt.close()
    
    # 4. window_class_entropy.png
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df_results, x="Window Size", y="Normalized Entropy", marker='o')
    plt.title("Normalized Class Entropy by Window Size")
    plt.xlabel("Window Size (hours)")
    plt.ylabel("Normalized Entropy")
    plt.grid(True)
    plt.savefig(os.path.join(plot_dir, "window_class_entropy.png"))
    plt.close()
    
    # 5. window_gini.png
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df_results, x="Window Size", y="Gini", marker='o')
    plt.title("Gini Impurity by Window Size")
    plt.xlabel("Window Size (hours)")
    plt.ylabel("Gini Impurity")
    plt.grid(True)
    plt.savefig(os.path.join(plot_dir, "window_gini.png"))
    plt.close()
    
    # 6. window_balance_dashboard.png
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Window Balance Dashboard')
    
    sns.barplot(data=df_results, x="Window Size", y="Positive Window %", ax=axes[0,0])
    axes[0,0].set_title('Positive Window %')
    
    sns.barplot(data=df_results, x="Window Size", y="Positive Timestep %", ax=axes[0,1])
    axes[0,1].set_title('Positive Timestep %')
    
    sns.barplot(data=df_results, x="Window Size", y="Imbalance Ratio", ax=axes[0,2])
    axes[0,2].set_title('Imbalance Ratio')
    
    sns.barplot(data=df_results, x="Window Size", y="Entropy", ax=axes[1,0])
    axes[1,0].set_title('Entropy')
    
    sns.barplot(data=df_results, x="Window Size", y="Total Windows", ax=axes[1,1])
    axes[1,1].set_title('Total Windows')
    
    axes[1,2].axis('off') # empty space
    
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "window_balance_dashboard.png"))
    plt.close()
    
    # 7. window_information_quality.png
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df_results, x="Window Size", y="Observation Density", marker='o', label="Observation Density")
    sns.lineplot(data=df_results, x="Window Size", y="Missingness", marker='s', label="Missingness")
    plt.title("Window Information Quality")
    plt.xlabel("Window Size (hours)")
    plt.ylabel("Ratio")
    plt.grid(True)
    plt.savefig(os.path.join(context_plot_dir, "window_information_quality.png"))
    plt.close()
    
    # 8. transformer_context_analysis
    transformer_df = df_results[['Window Size', 'Positive Windows', 'Positive Window %', 'Observation Density', 'Missingness', 'Entropy', 'Imbalance Ratio']]
    transformer_df.to_csv(os.path.join(get_results_dir(), "transformer_context_analysis.csv"), index=False)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Transformer Context Analysis (EDA-based context sensitivity analysis)')
    sns.lineplot(data=transformer_df, x="Window Size", y="Positive Window %", ax=axes[0], marker='o')
    sns.lineplot(data=transformer_df, x="Window Size", y="Observation Density", ax=axes[1], marker='o')
    sns.lineplot(data=transformer_df, x="Window Size", y="Entropy", ax=axes[2], marker='o')
    plt.tight_layout()
    plt.savefig(os.path.join(context_plot_dir, "transformer_context_analysis.png"))
    plt.close()

    print("Window Balance Analysis Completed.")

if __name__ == "__main__":
    run_window_balance()
