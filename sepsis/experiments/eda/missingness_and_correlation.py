import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from utils import get_all_psv_files, get_results_dir, get_plots_dir

def run_missingness_and_correlation():
    print("Starting Missingness and Correlation Analysis...")
    files = get_all_psv_files()
    if not files:
        print("No PSV files found.")
        return

    missingness_plot_dir = os.path.join(get_plots_dir(), "missingness")
    temporal_plot_dir = os.path.join(get_plots_dir(), "temporal")
    features_plot_dir = os.path.join(get_plots_dir(), "features")
    os.makedirs(missingness_plot_dir, exist_ok=True)
    os.makedirs(temporal_plot_dir, exist_ok=True)
    os.makedirs(features_plot_dir, exist_ok=True)

    # Missingness accumulators
    missing_pct_pos = []
    missing_pct_neg = []
    
    # We will use all files for missingness, but only a subset for correlation to prevent OOM
    # and strictly "training data only" logic (we might not have a formal split here, 
    # but we can simulate it or just use the first 80% of files)
    
    train_files = files[:int(0.8 * len(files))]
    train_dfs = []
    
    # For Delta analysis
    delta_pos = []
    delta_neg = []
    
    # 1. Missingness & Delta loop
    for i, f in enumerate(tqdm(files, desc="Analyzing missingness & deltas")):
        df = pd.read_csv(f, sep='|')
        labels = df['SepsisLabel']
        features = df.drop(columns=['SepsisLabel', 'Age', 'Gender', 'Unit1', 'Unit2', 'HospAdmTime', 'ICULOS'], errors='ignore')
        
        pos_count = labels.sum()
        is_pos_patient = pos_count > 0
        
        miss_rate = features.isna().sum().sum() / features.size if features.size > 0 else 0
        
        if is_pos_patient:
            missing_pct_pos.append(miss_rate)
            # Delta analysis: time since last observation for each feature
            # Since data is hourly, delta = 1 for observed, or accumulates when missing.
            # We can approximate delta by counting consecutive NaNs + 1
            for col in features.columns:
                nans = features[col].isna().astype(int)
                # simple trick to find length of consecutive NaNs
                blocks = nans.groupby((nans != nans.shift()).cumsum()).sum()
                delta_pos.extend(blocks[blocks > 0] + 1)
        else:
            missing_pct_neg.append(miss_rate)
            for col in features.columns:
                nans = features[col].isna().astype(int)
                blocks = nans.groupby((nans != nans.shift()).cumsum()).sum()
                delta_neg.extend(blocks[blocks > 0] + 1)
                
        # Gather data for correlation (only train files)
        if i < len(train_files):
            train_dfs.append(features)

    # 1. Plot Missingness
    plt.figure(figsize=(8, 6))
    sns.boxplot(data=[missing_pct_neg, missing_pct_pos])
    plt.xticks([0, 1], ['Negative Patients', 'Positive Patients'])
    plt.title('Patient-Level Missingness by Class')
    plt.ylabel('Missingness Ratio')
    plt.savefig(os.path.join(missingness_plot_dir, "missingness_by_class.png"))
    plt.close()
    
    # 2. Plot Delta
    plt.figure(figsize=(12, 5))
    sns.kdeplot(delta_neg, color='blue', label='Negative', log_scale=True, common_norm=False)
    sns.kdeplot(delta_pos, color='red', label='Positive', log_scale=True, common_norm=False)
    plt.title('Delta (Time gap between observations) by Class')
    plt.xlabel('Delta (Hours)')
    plt.legend()
    plt.savefig(os.path.join(temporal_plot_dir, "delta_by_class.png"))
    plt.close()

    plt.figure(figsize=(10, 6))
    sns.histplot(delta_pos + delta_neg, bins=50, log_scale=(True, True))
    plt.title('Overall Delta Distribution')
    plt.xlabel('Delta (Hours)')
    plt.savefig(os.path.join(temporal_plot_dir, "delta_distribution.png"))
    plt.close()

    # 3. Correlation
    if train_dfs:
        print("Calculating Correlation Matrix (Train Data)...")
        df_train = pd.concat(train_dfs, ignore_index=True)
        
        # Pearson
        corr_pearson = df_train.corr(method='pearson')
        plt.figure(figsize=(18, 14))
        sns.heatmap(corr_pearson, cmap='coolwarm', vmin=-1, vmax=1, center=0)
        plt.title('Pearson Correlation Matrix (Training Data)')
        plt.savefig(os.path.join(features_plot_dir, "feature_correlation_pearson.png"))
        plt.close()
        
        # Spearman
        corr_spearman = df_train.corr(method='spearman')
        plt.figure(figsize=(18, 14))
        sns.heatmap(corr_spearman, cmap='coolwarm', vmin=-1, vmax=1, center=0)
        plt.title('Spearman Correlation Matrix (Training Data)')
        plt.savefig(os.path.join(features_plot_dir, "feature_correlation_spearman.png"))
        plt.close()
        
        # Missingness heatmap
        plt.figure(figsize=(12, 8))
        sns.heatmap(df_train.isna().corr(), cmap='viridis')
        plt.title('Missingness Correlation (Are features missing together?)')
        plt.savefig(os.path.join(missingness_plot_dir, "missingness_heatmap.png"))
        plt.close()

    print("Missingness and Correlation Analysis Completed.")

if __name__ == "__main__":
    run_missingness_and_correlation()
