import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests
from tqdm import tqdm
from utils import get_all_psv_files, get_results_dir, get_plots_dir

def run_feature_stats():
    print("Starting Feature Statistics Analysis...")
    files = get_all_psv_files()
    if not files:
        print("No PSV files found.")
        return

    res_dir = get_results_dir()
    feature_plot_dir = os.path.join(get_plots_dir(), "features")
    os.makedirs(feature_plot_dir, exist_ok=True)

    # We need to accumulate data for stats
    # For large datasets, we can compute running stats, or just load everything if it fits.
    # The dataset might be large, so we should collect data in chunks or just list arrays.
    
    # Let's accumulate arrays for important features to plot, and summary stats for all.
    important_features = ['HR', 'O2Sat', 'Temp', 'SBP', 'MAP', 'DBP', 'Resp', 'WBC', 'Lactate', 'Creatinine', 'Bilirubin', 'pH']
    
    # A list of dataframes might be too big, so we will do a fast pass to collect data.
    # We will sample or collect all if memory permits.
    all_data_pos = []
    all_data_neg = []
    
    for f in tqdm(files, desc="Loading data for feature stats"):
        df = pd.read_csv(f, sep='|')
        # separate by label
        pos_mask = df['SepsisLabel'] == 1
        
        if pos_mask.any():
            all_data_pos.append(df[pos_mask])
        if (~pos_mask).any():
            all_data_neg.append(df[~pos_mask])

    if all_data_pos:
        df_pos = pd.concat(all_data_pos, ignore_index=True)
    else:
        df_pos = pd.DataFrame()
        
    if all_data_neg:
        df_neg = pd.concat(all_data_neg, ignore_index=True)
    else:
        df_neg = pd.DataFrame()

    df_all = pd.concat([df_pos, df_neg], ignore_index=True)

    dynamic_features = df_all.drop(columns=['SepsisLabel', 'Age', 'Gender', 'Unit1', 'Unit2', 'HospAdmTime', 'ICULOS'], errors='ignore').columns

    # 1. Feature Statistics (Part 10)
    stats_list = []
    
    for f in dynamic_features:
        for group, df_group, label in [("All", df_all, "All"), ("Sepsis", df_pos, "Sepsis-Positive"), ("Non-Sepsis", df_neg, "Sepsis-Negative")]:
            if df_group.empty or f not in df_group.columns:
                continue
                
            s = df_group[f].dropna()
            count = len(s)
            missing = df_group[f].isna().sum()
            total = count + missing
            obs_rate = count / total if total > 0 else 0
            
            if count > 0:
                stats_list.append({
                    "Feature": f,
                    "Class": label,
                    "Count": count,
                    "Missing Count": missing,
                    "Observation Rate": obs_rate,
                    "Mean": s.mean(),
                    "Median": s.median(),
                    "Std": s.std(),
                    "Variance": s.var(),
                    "Min": s.min(),
                    "Max": s.max(),
                    "Q1": s.quantile(0.25),
                    "Q3": s.quantile(0.75),
                    "IQR": s.quantile(0.75) - s.quantile(0.25),
                    "1st Percentile": s.quantile(0.01),
                    "99th Percentile": s.quantile(0.99)
                })

    df_stats = pd.DataFrame(stats_list)
    df_stats.to_csv(os.path.join(res_dir, "feature_statistics.csv"), index=False)

    # 2. Feature Boxplots (Part 11)
    plot_features = [f for f in important_features if f in dynamic_features]
    
    if plot_features and not df_pos.empty and not df_neg.empty:
        # We need a long format df for seaborn
        df_all_subset = df_all[plot_features + ['SepsisLabel']].copy()
        df_long = df_all_subset.melt(id_vars='SepsisLabel', var_name='Feature', value_name='Value').dropna()
        
        plt.figure(figsize=(15, 10))
        sns.boxplot(data=df_long, x='Feature', y='Value', hue='SepsisLabel')
        plt.title('Feature Boxplots (Sepsis vs Non-Sepsis)')
        plt.xticks(rotation=45)
        plt.savefig(os.path.join(feature_plot_dir, "feature_boxplots.png"))
        plt.close()
        
        # 3. Feature Distributions (KDE)
        plt.figure(figsize=(20, 15))
        for i, f in enumerate(plot_features, 1):
            plt.subplot(3, 4, i)
            sns.kdeplot(data=df_all_subset, x=f, hue='SepsisLabel', common_norm=False, fill=True)
            plt.title(f"Distribution of {f}")
        plt.tight_layout()
        plt.savefig(os.path.join(feature_plot_dir, "feature_distributions.png"))
        plt.close()

        # 4. Feature Variance comparison
        var_pos = df_pos[plot_features].var()
        var_neg = df_neg[plot_features].var()
        df_var = pd.DataFrame({'Sepsis': var_pos, 'Non-Sepsis': var_neg}).reset_index()
        df_var = df_var.melt(id_vars='index', var_name='Class', value_name='Variance')
        
        plt.figure(figsize=(12, 6))
        sns.barplot(data=df_var, x='index', y='Variance', hue='Class')
        plt.title('Feature Variance (Sepsis vs Non-Sepsis)')
        plt.yscale('log')
        plt.xticks(rotation=45)
        plt.savefig(os.path.join(feature_plot_dir, "feature_variance.png"))
        plt.close()

    # 5. Feature vs Sepsis Association (Part 13)
    assoc_list = []
    pvals = []
    
    for f in dynamic_features:
        s_pos = df_pos[f].dropna()
        s_neg = df_neg[f].dropna()
        
        if len(s_pos) > 0 and len(s_neg) > 0:
            # Mann-Whitney U
            try:
                stat, p = mannwhitneyu(s_pos, s_neg, alternative='two-sided')
                assoc_list.append({
                    "Feature": f,
                    "Statistic (U)": stat,
                    "Raw p-value": p
                })
                pvals.append(p)
            except ValueError:
                pass

    if assoc_list:
        df_assoc = pd.DataFrame(assoc_list)
        # Benjamini-Hochberg correction
        reject, pvals_corrected, _, _ = multipletests(pvals, alpha=0.05, method='fdr_bh')
        df_assoc['Corrected p-value'] = pvals_corrected
        df_assoc['Significant'] = reject
        
        df_assoc.to_csv(os.path.join(res_dir, "feature_sepsis_association.csv"), index=False)
        
        # Plot -log10 p-value
        df_assoc['-log10(p)'] = -np.log10(df_assoc['Corrected p-value'] + 1e-300)
        df_assoc = df_assoc.sort_values('-log10(p)', ascending=False)
        
        plt.figure(figsize=(10, 8))
        sns.barplot(data=df_assoc.head(20), x='-log10(p)', y='Feature')
        plt.title('Feature Association with Sepsis (Top 20 by -log10 Corrected p-value)')
        plt.axvline(x=-np.log10(0.05), color='r', linestyle='--')
        plt.savefig(os.path.join(feature_plot_dir, "feature_sepsis_effect.png"))
        plt.close()

    print("Feature Statistics Analysis Completed.")

if __name__ == "__main__":
    run_feature_stats()
