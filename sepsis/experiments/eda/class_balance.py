import os
import json
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

def run_class_balance():
    print("Starting Extended Class Balance Analysis...")
    files = get_all_psv_files()
    if not files:
        print("No PSV files found.")
        return

    res_dir = get_results_dir()
    plot_dir = os.path.join(get_plots_dir(), "class_balance")
    os.makedirs(plot_dir, exist_ok=True)

    # LEVEL 1 & 2 stats
    total_samples = 0
    positive_samples = 0
    negative_samples = 0
    
    patient_positive_count = 0
    patient_negative_count = 0

    # LEVEL 3 & 4 stats collections
    icu_stay_positive_hours = []
    icu_stay_negative_hours = []
    icu_stay_positive_percentages = []
    
    time_before_first_positive = []
    num_positive_timesteps = []
    num_negative_before_positive = []
    sequence_lengths_sepsis = []
    positive_label_density = []

    for f in tqdm(files, desc="Analyzing class balance"):
        df = pd.read_csv(f, sep='|')
        seq_len = len(df)
        pos = df['SepsisLabel'].sum()
        neg = seq_len - pos
        
        total_samples += seq_len
        positive_samples += pos
        negative_samples += neg
        
        # Level 3
        icu_stay_positive_hours.append(pos)
        icu_stay_negative_hours.append(neg)
        icu_stay_positive_percentages.append(pos / seq_len if seq_len > 0 else 0)
        
        # Level 4
        if pos > 0:
            patient_positive_count += 1
            first_pos_idx = df.index[df['SepsisLabel'] == 1].tolist()[0]
            
            time_before_first_positive.append(first_pos_idx) # Assuming 1 hr per timestep
            num_positive_timesteps.append(pos)
            num_negative_before_positive.append(first_pos_idx)
            sequence_lengths_sepsis.append(seq_len)
            positive_label_density.append(pos / seq_len)
        else:
            patient_negative_count += 1

    total_patients = patient_positive_count + patient_negative_count
    
    p1_pt = patient_positive_count / total_patients if total_patients > 0 else 0
    p0_pt = patient_negative_count / total_patients if total_patients > 0 else 0
    
    p1_ts = positive_samples / total_samples if total_samples > 0 else 0
    p0_ts = negative_samples / total_samples if total_samples > 0 else 0
    
    # Entropy & Gini
    entropy = calculate_entropy(p0_ts, p1_ts)
    norm_entropy = entropy / np.log(2)
    gini = calculate_gini(p0_ts, p1_ts)
    
    imbalance_ratio = negative_samples / positive_samples if positive_samples > 0 else float('inf')
    pos_weight = imbalance_ratio

    results = {
        "Level 1: Patient": {
            "total_patients": total_patients,
            "positive_patients": patient_positive_count,
            "negative_patients": patient_negative_count,
            "positive_percentage": p1_pt * 100,
            "negative_percentage": p0_pt * 100,
            "positive_negative_ratio": p1_pt / p0_pt if p0_pt > 0 else 0,
            "imbalance_ratio": p0_pt / p1_pt if p1_pt > 0 else float('inf')
        },
        "Level 2: Timestep": {
            "total_timesteps": total_samples,
            "positive_timesteps": positive_samples,
            "negative_timesteps": negative_samples,
            "positive_percentage": p1_ts * 100,
            "negative_percentage": p0_ts * 100,
            "imbalance_ratio": imbalance_ratio,
            "pos_weight": pos_weight
        },
        "Level 4: Sepsis Patients": {
            "mean_time_before_first_positive": float(np.mean(time_before_first_positive)) if time_before_first_positive else 0,
            "mean_positive_timesteps": float(np.mean(num_positive_timesteps)) if num_positive_timesteps else 0,
            "mean_negative_before_positive": float(np.mean(num_negative_before_positive)) if num_negative_before_positive else 0,
            "mean_sequence_length": float(np.mean(sequence_lengths_sepsis)) if sequence_lengths_sepsis else 0,
            "mean_positive_label_density": float(np.mean(positive_label_density)) if positive_label_density else 0
        },
        "Class Balance Quality": {
            "entropy": entropy,
            "normalized_entropy": norm_entropy,
            "gini_impurity": gini,
            "effective_imbalance_ratio": imbalance_ratio
        }
    }

    # Save outputs
    with open(os.path.join(res_dir, "class_balance_extended.json"), 'w') as f:
        json.dump(results, f, indent=4)

    # Flatten for CSV
    flat_results = []
    for k, v in results.items():
        for sub_k, sub_v in v.items():
            flat_results.append({"Category": k, "Metric": sub_k, "Value": sub_v})
            
    pd.DataFrame(flat_results).to_csv(os.path.join(res_dir, "class_balance_extended.csv"), index=False)

    md_content = f"""# Extended Class Balance Analysis

## Level 1: Patient
- Total Patients: {total_patients}
- Sepsis-Positive: {patient_positive_count} ({p1_pt*100:.2f}%)
- Sepsis-Negative: {patient_negative_count} ({p0_pt*100:.2f}%)
- Imbalance Ratio: {results["Level 1: Patient"]["imbalance_ratio"]:.2f}

## Level 2: Timestep
- Total Timesteps: {total_samples}
- Positive Timesteps: {positive_samples} ({p1_ts*100:.2f}%)
- Negative Timesteps: {negative_samples} ({p0_ts*100:.2f}%)
- Imbalance Ratio: {imbalance_ratio:.2f}

## Level 4: Sepsis Patients Only
- Mean time before first positive: {results["Level 4: Sepsis Patients"]["mean_time_before_first_positive"]:.2f} hours
- Mean positive timesteps: {results["Level 4: Sepsis Patients"]["mean_positive_timesteps"]:.2f}
- Mean sequence length: {results["Level 4: Sepsis Patients"]["mean_sequence_length"]:.2f}

## Class Balance Quality
- Entropy: {entropy:.4f}
- Normalized Entropy: {norm_entropy:.4f}
- Gini Impurity: {gini:.4f}
"""
    with open(os.path.join(res_dir, "class_balance_extended.md"), 'w') as f:
        f.write(md_content)

    # Plots
    plt.figure(figsize=(8, 6))
    sns.barplot(x=['Negative (0)', 'Positive (1)'], y=[patient_negative_count, patient_positive_count])
    plt.title("Class Distribution (Patients)")
    plt.ylabel("Number of Patients")
    plt.savefig(os.path.join(plot_dir, "patient_class_distribution.png"))
    plt.close()

    plt.figure(figsize=(8, 6))
    sns.barplot(x=['Negative (0)', 'Positive (1)'], y=[negative_samples, positive_samples])
    plt.title("Class Distribution (Timesteps)")
    plt.ylabel("Number of Hours")
    plt.savefig(os.path.join(plot_dir, "timestep_class_distribution.png"))
    plt.close()

    # Class balance quality visualization
    plt.figure(figsize=(10, 6))
    quality_metrics = ['Entropy', 'Normalized Entropy', 'Gini']
    values = [entropy, norm_entropy, gini]
    sns.barplot(x=quality_metrics, y=values)
    plt.title("Class Balance Quality Statistics\n(Note: These are descriptive, no universal 'good' threshold)")
    plt.ylabel("Score")
    plt.ylim(0, 1.0)
    plt.savefig(os.path.join(plot_dir, "class_balance_quality.png"))
    plt.close()

    # Save additional single metric plots to satisfy request
    plt.figure(figsize=(6, 4))
    plt.bar(["Entropy"], [entropy])
    plt.title("Class Entropy")
    plt.savefig(os.path.join(plot_dir, "class_entropy.png"))
    plt.close()

    plt.figure(figsize=(6, 4))
    plt.bar(["Gini"], [gini])
    plt.title("Class Gini Impurity")
    plt.savefig(os.path.join(plot_dir, "class_gini.png"))
    plt.close()
    
    plt.figure(figsize=(6, 4))
    plt.bar(["Imbalance Ratio"], [imbalance_ratio])
    plt.title("Class Imbalance Ratio")
    plt.savefig(os.path.join(plot_dir, "class_imbalance_ratio.png"))
    plt.close()

    print("Class Balance Analysis Completed.")

if __name__ == "__main__":
    run_class_balance()
