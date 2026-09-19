import os
import json
import random
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from tqdm import tqdm

# ──────────────────────────────────────────────────────────────────────
# Deterministic Seed
# ──────────────────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)

# ──────────────────────────────────────────────────────────────────────
# Feature Definitions
# ──────────────────────────────────────────────────────────────────────
# Candidate dynamic features (will be filtered after stats computation)
CANDIDATE_DYNAMIC_FEATURES = [
    'HR', 'O2Sat', 'Temp', 'SBP', 'MAP', 'DBP', 'Resp', 'BaseExcess',
    'HCO3', 'FiO2', 'pH', 'PaCO2', 'SaO2', 'AST', 'BUN', 'Alkalinephos',
    'Calcium', 'Chloride', 'Creatinine', 'Bilirubin_direct', 'Glucose',
    'Lactate', 'Magnesium', 'Phosphate', 'Potassium', 'Bilirubin_total',
    'TroponinI', 'Hct', 'Hgb', 'PTT', 'WBC', 'Fibrinogen', 'Platelets',
    'EtCO2', 'ICULOS'  # Include EtCO2 and ICULOS
]

# Static features: recorded once per patient (constant across all hours)
STATIC_FEATURES = ['Age', 'Gender', 'Unit1', 'Unit2', 'HospAdmTime']

# Encoding strategy for static features
STATIC_ENCODING = {
    'Age':         'normalize',   # z-score with train mean/std
    'Gender':      'binary',      # already 0/1
    'Unit1':       'binary',      # already 0/1
    'Unit2':       'binary',      # already 0/1
    'HospAdmTime': 'normalize',   # z-score with train mean/std
}

MIN_OBSERVATION_RATE = 0.001   # Drop features observed < 0.1% in training
MIN_STD_THRESHOLD    = 1e-6    # Drop near-constant features
PERCENTILE_SAMPLE    = 2000    # Number of train patients sampled for percentile estimation

# ──────────────────────────────────────────────────────────────────────
# Worker: First Pass (parallel over ALL files)
# ──────────────────────────────────────────────────────────────────────
def _process_file(f, file_to_dir):
    """Read one .psv, return per-file aggregates for both dynamic and static features."""
    df = pd.read_csv(os.path.join(file_to_dir[f], f), sep='|')

    max_label = df['SepsisLabel'].max()

    # Dynamic feature aggregates
    D = len(CANDIDATE_DYNAMIC_FEATURES)
    sum_vals = np.zeros(D, dtype=np.float64)
    count_vals = np.zeros(D, dtype=np.float64)
    sq_sum_vals = np.zeros(D, dtype=np.float64)

    for i, col_name in enumerate(CANDIDATE_DYNAMIC_FEATURES):
        if col_name in df.columns:
            col = pd.to_numeric(df[col_name], errors='coerce')
            sum_vals[i] = col.sum(skipna=True)
            count_vals[i] = col.count()
            sq_sum_vals[i] = (col ** 2).sum(skipna=True)

    # Static feature values (take first non-missing value)
    static_vals = {}
    for sf in STATIC_FEATURES:
        if sf in df.columns:
            valid_vals = pd.to_numeric(df[sf], errors='coerce').dropna()
            static_vals[sf] = float(valid_vals.iloc[0]) if len(valid_vals) > 0 else None
        else:
            static_vals[sf] = None

    # Class counts
    counts = df['SepsisLabel'].value_counts()
    pos = counts.get(1, 0)
    neg = counts.get(0, 0)
    num_rows = len(df)

    return f, max_label, sum_vals, count_vals, sq_sum_vals, pos, neg, static_vals, num_rows


# ──────────────────────────────────────────────────────────────────────
# Worker: Percentile Pass (parallel over sampled TRAIN files only)
# ──────────────────────────────────────────────────────────────────────
def _read_raw_dynamic(f, file_to_dir, dyn_features):
    """Read raw dynamic values with a fixed feature order."""
    df = pd.read_csv(
        os.path.join(file_to_dir[f], f),
        sep='|'
    )
    # Reindex to guarantee exact feature order and shape
    df_reindexed = df.reindex(columns=dyn_features)
    return df_reindexed.apply(pd.to_numeric, errors='coerce').to_numpy(dtype=np.float64)


# ──────────────────────────────────────────────────────────────────────
# Main Preprocessing
# ──────────────────────────────────────────────────────────────────────
def run_preprocessing(data_dirs, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    # ── Discover files ──────────────────────────────────────────────
    print("Finding all patient files...")
    all_files = []
    file_to_dir = {}
    for d in data_dirs:
        files = sorted([f for f in os.listdir(d) if f.endswith('.psv')])
        for f in files:
            if f in file_to_dir:
                raise ValueError(f"Duplicate patient filename found: {f}")
            file_to_dir[f] = d
            all_files.append(f)

    print(f"Total files found: {len(all_files)}")

    # ── Pass 1: Parallel aggregation ────────────────────────────────
    print("Pass 1: Aggregating statistics in parallel...")
    file_stats = {}
    patient_has_sepsis = {}
    static_data = {}

    with ProcessPoolExecutor(max_workers=4) as executor:
        func = partial(_process_file, file_to_dir=file_to_dir)
        results = list(tqdm(executor.map(func, all_files, chunksize=100),
                            total=len(all_files), desc="Parsing CSVs"))

    for f, max_label, s, c, sq, pos, neg, sv, num_rows in results:
        patient_has_sepsis[f] = max_label

        file_stats[f] = {
            'sum': s,
            'count': c,
            'sq_sum': sq,
            'pos': pos,
            'neg': neg,
            'num_rows': num_rows
        }

        static_data[f] = sv

    # ── Stratified Split ────────────────────────────────────────────
    print("Stratifying Train / Val / Test split...")
    patients_df = pd.DataFrame(
        list(patient_has_sepsis.items()), columns=['PatientID', 'SepsisLabel']
    )
    train_patients, temp_patients = train_test_split(
        patients_df, test_size=0.30,
        stratify=patients_df['SepsisLabel'], random_state=SEED
    )
    val_patients, test_patients = train_test_split(
        temp_patients, test_size=0.50,
        stratify=temp_patients['SepsisLabel'], random_state=SEED
    )

    train_files = train_patients['PatientID'].tolist()
    val_files   = val_patients['PatientID'].tolist()
    test_files  = test_patients['PatientID'].tolist()
    train_set   = set(train_files)
    val_set     = set(val_files)
    test_set    = set(test_files)

    assert len(train_set.intersection(val_set)) == 0, "Train and Val splits overlap!"
    assert len(train_set.intersection(test_set)) == 0, "Train and Test splits overlap!"
    assert len(val_set.intersection(test_set)) == 0, "Val and Test splits overlap!"

    print(f"  Train: {len(train_files)} | Val: {len(val_files)} | Test: {len(test_files)}")

    # ── Pass 2: Percentile estimation & clipping (sampled training files) ──────
    print(f"Pass 2: Estimating clipping bounds from {PERCENTILE_SAMPLE} sampled training patients...")
    D = len(CANDIDATE_DYNAMIC_FEATURES)
    sorted_train_files = sorted(train_files)
    rng = random.Random(SEED)
    sample_files = rng.sample(sorted_train_files, min(PERCENTILE_SAMPLE, len(sorted_train_files)))

    with ProcessPoolExecutor() as executor:
        func = partial(_read_raw_dynamic, file_to_dir=file_to_dir,
                        dyn_features=CANDIDATE_DYNAMIC_FEATURES)
        raw_chunks = list(tqdm(executor.map(func, sample_files, chunksize=50),
                               total=len(sample_files), desc="Reading for percentiles"))

    all_raw = np.vstack(raw_chunks)  # (total_hours, D_candidate)
    clip_lo_cand = np.zeros(D, dtype=np.float64)
    clip_hi_cand = np.zeros(D, dtype=np.float64)
    for i, feat in enumerate(CANDIDATE_DYNAMIC_FEATURES):
        col = all_raw[:, i]
        observed = col[~np.isnan(col)]
        if len(observed) > 0:
            clip_lo_cand[i] = float(np.percentile(observed, 1))
            clip_hi_cand[i] = float(np.percentile(observed, 99))
        else:
            clip_lo_cand[i] = 0.0
            clip_hi_cand[i] = 0.0

    print("  Clipping bounds computed (1st–99th percentile).")

    # ── Pass 3: Compute training-only statistics AFTER clipping ────────
    print("Pass 3: Computing training-only dynamic feature statistics from CLIPPED observations...")

    total_sum = np.zeros(D, dtype=np.float64)
    total_count = np.zeros(D, dtype=np.float64)
    total_sq_sum = np.zeros(D, dtype=np.float64)

    total_pos = 0
    total_neg = 0
    total_train_rows = 0

    for f in train_set:

        st = file_stats[f]

        total_pos += st['pos']
        total_neg += st['neg']
        total_train_rows += st['num_rows']

        # Read training patient
        df = pd.read_csv(
            os.path.join(file_to_dir[f], f),
            sep='|'
        )

        # Calculate statistics from CLIPPED observations
        for i, col_name in enumerate(CANDIDATE_DYNAMIC_FEATURES):

            if col_name in df.columns:

                col_vals = pd.to_numeric(df[col_name], errors='coerce').dropna().values.astype(np.float64)

                if len(col_vals) > 0:

                    clipped_vals = np.clip(
                        col_vals,
                        clip_lo_cand[i],
                        clip_hi_cand[i]
                    )

                    total_sum[i] += np.sum(clipped_vals)
                    total_count[i] += len(clipped_vals)
                    total_sq_sum[i] += np.sum(clipped_vals ** 2)

    safe_count = total_count.copy()
    safe_count[safe_count == 0] = 1

    train_means = total_sum / safe_count

    train_vars = (
        total_sq_sum / safe_count
    ) - (train_means ** 2)

    train_vars[train_vars < 0] = 0

    train_stds = np.sqrt(train_vars)

    # ── Filter out unusable features ────────────────────────────────
    obs_rate = total_count / max(total_train_rows, 1)

    kept_indices = []
    removed_features = []
    for i, feat in enumerate(CANDIDATE_DYNAMIC_FEATURES):
        if obs_rate[i] < MIN_OBSERVATION_RATE:
            removed_features.append((feat, f"obs_rate={obs_rate[i]:.6f}"))
        elif train_stds[i] < MIN_STD_THRESHOLD:
            removed_features.append((feat, f"std={train_stds[i]:.8f}"))
        else:
            kept_indices.append(i)

    filtered_features = [CANDIDATE_DYNAMIC_FEATURES[i] for i in kept_indices]
    filtered_means = train_means[kept_indices]
    filtered_stds  = train_stds[kept_indices]
    clip_lo = {CANDIDATE_DYNAMIC_FEATURES[i]: float(clip_lo_cand[i]) for i in kept_indices}
    clip_hi = {CANDIDATE_DYNAMIC_FEATURES[i]: float(clip_hi_cand[i]) for i in kept_indices}

    if removed_features:
        print(f"  Removed {len(removed_features)} unusable feature(s):")
        for name, reason in removed_features:
            print(f"    - {name}: {reason}")
    print(f"  Kept {len(filtered_features)} dynamic features.")

    # ── Compute training-only static feature statistics ─────────────
    print("Computing training-only static feature statistics...")
    static_stats = {}
    for sf in STATIC_FEATURES:
        vals = []
        for f in train_set:
            v = static_data[f].get(sf)
            if v is not None and not np.isnan(v):
                vals.append(v)
        vals = np.array(vals, dtype=np.float64)
        if len(vals) > 0:
            static_stats[sf] = {
                'mean': float(np.mean(vals)),
                'std':  float(max(np.std(vals), 1e-6)),
                'encoding': STATIC_ENCODING.get(sf, 'normalize'),
            }
        else:
            static_stats[sf] = {
                'mean': 0.0, 'std': 1.0,
                'encoding': STATIC_ENCODING.get(sf, 'normalize'),
            }

    # Assertions for finite statistics
    assert np.all(np.isfinite(filtered_means)), "Non-finite values found in train_means!"
    assert np.all(np.isfinite(filtered_stds)), "Non-finite values found in train_stds!"

    # ── Assemble & save config ──────────────────────────────────────
    pos_weight = total_neg / (total_pos + 1e-9)
    input_channels = len(filtered_features) * 3

    config = {
        "seed": SEED,
        "dynamic_features": filtered_features,
        "static_features":  STATIC_FEATURES,
        "static_encoding":  STATIC_ENCODING,
        "static_stats":     static_stats,
        "train_means": dict(zip(filtered_features, filtered_means.tolist())),
        "train_stds":  dict(zip(filtered_features, filtered_stds.tolist())),
        "clip_lo":     clip_lo,
        "clip_hi":     clip_hi,
        "class_weight": pos_weight,
        "input_channels": input_channels,
        "removed_features": {name: reason for name, reason in removed_features},
        "sampled_percentile_files": sample_files,
    }

    # Save splits to a SEPARATE file to reduce risk of accidental misuse
    splits = {
        "train": train_files,
        "val":   val_files,
        "test":  test_files,
    }

    config_path = os.path.join(output_dir, "preprocessing_config.json")
    splits_path = os.path.join(output_dir, "splits.json")

    with open(config_path, "w") as fp:
        json.dump(config, fp, indent=4)
    with open(splits_path, "w") as fp:
        json.dump(splits, fp, indent=4)

    print(f"\nDone!")
    print(f"  Config  → {config_path}")
    print(f"  Splits  → {splits_path}")
    print(f"  Positive class weight: {pos_weight:.2f}")
    print(f"  Dynamic features: {len(filtered_features)}")
    print(f"  Static features:  {len(STATIC_FEATURES)}")


if __name__ == "__main__":
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    data_dirs = [
        os.path.join(project_root, "physionet2019", "training", "training_setA"),
        os.path.join(project_root, "physionet2019", "training", "training_setB")
    ]
    out_dir = os.path.join(project_root, "artifacts")
    run_preprocessing(data_dirs, out_dir)

