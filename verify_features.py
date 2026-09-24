#!/usr/bin/env python3

import os
import json

# Derive paths from this file's location
_THIS_FILE    = os.path.abspath(__file__)
project_root  = os.path.dirname(_THIS_FILE)               # f:\Sepsis\sepsis.1
sepsis_root   = os.path.join(project_root, "sepsis")      # f:\Sepsis\sepsis.1\sepsis

# Load config
prep_cfg_path = os.path.join(sepsis_root, "artifacts", "preprocessing_config.json")
with open(prep_cfg_path) as f:
    config = json.load(f)

features = config['dynamic_features']
print(f"Total features: {len(features)}")
print(f"Last two features: {features[-2:]}")
print(f"Input channels for transformer: {len(features) * 3}")

# Verify ICULOS position
iculos_idx = features.index('ICULOS') if 'ICULOS' in features else -1
print(f"ICULOS index: {iculos_idx}")

# Verify expected features
expected_count = 35
actual_count = len(features)
print(f"Expected: {expected_count}, Actual: {actual_count}, Match: {expected_count == actual_count}")
