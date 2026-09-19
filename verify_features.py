#!/usr/bin/env python3

import json

# Load config
with open('physionet2019/artifacts/preprocessing_config.json') as f:
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