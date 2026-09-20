import os
import glob
import pandas as pd
import json

def get_data_dirs():
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    return [
        os.path.join(root, "training", "training_setA"),
        os.path.join(root, "training", "training_setB")
    ]

def get_all_psv_files():
    dirs = get_data_dirs()
    all_files = []
    for d in dirs:
        if os.path.exists(d):
            all_files.extend(glob.glob(os.path.join(d, "*.psv")))
    return all_files

def get_results_dir(sub_dir=""):
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    path = os.path.join(root, "experiments", "results", "eda", sub_dir)
    os.makedirs(path, exist_ok=True)
    return path

def get_plots_dir():
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    path = os.path.join(root, "experiments", "eda", "plots")
    os.makedirs(path, exist_ok=True)
    return path

def get_config():
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    config_path = os.path.join(root, "artifacts", "preprocessing_config.json")
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            return json.load(f)
    return None
