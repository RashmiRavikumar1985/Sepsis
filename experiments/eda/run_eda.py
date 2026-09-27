import os
import sys

def run_script(script_name):
    print(f"--- Running {script_name} ---")
    ret = os.system(f"{sys.executable} experiments/eda/{script_name}")
    if ret != 0:
        print(f"Error running {script_name}")

if __name__ == "__main__":
    print("====================================")
    print("  Starting Complete EDA Pipeline")
    print("====================================")
    
    run_script("dataset_audit.py")
    run_script("class_balance.py")
    run_script("window_balance.py")
    run_script("temporal_analysis.py")
    run_script("feature_stats.py")
    run_script("missingness_and_correlation.py")
    
    print("====================================")
    print("  EDA Pipeline Completed")
    print("====================================")
