# tsfm_ad_lib/config.py
import os
from pathlib import Path

# Project Root 
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# --- Data Paths ---
# Default path to the raw dataset CSV file
DEFAULT_DATA_CSV_PATH = os.getenv("DATA_CSV_PATH", str(PROJECT_ROOT / "DATASET" / "train.csv"))

# Default path to the directory containing pre-generated fold ID
DEFAULT_FOLD_ID_DIR = os.getenv("FOLD_ID_DIR", str(PROJECT_ROOT / "lead-val-ids"))

# --- Output Paths ---
DEFAULT_RESULTS_DIR = os.getenv("RESULTS_DIR", str(PROJECT_ROOT / "results"))

# --- Model & Training Constants ----

# VAE specific
VAE_DEFAULT_INPUT_SIZE = 512  
VAE_DEFAULT_TRAIN_BATCH_SIZE = 160 
VAE_DEFAULT_N_SPLITS_KFOLD = 5

# MOMENT specific
MOMENT_DEFAULT_MODEL_NAME = "AutonLab/MOMENT-1-large"
MOMENT_DEFAULT_CONTEXT_LEN = 512
MOMENT_DEFAULT_TRAIN_BATCH_SIZE = 8 # The BATCH_SIZE within train_one_epoch for MOMENT
MOMENT_DEFAULT_N_SPLITS_KFOLD = 5 # Assuming same number of splits

# Statistical/Tree-based Models
DEFAULT_CONTAMINATION_SEARCH_RANGE = (0.001, 0.087, 0.005) # (start, stop, step)
DEFAULT_IQR_K_SEARCH_VALUES = [i / 10.0 for i in range(5, 45, 5)] # [0.5, 1.0, ..., 4.0]
DEFAULT_MZSCORE_K_SEARCH_VALUES = [i / 10.0 for i in range(5, 45, 5)] # [0.5, 1.0, ..., 4.0]

# Default IForest base parameters (excluding contamination)
DEFAULT_IFOREST_BASE_PARAMS = {
    'n_estimators': 114,
    'max_samples': 0.11944670503770004,
    'max_features': 0.11829004013466254,
    'bootstrap': True,
    'n_jobs': -1,
    'random_state': 42,
}

# Default LOF base parameters (excluding contamination)
DEFAULT_LOF_BASE_PARAMS = {
    'n_neighbors': 500, 
    'algorithm': 'brute', # 'auto' another choice 
    'leaf_size': 458, 
    'metric': 'minkowski',
    'p': 48, 
    'novelty': False
}

# --- Other ---
DEFAULT_DEVICE = "cuda" 
RANDOM_SEED = 42


if __name__ == '__main__':
    print(f"Project Root: {PROJECT_ROOT}")
    print(f"Default Data CSV Path: {DEFAULT_DATA_CSV_PATH}")
    print(f"Default Fold ID Dir: {DEFAULT_FOLD_ID_DIR}")
    print(f"VAE Default Input Size: {VAE_DEFAULT_INPUT_SIZE}")