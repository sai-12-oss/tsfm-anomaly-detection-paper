# tsfm_ad_lib/utils.py
import torch
import numpy as np
import random
import joblib
from pathlib import Path
import os
import logging
from typing import List, Any, Optional

from . import config # To access default paths and seeds

def get_device(device_override: Optional[str] = None) -> torch.device:
    """
    Determines the torch device to use.
    Priority:
    1. `device_override` argument if provided.
    2. `config.DEFAULT_DEVICE` if it's 'cuda' and CUDA is available.
    3. 'cuda' if available and `config.DEFAULT_DEVICE` is not 'cpu'.
    4. 'cpu' otherwise.

    Args:
        device_override (Optional[str]): Specific device to use ('cuda', 'cpu').

    Returns:
        torch.device: The selected torch device.
    """
    if device_override:
        if device_override not in ['cuda', 'cpu']:
            raise ValueError("device_override must be 'cuda' or 'cpu'.")
        if device_override == 'cuda' and not torch.cuda.is_available():
            print("Warning: CUDA override requested, but CUDA not available. Falling back to CPU.")
            return torch.device("cpu")
        return torch.device(device_override)

    preferred_device_from_config = getattr(config, 'DEFAULT_DEVICE', 'cuda') # Default to 'cuda' if not in config

    if preferred_device_from_config == 'cuda':
        if torch.cuda.is_available():
            return torch.device("cuda")
        else:
            print("Info: config.DEFAULT_DEVICE is 'cuda', but CUDA not available. Using CPU.")
            return torch.device("cpu")
    elif preferred_device_from_config == 'cpu':
        return torch.device("cpu")
    else: # Default to trying CUDA if config is something else or not set properly
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")


def set_random_seeds(seed_value: Optional[int] = None) -> None:
    """
    Sets random seeds for Python's `random`, NumPy, and PyTorch for reproducibility.

    Args:
        seed_value (Optional[int]): The seed value to use.
                                    If None, uses `config.RANDOM_SEED`.
    """
    if seed_value is None:
        seed_value = config.RANDOM_SEED
    
    if not isinstance(seed_value, int):
        print(f"Warning: Invalid seed value type {type(seed_value)}. Using default seed {config.RANDOM_SEED} instead.")
        seed_value = config.RANDOM_SEED

    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value) # For multi-GPU setups
        # The following two lines can sometimes impact performance,
        # but ensure deterministic behavior for CUDA operations.
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False
    print(f"Random seeds set to: {seed_value}")


def load_fold_ids(
    fold_idx: int, 
    fold_id_dir: Optional[str] = None,
    filename_pattern: str = "val_id_fold{fold_idx}.pkl"
) -> List[Any]:
    """
    Loads a list of validation IDs for a specific fold from a .pkl file.

    Args:
        fold_idx (int): The index of the fold (e.g., 0, 1, 2...).
        fold_id_dir (Optional[str]): Directory containing the fold ID files.
                                     If None, uses `config.DEFAULT_FOLD_ID_DIR`.
        filename_pattern (str): Pattern for the filename, with {fold_idx} as a placeholder.
                                Default: "val_id_fold{fold_idx}.pkl".

    Returns:
        List[Any]: A list of validation IDs for the specified fold.

    Raises:
        FileNotFoundError: If the fold ID file is not found.
        Exception: For other errors during loading.
    """
    if fold_id_dir is None:
        fold_id_dir = config.DEFAULT_FOLD_ID_DIR
    
    file_name = filename_pattern.format(fold_idx=fold_idx)
    file_path = Path(fold_id_dir) / file_name

    if not file_path.is_file():
        raise FileNotFoundError(f"Fold ID file not found: {file_path}")

    try:
        ids = joblib.load(file_path)
        if not isinstance(ids, list):
            print(f"Warning: Loaded fold IDs from {file_path} are not a list (type: {type(ids)}). Attempting to convert.")
            try:
                ids = list(ids)
            except TypeError:
                raise TypeError(f"Could not convert loaded fold IDs from {file_path} to a list.")
        print(f"Successfully loaded {len(ids)} IDs for fold {fold_idx} from {file_path}.")
        return ids
    except Exception as e:
        print(f"Error loading fold IDs from {file_path}: {e}")
        raise


def setup_logger(
    name: str = "tsfm_ad_lib_logger", 
    level: int = logging.INFO,
    log_file: Optional[str] = None,
    log_to_console: bool = True
) -> logging.Logger:
    """
    Sets up a basic logger.

    Args:
        name (str): Name of the logger.
        level (int): Logging level (e.g., logging.INFO, logging.DEBUG).
        log_file (Optional[str]): Path to a file to save logs. If None, no file logging.
        log_to_console (bool): If True, logs to console.

    Returns:
        logging.Logger: The configured logger instance.
    """
    logger = logging.getLogger(name)
    
    # Prevent adding multiple handlers if logger already configured
    if logger.hasHandlers():
        logger.handlers.clear() # Or just return logger if already configured as desired

    logger.setLevel(level)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                                  datefmt='%Y-%m-%d %H:%M:%S')

    if log_to_console:
        ch = logging.StreamHandler()
        ch.setLevel(level)
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    if log_file:
        try:
            fh = logging.FileHandler(log_file, mode='a') # Append mode
            fh.setLevel(level)
            fh.setFormatter(formatter)
            logger.addHandler(fh)
        except Exception as e:
            logger.error(f"Failed to set up file handler for log file {log_file}: {e}", exc_info=False) # exc_info=False to avoid traceback in log for this error

    # To prevent propagation to root logger if it has handlers that duplicate output
    logger.propagate = False 
    
    return logger

if __name__ == '__main__':
    # Example Usage
    print("--- Testing get_device ---")
    selected_device = get_device()
    print(f"Selected device: {selected_device}")
    if torch.cuda.is_available():
        selected_device_cuda = get_device(device_override='cuda')
        print(f"Override to CUDA (if avail): {selected_device_cuda}")
    selected_device_cpu = get_device(device_override='cpu')
    print(f"Override to CPU: {selected_device_cpu}")

    print("\n--- Testing set_random_seeds ---")
    set_random_seeds(123)
    print(f"Numpy random: {np.random.rand(1)}")
    set_random_seeds() # Uses config.RANDOM_SEED
    print(f"Numpy random (after config seed): {np.random.rand(1)}")

    print("\n--- Testing setup_logger ---")
    # Create a dummy log file path for testing
    temp_log_dir = Path(config.PROJECT_ROOT) / "temp_logs"
    os.makedirs(temp_log_dir, exist_ok=True)
    example_log_file = temp_log_dir / "example_utils.log"
    
    my_logger = setup_logger("MyExampleLogger", level=logging.DEBUG, log_file=str(example_log_file))
    my_logger.debug("This is a debug message.")
    my_logger.info("This is an info message.")
    my_logger.warning("This is a warning.")
    print(f"Check for log messages in console and in '{example_log_file}' (if created).")


    print("\n--- Testing load_fold_ids (requires dummy files) ---")
    # To test load_fold_ids, you'd need to create dummy .pkl files
    # Example: Create a dummy fold ID file
    dummy_fold_dir = Path(config.DEFAULT_FOLD_ID_DIR)
    os.makedirs(dummy_fold_dir, exist_ok=True)
    dummy_fold_0_file = dummy_fold_dir / "val_id_fold0.pkl"
    try:
        if not dummy_fold_0_file.exists(): # Only create if it doesn't exist
            print(f"Creating dummy file for load_fold_ids test: {dummy_fold_0_file}")
            joblib.dump([101, 102, 103], dummy_fold_0_file)
        
        loaded_ids = load_fold_ids(fold_idx=0)
        print(f"Loaded IDs for fold 0: {loaded_ids}")
    except FileNotFoundError as e:
        print(f"Could not test load_fold_ids: {e}")
        print("Please ensure the fold ID directory and a dummy val_id_fold0.pkl exist or update config.DEFAULT_FOLD_ID_DIR.")
    except Exception as e:
        print(f"An error occurred during load_fold_ids test: {e}")
    finally:
        # Clean up dummy file if desired, or leave for manual inspection
        # if dummy_fold_0_file.exists():
        #     os.remove(dummy_fold_0_file)
        # if os.path.exists(temp_log_dir) and not os.listdir(temp_log_dir): # if temp_log_dir is empty
        #     os.rmdir(temp_log_dir)
        pass