# scripts/finetune_final_moment.py
import argparse
import pandas as pd
import numpy as np
import sys
import os
from pathlib import Path
import json # For loading best hyperparameters
import torch
import torch.nn as nn # For MSELoss
import time

# --- Add project root to sys.path ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
# --- End of sys.path modification ---

from tsfm_ad_lib.data_loader import load_raw_lead_data
from tsfm_ad_lib.preprocessing import preprocess_lead_data_by_building
from tsfm_ad_lib.models.moment import load_moment_pipeline_for_reconstruction
from tsfm_ad_lib.models.moment_utils import Masking
from tsfm_ad_lib.training import train_moment_epoch
from tsfm_ad_lib.utils import setup_logger, set_random_seeds, get_device
from tsfm_ad_lib import config as lib_config

def main(args):
    """
    Main function to fine-tune a final MOMENT model using specified (or best) hyperparameters.
    """
    logger = setup_logger("FinetuneFinalMomentScript", log_file=args.log_file, level=args.log_level.upper())
    set_random_seeds(args.seed)
    device = get_device(args.device)

    logger.info("Starting Final MOMENT Model Fine-tuning Script")
    logger.info(f"Arguments: {args}")
    logger.info(f"Using device: {device}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Model and logs will be saved in: {output_dir}")

    # 1. Load Best Hyperparameters
    if args.hyperparams_path:
        try:
            with open(args.hyperparams_path, 'r') as f:
                best_hyperparams = json.load(f)
            logger.info(f"Loaded best hyperparameters from: {args.hyperparams_path}")
            logger.info(f"Hyperparameters: {best_hyperparams}")
        except FileNotFoundError:
            logger.error(f"Hyperparameter file not found: {args.hyperparams_path}. Exiting.")
            return
        except json.JSONDecodeError:
            logger.error(f"Error decoding JSON from {args.hyperparams_path}. Exiting.")
            return
    else:
        logger.warning("No hyperparameter file provided. Using default/CLI arguments for hyperparameters.")
        best_hyperparams = {
            'learning_rate': args.learning_rate,
            'epochs': args.epochs,
            'mask_ratio': args.mask_ratio
            # Adam betas are taken from CLI defaults 
        }
        logger.info(f"Using parameters: {best_hyperparams}")

    # Extract hyperparameters
    lr = best_hyperparams.get('learning_rate', args.learning_rate)
    epochs = best_hyperparams.get('epochs', args.epochs)
    mask_ratio = best_hyperparams.get('mask_ratio', args.mask_ratio)
    adam_beta1 = args.adam_beta1
    adam_beta2 = args.adam_beta2

    # 2. Load and Preprocess Full Training Data
    try:
        raw_df = load_raw_lead_data(csv_file_path=args.data_path)
        # MOMENT original script didn't scale, only imputed.
        full_train_df_processed = preprocess_lead_data_by_building(
            raw_df,
            scale_meter_reading=False, # Match MOMENT script's preprocessing
            building_id_col=args.building_id_col,
            meter_reading_col=args.meter_reading_col
        )
        logger.info(f"Loaded and processed full training data for MOMENT. Shape: {full_train_df_processed.shape}")
    except Exception as e:
        logger.error(f"Error during data loading/preprocessing for final fine-tuning: {e}. Exiting.")
        return

    if full_train_df_processed.empty:
        logger.error("No data available for final fine-tuning after preprocessing. Exiting.")
        return

    all_training_building_ids = sorted(full_train_df_processed[args.building_id_col].unique())
    if not all_training_building_ids:
        logger.error("No building IDs found in the processed training data. Exiting.")
        return

    # 3. Initialize Model, Optimizer, Loss, Scheduler
    try:
        model = load_moment_pipeline_for_reconstruction(
            pretrained_model_name_or_path=args.moment_model_name,
            device=str(device) 
        )
        if model is None: raise ValueError("Failed to load MOMENT model.")
        logger.info(f"MOMENT model '{args.moment_model_name}' loaded for fine-tuning.")
    except Exception as e:
        logger.error(f"Error loading MOMENT model: {e}. Exiting.")
        return

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(adam_beta1, adam_beta2))
    loss_fn = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=args.scheduler_eta_min)
    mask_generator = Masking(mask_ratio=mask_ratio)

    # 4. Fine-tuning Loop
    logger.info(f"Starting final MOMENT fine-tuning for {epochs} epochs...")
    finetuning_start_time = time.time()
    for epoch in range(epochs):
        logger.info(f"--- Epoch {epoch+1}/{epochs} ---")
        
        train_loss = train_moment_epoch(
            model=model,
            df_train_fold=full_train_df_processed,
            train_building_ids=all_training_building_ids,
            optimizer=optimizer,
            scheduler=scheduler,
            loss_fn=loss_fn,
            mask_generator=mask_generator,
            device=device,
            context_len=args.context_len,
            batch_size_script=lib_config.MOMENT_DEFAULT_TRAIN_BATCH_SIZE,
            meter_reading_col=args.meter_reading_col,
            anomaly_col=args.anomaly_col,
            building_id_col=args.building_id_col
        )
        logger.info(f"Epoch {epoch+1} completed. MOMENT Fine-tuning Loss: {train_loss:.4f}")

    finetuning_duration = time.time() - finetuning_start_time
    logger.info(f"Final MOMENT fine-tuning finished in {finetuning_duration:.2f} seconds.")

    # 5. Save the Fine-tuned Model
    # MOMENTPipeline saves the whole pipeline (including preprocessor if any, and model)
    # For just model weights: torch.save(model.model.state_dict(), path)
    # For Hugging Face Transformers-like models, pipeline.save_pretrained(path) is common.
    # Let's assume MOMENTPipeline has a save_pretrained method or we save the state_dict.
    # The original library `momentfm` might have its own preferred saving method.
    # For now, let's save the state_dict of the underlying model.
    model_save_path_dir = output_dir / args.model_savename # This will be a directory
    try:
        # If MOMENTPipeline has a save_pretrained method:
        if hasattr(model, 'save_pretrained'):
             model.save_pretrained(str(model_save_path_dir))
             logger.info(f"Fine-tuned final MOMENT pipeline saved to directory: {model_save_path_dir}")
        else: # Fallback to saving state_dict of the model component
            torch.save(model.model.state_dict(), str(model_save_path_dir) + ".pth") # Save as a .pth file
            logger.warning(f"MOMENTPipeline does not have save_pretrained. Saved model.model.state_dict() to: {model_save_path_dir}.pth")
            logger.info("Note: For full pipeline reproducibility with MOMENT, using its own saving mechanism is preferred if available.")

    except Exception as e:
        logger.error(f"Error saving fine-tuned MOMENT model: {e}")

    logger.info("Final MOMENT Model Fine-tuning Script Finished.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune a final MOMENT model with given hyperparameters.")

    # Data and Paths
    parser.add_argument('--data_path', type=str, default=lib_config.DEFAULT_DATA_CSV_PATH)
    parser.add_argument('--output_dir', type=str, default=str(Path(lib_config.DEFAULT_RESULTS_DIR) / "final_moment_model"))
    parser.add_argument('--model_savename', type=str, default="final_finetuned_moment_pipeline",
                        help="Name for the saved fine-tuned MOMENT model/pipeline (will be a directory if save_pretrained is used).")
    parser.add_argument('--hyperparams_path', type=str, default=None,
                        help="Path to a JSON file containing best hyperparameters for fine-tuning.")
    parser.add_argument('--moment_model_name', type=str, default=lib_config.MOMENT_DEFAULT_MODEL_NAME,
                        help="Name or path of the base pre-trained MOMENT model to start fine-tuning from.")

    # Columns
    parser.add_argument('--building_id_col', type=str, default='building_id')
    parser.add_argument('--meter_reading_col', type=str, default='meter_reading')
    parser.add_argument('--anomaly_col', type=str, default='anomaly')

    # Model & Fine-tuning Hyperparameters (can be overridden by --hyperparams_path)
    parser.add_argument('--learning_rate', type=float, default=1e-5) # Typical fine-tuning LR
    parser.add_argument('--epochs', type=int, default=10) 
    parser.add_argument('--mask_ratio', type=float, default=0.15)
    parser.add_argument('--adam_beta1', type=float, default=0.9)
    parser.add_argument('--adam_beta2', type=float, default=0.999)
    parser.add_argument('--scheduler_eta_min', type=float, default=1e-7)

    # Fixed parameters
    parser.add_argument('--context_len', type=int, default=lib_config.MOMENT_DEFAULT_CONTEXT_LEN)
    
    # Script execution
    parser.add_argument('--device', type=str, default=None, choices=['cuda', 'cpu'])
    parser.add_argument('--seed', type=int, default=lib_config.RANDOM_SEED)
    parser.add_argument('--log_file', type=str, default=None)
    parser.add_argument('--log_level', type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])

    args = parser.parse_args()

    if args.log_file is None:
        args.log_file = str(Path(args.output_dir) / "finetune_final_moment.log")
    Path(args.log_file).parent.mkdir(parents=True, exist_ok=True)

    main(args)