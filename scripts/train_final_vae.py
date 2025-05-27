# scripts/train_final_vae.py
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
from tsfm_ad_lib.models.vae import VarEncoderDecoder
from tsfm_ad_lib.training import train_vae_epoch
# No direct evaluation against a validation set here, focus is on training the final model
# If post-training evaluation is needed, it can be added or done by a separate script.
from tsfm_ad_lib.utils import setup_logger, set_random_seeds, get_device
from tsfm_ad_lib import config as lib_config

def main(args):
    """
    Main function to train a final VAE model using specified (or best) hyperparameters.
    """
    logger = setup_logger("TrainFinalVAEScript", log_file=args.log_file, level=args.log_level.upper())
    set_random_seeds(args.seed)
    device = get_device(args.device)

    logger.info("Starting Final VAE Model Training Script")
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
        # Use CLI args or defaults if no file provided
        best_hyperparams = {
            'learning_rate': args.learning_rate,
            'epochs': args.epochs,
            'initial_hidden_size': args.initial_hidden_size,
            'latent_dim': args.latent_dim,
            'num_hidden_layers': args.num_hidden_layers,
            'kld_weight': args.kld_weight
        }
        logger.info(f"Using parameters: {best_hyperparams}")

    # Extract hyperparameters
    lr = best_hyperparams.get('learning_rate', args.learning_rate) # Fallback to CLI/default
    epochs = best_hyperparams.get('epochs', args.epochs)
    initial_hidden_size = best_hyperparams.get('initial_hidden_size', args.initial_hidden_size)
    latent_dim = best_hyperparams.get('latent_dim', args.latent_dim)
    num_hidden_layers = best_hyperparams.get('num_hidden_layers', args.num_hidden_layers)
    kld_weight = best_hyperparams.get('kld_weight', args.kld_weight)
    adam_beta1 = args.adam_beta1
    adam_beta2 = args.adam_beta2

    # 2. Load and Preprocess Full Training Data
    try:
        raw_df = load_raw_lead_data(csv_file_path=args.data_path)
        # For final training, we typically use all available training data.
        # Scaling is applied per building.
        full_train_df_processed = preprocess_lead_data_by_building(
            raw_df,
            scale_meter_reading=True, # VAE expects scaled data
            building_id_col=args.building_id_col,
            meter_reading_col=args.meter_reading_col
            # building_ids_to_process can be used if a specific subset is 'full training'
        )
        logger.info(f"Loaded and processed full training data. Shape: {full_train_df_processed.shape}")
    except Exception as e:
        logger.error(f"Error during data loading/preprocessing for final training: {e}. Exiting.")
        return

    if full_train_df_processed.empty:
        logger.error("No data available for final training after preprocessing. Exiting.")
        return

    all_training_building_ids = sorted(full_train_df_processed[args.building_id_col].unique())
    if not all_training_building_ids:
        logger.error("No building IDs found in the processed training data. Exiting.")
        return

    # 3. Initialize Model, Optimizer, Loss
    model = VarEncoderDecoder(
        input_seq_length=args.context_len,
        num_hidden_layers=num_hidden_layers,
        initial_hidden_size=initial_hidden_size,
        latent_dim=latent_dim
    ).to(device)
    model = model.float() # Assuming float32, adjust if double needed
    logger.info("VAE model initialized.")
    # print(model) # Optionally print model summary

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(adam_beta1, adam_beta2))
    reconstruction_loss_fn_for_vae = nn.MSELoss(reduction='sum')

    # 4. Training Loop
    logger.info(f"Starting final VAE training for {epochs} epochs...")
    training_start_time = time.time()
    for epoch in range(epochs):
        logger.info(f"--- Epoch {epoch+1}/{epochs} ---")
        
        train_loss = train_vae_epoch(
            model=model,
            df_train_fold=full_train_df_processed, # Using the full processed dataset
            train_building_ids=all_training_building_ids, # Train on all buildings
            optimizer=optimizer,
            reconstruction_loss_fn=reconstruction_loss_fn_for_vae,
            kld_weight=kld_weight,
            device=device,
            context_len=args.context_len,
            batch_size_script=lib_config.VAE_DEFAULT_TRAIN_BATCH_SIZE, # From original script logic
            meter_reading_col=args.meter_reading_col,
            anomaly_col=args.anomaly_col,
            building_id_col=args.building_id_col
        )
        logger.info(f"Epoch {epoch+1} completed. Training Loss: {train_loss:.4f}")
        # No validation here, as we are training the final model.
        # If desired, a small holdout could be used for monitoring, but not for model selection.

    training_duration = time.time() - training_start_time
    logger.info(f"Final VAE training finished in {training_duration:.2f} seconds.")

    # 5. Save the Trained Model
    model_save_path = output_dir / args.model_filename
    try:
        torch.save(model.state_dict(), model_save_path)
        logger.info(f"Trained final VAE model state_dict saved to: {model_save_path}")
    except Exception as e:
        logger.error(f"Error saving VAE model: {e}")

    logger.info("Final VAE Model Training Script Finished.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a final VAE model with given hyperparameters.")

    # Data and Paths
    parser.add_argument('--data_path', type=str, default=lib_config.DEFAULT_DATA_CSV_PATH)
    parser.add_argument('--output_dir', type=str, default=str(Path(lib_config.DEFAULT_RESULTS_DIR) / "final_vae_model"))
    parser.add_argument('--model_filename', type=str, default="final_vae_model.pth",
                        help="Filename for the saved trained model state_dict.")
    parser.add_argument('--hyperparams_path', type=str, default=None,
                        help="Path to a JSON file containing best hyperparameters. If None, uses CLI args.")

    # Columns
    parser.add_argument('--building_id_col', type=str, default='building_id')
    parser.add_argument('--meter_reading_col', type=str, default='meter_reading')
    parser.add_argument('--anomaly_col', type=str, default='anomaly') # Needed by TimeDataset for masking logic

    # Model & Training Hyperparameters (can be overridden by --hyperparams_path)
    # These serve as defaults if the JSON file doesn't provide them or if no file is given.
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--epochs', type=int, default=30) # Example default
    parser.add_argument('--initial_hidden_size', type=int, default=128) # Example default
    parser.add_argument('--latent_dim', type=int, default=32) # Example default
    parser.add_argument('--num_hidden_layers', type=int, default=1) # Example default
    parser.add_argument('--kld_weight', type=float, default=1.0) # Example default
    parser.add_argument('--adam_beta1', type=float, default=0.9)
    parser.add_argument('--adam_beta2', type=float, default=0.999)
    
    # Fixed parameters (consistent with VAE training)
    parser.add_argument('--context_len', type=int, default=lib_config.VAE_DEFAULT_INPUT_SIZE)
    
    # Script execution
    parser.add_argument('--device', type=str, default=None, choices=['cuda', 'cpu'])
    parser.add_argument('--seed', type=int, default=lib_config.RANDOM_SEED)
    parser.add_argument('--log_file', type=str, default=None)
    parser.add_argument('--log_level', type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])

    args = parser.parse_args()

    if args.log_file is None:
        args.log_file = str(Path(args.output_dir) / "train_final_vae.log")
    Path(args.log_file).parent.mkdir(parents=True, exist_ok=True)

    main(args)