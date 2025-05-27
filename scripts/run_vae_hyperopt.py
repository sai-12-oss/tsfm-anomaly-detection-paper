# scripts/run_vae_hyperopt.py
import argparse
import pandas as pd
import numpy as np
import sys
import os
from pathlib import Path
import joblib # For saving Optuna study and params
import optuna # The hyperparameter optimization framework
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
from tsfm_ad_lib.evaluation import evaluate_vae_model
from tsfm_ad_lib.utils import setup_logger, set_random_seeds, get_device, load_fold_ids
from tsfm_ad_lib import config as lib_config

# Global logger for this script
logger = None 

# --- Optuna Objective Function ---
def vae_objective(trial: optuna.Trial, args, imputed_df_global: pd.DataFrame, device: torch.device) -> float:
    """
    Optuna objective function for VAE hyperparameter optimization.
    A single call to this function constitutes one 'trial' in the Optuna study.
    """
    global logger # Use the globally defined logger

    # 1. Suggest Hyperparameters for this trial
    lr = trial.suggest_float('learning_rate', args.lr_min, args.lr_max, log=True)
    epochs = trial.suggest_int('epochs', args.epochs_min, args.epochs_max)
    # Adam betas from original script: beta_1=trial.suggest_float('beta_1',0.1,1), beta_2=trial.suggest_float('beta_2',0.1,1)
    # Using fixed betas for simplicity now, can be added back if critical
    adam_beta1 = args.adam_beta1
    adam_beta2 = args.adam_beta2
    
    # Model architecture params (from original Optuna search space)
    # hidden_size=trial.suggest_categorical('hidden_size',[256,192,164,128,64])
    # latent_dim=trial.suggest_categorical('latent_dim',[32,48,64,92])
    # num_layers=trial.suggest_int('num_layers',1,1) # Original fixed num_layers to 1
    initial_hidden_size = trial.suggest_categorical('initial_hidden_size', args.vae_hidden_sizes)
    latent_dim = trial.suggest_categorical('latent_dim', args.vae_latent_dims)
    num_hidden_layers = trial.suggest_int('num_hidden_layers', args.vae_num_layers_min, args.vae_num_layers_max)
    
    kld_weight = trial.suggest_float('kld_weight', args.kld_weight_min, args.kld_weight_max)


    logger.info(f"[Trial {trial.number}] Starting with params: LR={lr:.2e}, Epochs={epochs}, "
                f"InitialHidden={initial_hidden_size}, LatentDim={latent_dim}, NumLayers={num_hidden_layers}, KLD_Weight={kld_weight:.3f}")

    fold_avg_f1_scores = []
    
    # --- K-Fold Cross-Validation ---
    # `imputed_df_global` is the pre-imputed (but not necessarily scaled per fold) data
    all_available_building_ids = sorted(imputed_df_global[args.building_id_col].unique())

    for fold_idx in range(args.n_splits):
        logger.info(f"[Trial {trial.number} Fold {fold_idx+1}/{args.n_splits}] Starting...")
        
        # 2. Load Fold IDs (validation IDs for this fold)
        try:
            val_ids_for_fold = load_fold_ids(
                fold_idx=fold_idx, 
                fold_id_dir=args.fold_id_dir
            )
        except FileNotFoundError:
            logger.error(f"Validation ID file for fold {fold_idx} not found in {args.fold_id_dir}. Skipping trial or study.")
            # This is critical, Optuna might need to handle this (e.g. raise optuna.TrialPruned)
            # For now, we might return a very bad score to prune this trial.
            return -1.0 # Indicate failure for this trial
        
        train_ids_for_fold = [bid for bid in all_available_building_ids if bid not in val_ids_for_fold]

        if not train_ids_for_fold or not val_ids_for_fold:
            logger.warning(f"[Trial {trial.number} Fold {fold_idx}] No train or val IDs. Skipping fold.")
            # If a fold is invalid, how to score? This could skew results.
            # Maybe append a 0.0 score for this fold or handle more gracefully.
            fold_avg_f1_scores.append(0.0) 
            continue

        # 3. Prepare Data for the Fold (Scaling happens here, per fold, on training part)
        # The original script scaled globally then split. A more robust way is to
        # scale training part of fold, then use that scaler for validation part.
        # For simplicity and to match original global scaling behavior for now:
        # We use the imputed_df_global and then the VAE model's training/eval functions
        # will get dataframes filtered by train/val IDs.
        # The preprocess_lead_data_by_building handles scaling if scale_meter_reading=True.
        # For VAE, scaling is desired.
        
        df_train_this_fold = preprocess_lead_data_by_building(
            imputed_df_global[imputed_df_global[args.building_id_col].isin(train_ids_for_fold)],
            scale_meter_reading=True, # Scale training data for this fold
            building_id_col=args.building_id_col,
            meter_reading_col=args.meter_reading_col
        )
        df_val_this_fold = preprocess_lead_data_by_building(
            imputed_df_global[imputed_df_global[args.building_id_col].isin(val_ids_for_fold)],
            scale_meter_reading=True, # Scale validation data for this fold (ideally with scaler from train)
            building_id_col=args.building_id_col,
            meter_reading_col=args.meter_reading_col
        )
        # A more correct CV approach: fit scaler on df_train_this_fold_unscaled, then transform both.
        # The current `preprocess_lead_data_by_building` scales each building independently.

        if df_train_this_fold.empty or df_val_this_fold.empty:
            logger.warning(f"[Trial {trial.number} Fold {fold_idx}] Empty train or val DataFrame after preprocessing. Skipping fold.")
            fold_avg_f1_scores.append(0.0)
            continue

        # 4. Initialize Model, Optimizer, Loss
        model = VarEncoderDecoder(
            input_seq_length=args.context_len,
            num_hidden_layers=num_hidden_layers,
            initial_hidden_size=initial_hidden_size,
            latent_dim=latent_dim
        ).to(device)
        # Original VAE script used .double(), but library defaults to float for now.
        # If double: model = model.double()
        model = model.float()


        optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(adam_beta1, adam_beta2))
        # Original VAE script used nn.MSELoss() without reduction='sum' for train_one_epoch,
        # but VAE loss usually sums reconstruction. Let's use sum for recon part.
        reconstruction_loss_fn_for_vae = nn.MSELoss(reduction='sum') 

        best_f1_for_this_fold = 0.0
        
        # 5. Training Loop (Epochs)
        for epoch in range(epochs):
            logger.debug(f"[Trial {trial.number} Fold {fold_idx} Epoch {epoch+1}/{epochs}] Training...")
            train_loss = train_vae_epoch(
                model=model,
                df_train_fold=df_train_this_fold, # Pass the already scaled training data
                train_building_ids=train_ids_for_fold,
                optimizer=optimizer,
                reconstruction_loss_fn=reconstruction_loss_fn_for_vae,
                kld_weight=kld_weight,
                device=device,
                context_len=args.context_len,
                batch_size_script=lib_config.VAE_DEFAULT_TRAIN_BATCH_SIZE, # From original script
                meter_reading_col=args.meter_reading_col,
                anomaly_col=args.anomaly_col,
                building_id_col=args.building_id_col
            )
            logger.debug(f"[Trial {trial.number} Fold {fold_idx} Epoch {epoch+1}/{epochs}] Train Loss: {train_loss:.4f}")

            # Evaluation after each epoch
            val_f1_score = evaluate_vae_model(
                model=model,
                df_val_fold=df_val_this_fold, # Pass the already scaled validation data
                val_building_ids=val_ids_for_fold,
                device=device,
                context_len=args.context_len,
                meter_reading_col=args.meter_reading_col,
                anomaly_col=args.anomaly_col,
                building_id_col=args.building_id_col
            )
            logger.info(f"[Trial {trial.number} Fold {fold_idx} Epoch {epoch+1}/{epochs}] Val F1: {val_f1_score:.4f}")

            if val_f1_score > best_f1_for_this_fold:
                best_f1_for_this_fold = val_f1_score
                # Optionally save best model for this fold/trial if needed later
                # For Optuna, we just need the score.

            # Optuna Pruning: Check if the trial should be pruned
            trial.report(val_f1_score, epoch) # Report intermediate value
            if trial.should_prune():
                logger.info(f"[Trial {trial.number} Fold {fold_idx}] Pruned at epoch {epoch+1}.")
                # If pruned in one fold, what about others? Optuna typically prunes the whole trial.
                # We'll append the current best F1 for this fold and let the outer loop continue for other folds,
                # but Optuna will mark this trial as pruned based on this one call.
                # Or, raise optuna.TrialPruned() to stop this trial immediately.
                raise optuna.TrialPruned()


        fold_avg_f1_scores.append(best_f1_for_this_fold)
        logger.info(f"[Trial {trial.number} Fold {fold_idx}] Finished. Best F1 for this fold: {best_f1_for_this_fold:.4f}")

    # 6. Calculate average F1 score across folds for this trial
    mean_f1_all_folds = np.mean(fold_avg_f1_scores) if fold_avg_f1_scores else 0.0
    logger.info(f"[Trial {trial.number}] Completed. Average F1 across folds: {mean_f1_all_folds:.4f}")
    
    return float(mean_f1_all_folds)


def main(args):
    global logger # Make logger accessible in vae_objective
    logger = setup_logger("VAEHyperOptScript", log_file=args.log_file, level=args.log_level.upper())
    set_random_seeds(args.seed)
    device = get_device(args.device)

    logger.info("Starting VAE Hyperparameter Optimization Script")
    logger.info(f"Arguments: {args}")
    logger.info(f"Using device: {device}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory for Optuna study: {output_dir}")

    # 1. Load and globally impute data (scaling will be done per-fold within objective)
    try:
        raw_df = load_raw_lead_data(csv_file_path=args.data_path)
        # Global imputation (median, no scaling yet)
        imputed_df_global = preprocess_lead_data_by_building(
            raw_df, 
            scale_meter_reading=False, # Scaling done inside objective, per fold
            building_id_col=args.building_id_col,
            meter_reading_col=args.meter_reading_col
        )
        logger.info(f"Loaded and globally imputed data. Shape: {imputed_df_global.shape}")
    except Exception as e:
        logger.error(f"Error during data loading/preprocessing: {e}. Exiting.")
        return

    if imputed_df_global.empty:
        logger.error("No data after initial loading/imputation. Exiting.")
        return

    # 2. Setup Optuna Study
    study_name = args.study_name
    # For resuming: optuna.load_study(study_name=study_name, storage=f"sqlite:///{output_dir}/{study_name}.db")
    # For new study:
    storage_path = f"sqlite:///{output_dir}/{study_name}.db" # Use SQLite for resumable studies
    try:
        study = optuna.create_study(
            study_name=study_name,
            direction='maximize', # We want to maximize F1 score
            storage=storage_path,
            load_if_exists=True, # Resume if study with this name exists in the storage
            pruner=optuna.pruners.MedianPruner() # Example pruner
        )
    except Exception as e:
        logger.error(f"Error creating/loading Optuna study from {storage_path}: {e}")
        return
        
    logger.info(f"Optuna study '{study_name}' initialized. Storage: {storage_path}")
    logger.info(f"Number of finished trials in study: {len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])}")


    # 3. Run Optuna Optimization
    start_time = time.time()
    try:
        study.optimize(
            lambda trial: vae_objective(trial, args, imputed_df_global, device), 
            n_trials=args.n_trials,
            timeout=args.timeout_seconds # Optional timeout for the whole study
        )
    except KeyboardInterrupt:
        logger.warning("Optimization interrupted by user (KeyboardInterrupt).")
    except Exception as e:
        logger.error(f"An error occurred during Optuna optimization: {e}", exc_info=True)
    
    end_time = time.time()
    logger.info(f"Optuna optimization finished. Total time: {(end_time - start_time):.2f} seconds.")

    # 4. Save Results
    logger.info(f"Number of trials in study: {len(study.trials)}")
    if study.best_trial:
        logger.info(f"Best trial number: {study.best_trial.number}")
        logger.info(f"Best F1 score: {study.best_trial.value:.4f}")
        logger.info(f"Best hyperparameters: {study.best_trial.params}")
        
        best_params_path = output_dir / f"{study_name}_best_params.json" # Save as JSON
        with open(best_params_path, 'w') as f:
            import json
            json.dump(study.best_trial.params, f, indent=4)
        logger.info(f"Best hyperparameters saved to {best_params_path}")
    else:
        logger.warning("No best trial found (e.g., if all trials failed or were pruned early).")

    # Optuna study is automatically saved to the .db file if storage is used.
    # If not using storage, can save with joblib:
    # study_path_joblib = output_dir / f"{study_name}_study.pkl"
    # joblib.dump(study, study_path_joblib)
    # logger.info(f"Optuna study object saved to {study_path_joblib} (using joblib)")
    logger.info(f"Optuna study progress is saved in {storage_path}")
    logger.info("VAE Hyperparameter Optimization Script Finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run VAE Hyperparameter Optimization using Optuna.")

    # Data and Paths
    parser.add_argument('--data_path', type=str, default=lib_config.DEFAULT_DATA_CSV_PATH)
    parser.add_argument('--output_dir', type=str, default=str(Path(lib_config.DEFAULT_RESULTS_DIR) / "vae_hyperopt"))
    parser.add_argument('--fold_id_dir', type=str, default=lib_config.DEFAULT_FOLD_ID_DIR,
                        help="Directory containing pre-generated fold ID .pkl files.")
    parser.add_argument('--study_name', type=str, default="vae_optimization_study",
                        help="Name for the Optuna study (and .db file).")

    # Columns
    parser.add_argument('--building_id_col', type=str, default='building_id')
    parser.add_argument('--meter_reading_col', type=str, default='meter_reading')
    parser.add_argument('--anomaly_col', type=str, default='anomaly')

    # Optuna Study Parameters
    parser.add_argument('--n_trials', type=int, default=50, help="Number of Optuna trials.")
    parser.add_argument('--n_splits', type=int, default=lib_config.VAE_DEFAULT_N_SPLITS_KFOLD,
                        help="Number of K-fold cross-validation splits.")
    parser.add_argument('--timeout_seconds', type=int, default=None, help="Max optimization time in seconds.")
    
    # VAE Model & Training Hyperparameter Search Space (for Optuna)
    parser.add_argument('--lr_min', type=float, default=1e-6)
    parser.add_argument('--lr_max', type=float, default=1e-1)
    parser.add_argument('--epochs_min', type=int, default=5)
    parser.add_argument('--epochs_max', type=int, default=50) # Original script had this up to 50
    parser.add_argument('--adam_beta1', type=float, default=0.9, help="Adam optimizer beta1.") # Fixed for simplicity
    parser.add_argument('--adam_beta2', type=float, default=0.999, help="Adam optimizer beta2.")# Fixed for simplicity
    
    parser.add_argument('--vae_hidden_sizes', type=int, nargs='+', default=[64, 128, 192, 256],
                        help="List of initial hidden sizes for VAE to search.")
    parser.add_argument('--vae_latent_dims', type=int, nargs='+', default=[32, 48, 64, 92],
                        help="List of latent dimensions for VAE to search.")
    parser.add_argument('--vae_num_layers_min', type=int, default=1, help="Min number of VAE hidden layers.")
    parser.add_argument('--vae_num_layers_max', type=int, default=1, help="Max number of VAE hidden layers (original was fixed at 1).")
    
    parser.add_argument('--kld_weight_min', type=float, default=0.1, help="Min KLD weight for VAE loss.")
    parser.add_argument('--kld_weight_max', type=float, default=1.0, help="Max KLD weight for VAE loss.")


    # Fixed parameters (not part of Optuna search for this script)
    parser.add_argument('--context_len', type=int, default=lib_config.VAE_DEFAULT_INPUT_SIZE,
                        help="Sequence length for VAE input.")
    
    # Script execution
    parser.add_argument('--device', type=str, default=None, choices=['cuda', 'cpu'],
                        help="Device to use ('cuda' or 'cpu'). Defaults to config then auto-detect.")
    parser.add_argument('--seed', type=int, default=lib_config.RANDOM_SEED)
    parser.add_argument('--log_file', type=str, default=None)
    parser.add_argument('--log_level', type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])

    args = parser.parse_args()

    if args.log_file is None:
        args.log_file = str(Path(args.output_dir) / f"{args.study_name}.log")
    Path(args.log_file).parent.mkdir(parents=True, exist_ok=True)

    main(args)