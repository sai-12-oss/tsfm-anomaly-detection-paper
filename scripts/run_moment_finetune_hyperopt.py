# scripts/run_moment_finetune_hyperopt.py
import argparse
import pandas as pd
import numpy as np
import sys
import os
from pathlib import Path
import joblib
import optuna
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
from tsfm_ad_lib.evaluation import evaluate_moment_model
from tsfm_ad_lib.utils import setup_logger, set_random_seeds, get_device, load_fold_ids
from tsfm_ad_lib import config as lib_config

logger = None # Global logger

# --- Optuna Objective Function ---
def moment_objective(trial: optuna.Trial, args, imputed_df_global: pd.DataFrame, device: torch.device) -> float:
    global logger

    # 1. Suggest Hyperparameters for this trial
    lr = trial.suggest_float('learning_rate', args.lr_min, args.lr_max, log=True)
    epochs = trial.suggest_int('epochs', args.epochs_min, args.epochs_max)
    mask_ratio = trial.suggest_float('mask_ratio', args.mask_ratio_min, args.mask_ratio_max)
    # Adam betas from original script: beta_1=trial.suggest_float('beta_1',0.5,1.0), beta_2=trial.suggest_float('beta_2',0.5,1.0)
    # Using fixed for simplicity, can be added to Optuna
    adam_beta1 = args.adam_beta1 
    adam_beta2 = args.adam_beta2

    logger.info(f"[Trial {trial.number}] Starting MOMENT fine-tuning with params: LR={lr:.2e}, Epochs={epochs}, MaskRatio={mask_ratio:.3f}")

    fold_avg_f1_scores = []
    mask_generator = Masking(mask_ratio=mask_ratio) # Initialize Masking object for this trial
    
    all_available_building_ids = sorted(imputed_df_global[args.building_id_col].unique())

    for fold_idx in range(args.n_splits):
        logger.info(f"[Trial {trial.number} Fold {fold_idx+1}/{args.n_splits}] Starting...")
        
        try:
            val_ids_for_fold = load_fold_ids(fold_idx=fold_idx, fold_id_dir=args.fold_id_dir)
        except FileNotFoundError:
            logger.error(f"Validation ID file for fold {fold_idx} not found in {args.fold_id_dir}. Skipping trial.")
            return -1.0 
        
        train_ids_for_fold = [bid for bid in all_available_building_ids if bid not in val_ids_for_fold]

        if not train_ids_for_fold or not val_ids_for_fold:
            logger.warning(f"[Trial {trial.number} Fold {fold_idx}] No train or val IDs. Skipping fold.")
            fold_avg_f1_scores.append(0.0)
            continue

        # Prepare Data for the Fold (MOMENT script did not scale meter_reading globally, only median imputation)
        # The preprocess_lead_data_by_building by default will just do median imputation if scale_meter_reading=False
        df_train_this_fold = imputed_df_global[imputed_df_global[args.building_id_col].isin(train_ids_for_fold)]
        df_val_this_fold = imputed_df_global[imputed_df_global[args.building_id_col].isin(val_ids_for_fold)]
        
        # If scaling was desired for MOMENT (original script didn't explicitly show it for MOMENT before training loop):
        # df_train_this_fold = preprocess_lead_data_by_building(df_train_this_fold, scale_meter_reading=True)
        # df_val_this_fold = preprocess_lead_data_by_building(df_val_this_fold, scale_meter_reading=True)


        if df_train_this_fold.empty or df_val_this_fold.empty:
            logger.warning(f"[Trial {trial.number} Fold {fold_idx}] Empty train or val DataFrame. Skipping fold.")
            fold_avg_f1_scores.append(0.0)
            continue

        # 4. Initialize Model, Optimizer, Loss, Scheduler
        try:
            model = load_moment_pipeline_for_reconstruction(
                pretrained_model_name_or_path=args.moment_model_name,
                device=str(device) # load_moment expects string
            )
            if model is None: raise ValueError("Failed to load MOMENT model.")
        except Exception as e:
            logger.error(f"[Trial {trial.number} Fold {fold_idx}] Error loading MOMENT model: {e}. Skipping fold.")
            fold_avg_f1_scores.append(0.0) # Penalize if model loading fails
            continue


        optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(adam_beta1, adam_beta2))
        loss_fn = nn.MSELoss() # Standard MSE for reconstruction
        
        # Scheduler (original script had T_max calculation that needs EPOCHS from Optuna)
        # TOTAL_LEN=365*24; CONTEXT_LEN=512; BATCH_SIZE=8 (original script)
        # T_MAX_orig = int((TOTAL_LEN/CONTEXT_LEN)/BATCH_SIZE*EPOCHS) # This BATCH_SIZE is script's accumulation batch
        # For scheduler, usually number of steps per epoch or total steps.
        # If stepping per epoch, T_max = epochs.
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=args.scheduler_eta_min)

        best_f1_for_this_fold = 0.0
        
        for epoch in range(epochs):
            logger.debug(f"[Trial {trial.number} Fold {fold_idx} Epoch {epoch+1}/{epochs}] Training MOMENT...")
            train_loss = train_moment_epoch(
                model=model,
                df_train_fold=df_train_this_fold,
                train_building_ids=train_ids_for_fold,
                optimizer=optimizer,
                scheduler=scheduler, # Pass scheduler to step it
                loss_fn=loss_fn,
                mask_generator=mask_generator,
                device=device,
                context_len=args.context_len,
                batch_size_script=lib_config.MOMENT_DEFAULT_TRAIN_BATCH_SIZE, # From original MOMENT script
                meter_reading_col=args.meter_reading_col,
                anomaly_col=args.anomaly_col,
                building_id_col=args.building_id_col
            )
            logger.debug(f"[Trial {trial.number} Fold {fold_idx} Epoch {epoch+1}/{epochs}] MOMENT Train Loss: {train_loss:.4f}")

            val_f1_score = evaluate_moment_model(
                model=model,
                df_val_fold=df_val_this_fold,
                val_building_ids=val_ids_for_fold,
                device=device,
                context_len=args.context_len,
                meter_reading_col=args.meter_reading_col,
                anomaly_col=args.anomaly_col,
                building_id_col=args.building_id_col
            )
            logger.info(f"[Trial {trial.number} Fold {fold_idx} Epoch {epoch+1}/{epochs}] MOMENT Val F1: {val_f1_score:.4f}")

            if val_f1_score > best_f1_for_this_fold:
                best_f1_for_this_fold = val_f1_score
            
            trial.report(val_f1_score, epoch)
            if trial.should_prune():
                logger.info(f"[Trial {trial.number} Fold {fold_idx}] Pruned at epoch {epoch+1}.")
                raise optuna.TrialPruned()
        
        fold_avg_f1_scores.append(best_f1_for_this_fold)
        logger.info(f"[Trial {trial.number} Fold {fold_idx}] Finished. Best F1 for this fold: {best_f1_for_this_fold:.4f}")

    mean_f1_all_folds = np.mean(fold_avg_f1_scores) if fold_avg_f1_scores else 0.0
    logger.info(f"[Trial {trial.number}] Completed. Average F1 across folds for MOMENT: {mean_f1_all_folds:.4f}")
    
    return float(mean_f1_all_folds)

def main(args):
    global logger
    logger = setup_logger("MomentHyperOptScript", log_file=args.log_file, level=args.log_level.upper())
    set_random_seeds(args.seed)
    device = get_device(args.device)

    logger.info("Starting MOMENT Fine-tuning Hyperparameter Optimization Script")
    logger.info(f"Arguments: {args}")
    logger.info(f"Using device: {device}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory for Optuna study: {output_dir}")

    try:
        raw_df = load_raw_lead_data(csv_file_path=args.data_path)
        # MOMENT script did not scale data before training loop, only median imputation.
        imputed_df_global = preprocess_lead_data_by_building(
            raw_df, 
            scale_meter_reading=False, # Match original MOMENT script's preprocessing
            building_id_col=args.building_id_col,
            meter_reading_col=args.meter_reading_col
        )
        logger.info(f"Loaded and globally imputed data for MOMENT. Shape: {imputed_df_global.shape}")
    except Exception as e:
        logger.error(f"Error during data loading/preprocessing: {e}. Exiting.")
        return

    if imputed_df_global.empty:
        logger.error("No data after initial loading/imputation. Exiting.")
        return

    storage_path = f"sqlite:///{output_dir}/{args.study_name}.db"
    try:
        study = optuna.create_study(
            study_name=args.study_name,
            direction='maximize',
            storage=storage_path,
            load_if_exists=True,
            pruner=optuna.pruners.MedianPruner()
        )
    except Exception as e:
        logger.error(f"Error creating/loading Optuna study from {storage_path}: {e}")
        return
        
    logger.info(f"Optuna study '{args.study_name}' initialized for MOMENT. Storage: {storage_path}")
    logger.info(f"Number of finished trials in study: {len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])}")

    start_time = time.time()
    try:
        study.optimize(
            lambda trial: moment_objective(trial, args, imputed_df_global, device),
            n_trials=args.n_trials,
            timeout=args.timeout_seconds
        )
    except KeyboardInterrupt:
        logger.warning("MOMENT optimization interrupted by user.")
    except Exception as e:
        logger.error(f"An error occurred during MOMENT Optuna optimization: {e}", exc_info=True)
    
    end_time = time.time()
    logger.info(f"MOMENT Optuna optimization finished. Total time: {(end_time - start_time):.2f} seconds.")

    if study.best_trial:
        logger.info(f"Best MOMENT trial number: {study.best_trial.number}")
        logger.info(f"Best F1 score (MOMENT): {study.best_trial.value:.4f}")
        logger.info(f"Best hyperparameters (MOMENT): {study.best_trial.params}")
        
        best_params_path = output_dir / f"{args.study_name}_best_params.json"
        with open(best_params_path, 'w') as f:
            import json
            json.dump(study.best_trial.params, f, indent=4)
        logger.info(f"Best MOMENT hyperparameters saved to {best_params_path}")
    else:
        logger.warning("No best trial found for MOMENT.")

    logger.info(f"MOMENT Optuna study progress is saved in {storage_path}")
    logger.info("MOMENT Fine-tuning Hyperparameter Optimization Script Finished.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run MOMENT Fine-tuning Hyperparameter Optimization using Optuna.")

    # Data and Paths
    parser.add_argument('--data_path', type=str, default=lib_config.DEFAULT_DATA_CSV_PATH)
    parser.add_argument('--output_dir', type=str, default=str(Path(lib_config.DEFAULT_RESULTS_DIR) / "moment_hyperopt"))
    parser.add_argument('--fold_id_dir', type=str, default=lib_config.DEFAULT_FOLD_ID_DIR, # Assuming same folds are used
                        help="Directory containing pre-generated fold ID .pkl files. "
                             "Original MOMENT script used '/BASU_LEAD/validation_folds/', ensure consistency or update path.")
    parser.add_argument('--study_name', type=str, default="moment_finetune_study",
                        help="Name for the Optuna study (and .db file).")
    parser.add_argument('--moment_model_name', type=str, default=lib_config.MOMENT_DEFAULT_MODEL_NAME,
                        help="Name or path of the pre-trained MOMENT model.")


    # Columns
    parser.add_argument('--building_id_col', type=str, default='building_id')
    parser.add_argument('--meter_reading_col', type=str, default='meter_reading')
    parser.add_argument('--anomaly_col', type=str, default='anomaly')

    # Optuna Study Parameters
    parser.add_argument('--n_trials', type=int, default=50)
    parser.add_argument('--n_splits', type=int, default=lib_config.MOMENT_DEFAULT_N_SPLITS_KFOLD)
    parser.add_argument('--timeout_seconds', type=int, default=None)
    
    # MOMENT Fine-tuning Hyperparameter Search Space
    parser.add_argument('--lr_min', type=float, default=1e-6)
    parser.add_argument('--lr_max', type=float, default=1e-3) # MOMENT fine-tuning often uses smaller LRs
    parser.add_argument('--epochs_min', type=int, default=1)  # Original MOMENT script: 1 to 25
    parser.add_argument('--epochs_max', type=int, default=25)
    parser.add_argument('--mask_ratio_min', type=float, default=0.1) # Original: 0.1 to 0.5
    parser.add_argument('--mask_ratio_max', type=float, default=0.5)
    parser.add_argument('--adam_beta1', type=float, default=0.9) # Fixed for simplicity
    parser.add_argument('--adam_beta2', type=float, default=0.999) # Fixed for simplicity
    parser.add_argument('--scheduler_eta_min', type=float, default=1e-7, help="Min LR for CosineAnnealingLR.")


    # Fixed parameters for MOMENT
    parser.add_argument('--context_len', type=int, default=lib_config.MOMENT_DEFAULT_CONTEXT_LEN)
    
    # Script execution
    parser.add_argument('--device', type=str, default=None, choices=['cuda', 'cpu'])
    parser.add_argument('--seed', type=int, default=lib_config.RANDOM_SEED)
    parser.add_argument('--log_file', type=str, default=None)
    parser.add_argument('--log_level', type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])

    args = parser.parse_args()

    if args.log_file is None:
        args.log_file = str(Path(args.output_dir) / f"{args.study_name}.log")
    Path(args.log_file).parent.mkdir(parents=True, exist_ok=True)

    main(args)