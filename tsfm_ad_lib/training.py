# tsfm_ad_lib/training.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import math
import pandas as pd
import numpy as np
from typing import List, Any, Callable, Optional

from .data_loader import TimeDataset # Assuming TimeDataset is the primary dataset type
from .models.moment_utils import Masking # For MOMENT model training

# It's good practice to pass device explicitly or get it from a central place
# For now, functions will take 'device' as an argument.

def train_vae_epoch(
    model: nn.Module, # Specifically the VarEncoderDecoder model
    df_train_fold: pd.DataFrame,
    train_building_ids: List[Any],
    optimizer: torch.optim.Optimizer,
    reconstruction_loss_fn, # e.g., nn.MSELoss() instance for VAE's own loss_function
    kld_weight: float, # Weight for the KLD term in VAE loss
    device: torch.device,
    context_len: int,
    batch_size_script: int, # The BATCH_SIZE from the original script's single_run
    meter_reading_col: str = 'meter_reading',
    anomaly_col: str = 'anomaly',
    building_id_col: str = 'building_id'
) -> float:
    """
    Trains the Variational Autoencoder (VAE) model for one epoch using the custom
    batching strategy from the original script.

    Args:
        model: The VAE model instance (VarEncoderDecoder).
        df_train_fold: DataFrame containing training data for the current fold.
        train_building_ids: List of building IDs to use for training in this epoch.
        optimizer: The PyTorch optimizer.
        reconstruction_loss_fn: PyTorch loss function for reconstruction (e.g., nn.MSELoss(reduction='sum')).
        kld_weight: Weight for the KL divergence term in the VAE loss.
        device: The torch device ('cuda' or 'cpu').
        context_len: The sequence length for model input.
        batch_size_script: The accumulation batch size used in the original script's loop.
        meter_reading_col (str): Name of the meter reading column.
        anomaly_col (str): Name of the anomaly label column.
        building_id_col (str): Name of the building ID column.

    Returns:
        float: The average training loss for the epoch.
    """
    model.train()
    
    epoch_losses = []
    
    # Prepare DataLoaders for each training building ID (as per original script)
    list_of_loaders = []
    list_of_medians = [] # For anomaly masking
    
    for b_id in train_building_ids:
        building_df = df_train_fold[df_train_fold[building_id_col] == b_id].copy()
        if building_df.empty:
            continue
        building_df.reset_index(drop=True, inplace=True)
        
        # Calculate median for anomaly masking (on potentially scaled data)
        median_val = building_df[meter_reading_col].median()
        if pd.isna(median_val): # If all NaNs, or empty after filtering
            median_val = 0.0 # Fallback if median cannot be computed
            
        dataset = TimeDataset(building_df, meter_reading_col=meter_reading_col, anomaly_col=anomaly_col)
        # Original VAE script used shuffle=False, drop_last=True
        loader = DataLoader(dataset, batch_size=context_len, shuffle=False, drop_last=True)
        
        if len(loader) > 0: # Only add if there's at least one full batch
            list_of_loaders.append(iter(loader))
            list_of_medians.append(median_val)

    if not list_of_loaders:
        print("Warning: No data loaders created for VAE training epoch. Skipping.")
        return 0.0 # Or np.nan

    # Determine number of iterations (min_len from original script)
    # This ensures all loaders are iterated roughly the same number of times.
    # The original had min_len=14, which seems arbitrary. A better approach
    # might be to iterate until the shortest loader is exhausted.
    try:
        num_iterations = min(len(ldr.dataset) // context_len for ldr in list_of_loaders if hasattr(ldr, 'dataset') and len(ldr.dataset) // context_len > 0)
        num_iterations = max(1, num_iterations) # Ensure at least 1 iteration if possible
    except ValueError: # Handles case where a loader might be empty after all
        num_iterations = 0
    
    if num_iterations == 0:
        # min_len_original = 14 # Fallback to original script's arbitrary value if dynamic calc fails
        # num_iterations = min_len_original
        # print(f"Warning: Could not determine dynamic num_iterations for VAE training. Defaulting to 14, this may cause errors if loaders are too short.")
        print("Warning: Not enough data in some loaders for VAE training iteration. Skipping epoch or loss might be 0.")
        return 0.0


    # Custom batch accumulation loop from original script
    pbar = tqdm(range(num_iterations), desc="VAE Epoch Training")
    for _ in pbar:
        current_batch_x_list = []
        current_batch_target_list = []
        
        num_series_in_batch = 0
        for i, loader_iter in enumerate(list_of_loaders):
            try:
                # meter_reading, placeholder_val, anomaly_label
                single_x_series, _, labels = next(loader_iter) 
            except StopIteration:
                # This loader is exhausted for this epoch, try to re-init for next meta-batch or skip
                # For simplicity, we'll assume num_iterations handles this.
                # If a loader finishes early, this inner loop might break or behave unexpectedly
                # with the original script's batching logic.
                # This part of the original logic is complex to make perfectly robust without
                # knowing the exact intent of `min_len=14`.
                continue # Skip this loader for this meta-iteration

            median_for_series = list_of_medians[i]
            labels_bool = labels.bool() # Convert to boolean mask

            # Prepare target: mask anomalies with median
            target_x_series = single_x_series.clone().detach()
            target_x_series[labels_bool] = median_for_series
            
            current_batch_x_list.append(single_x_series.unsqueeze(0)) # [1, context_len]
            current_batch_target_list.append(target_x_series.unsqueeze(0)) # [1, context_len]
            num_series_in_batch += 1

            if (num_series_in_batch % batch_size_script == 0) or (i == len(list_of_loaders) - 1 and num_series_in_batch > 0) :
                if not current_batch_x_list: continue # Nothing to process

                batch_x = torch.cat(current_batch_x_list, dim=0).to(device)
                batch_target = torch.cat(current_batch_target_list, dim=0).to(device)
                
                # Original script used .double() for VAE. Adapt if necessary.
                # Assuming model and data are float32 for consistency.
                # If double: batch_x = batch_x.double(); batch_target = batch_target.double()
                batch_x = batch_x.float() 
                batch_target = batch_target.float()

                optimizer.zero_grad()
                reconstructed_x, mu, log_var = model(batch_x)
                
                # Use the VAE's internal loss function
                loss = model.loss_function(reconstructed_x, batch_target, mu, log_var, 
                                           reconstruction_loss_fn=reconstruction_loss_fn, 
                                           kld_weight=kld_weight)

                if torch.isnan(loss):
                    print("NaN loss encountered during VAE training! Skipping backward pass.")
                    # Optionally break or handle error
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0) # clip_grad_norm_ is often preferred
                    optimizer.step()
                    epoch_losses.append(loss.item())
                
                # Reset for next accumulation
                current_batch_x_list = []
                current_batch_target_list = []
                num_series_in_batch = 0
        pbar.set_postfix({"loss": np.mean(epoch_losses) if epoch_losses else 0.0})


    avg_epoch_loss = np.mean(epoch_losses) if epoch_losses else 0.0 # Avoid error if epoch_losses is empty
    print(f"Epoch VAE Training Loss: {avg_epoch_loss:.4f}")
    return float(avg_epoch_loss)


def train_moment_epoch(
    model: nn.Module, # MOMENTPipeline instance
    df_train_fold: pd.DataFrame,
    train_building_ids: List[Any],
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler], # e.g., CosineAnnealingLR
    loss_fn: nn.Module, # e.g., nn.MSELoss()
    mask_generator: Masking,
    device: torch.device,
    context_len: int,
    batch_size_script: int, # The BATCH_SIZE from the original script's single_run
    meter_reading_col: str = 'meter_reading',
    anomaly_col: str = 'anomaly',
    building_id_col: str = 'building_id'
) -> float:
    """
    Trains the MOMENT model for one epoch using the custom batching strategy.
    """
    model.train()
    epoch_losses = []

    list_of_loaders = []
    list_of_medians = []
    for b_id in train_building_ids:
        building_df = df_train_fold[df_train_fold[building_id_col] == b_id].copy()
        if building_df.empty:
            continue
        building_df.reset_index(drop=True, inplace=True)
        
        median_val = building_df[meter_reading_col].median()
        if pd.isna(median_val): median_val = 0.0
            
        dataset = TimeDataset(building_df, meter_reading_col=meter_reading_col, anomaly_col=anomaly_col)
        # Original MOMENT script used shuffle=True, drop_last=True
        loader = DataLoader(dataset, batch_size=context_len, shuffle=True, drop_last=True)
        if len(loader) > 0:
            list_of_loaders.append(iter(loader))
            list_of_medians.append(median_val)
            
    if not list_of_loaders:
        print("Warning: No data loaders created for MOMENT training epoch. Skipping.")
        return 0.0

    try:
        num_iterations = min(len(ldr.dataset) // context_len for ldr in list_of_loaders if hasattr(ldr, 'dataset') and len(ldr.dataset) // context_len > 0)
        num_iterations = max(1, num_iterations)
    except ValueError:
        num_iterations = 0

    if num_iterations == 0:
        print("Warning: Not enough data in some loaders for MOMENT training iteration. Skipping epoch.")
        return 0.0
        
    pbar = tqdm(range(num_iterations), desc="MOMENT Epoch Training")
    for _ in pbar:
        current_batch_x_list = []
        current_batch_input_masks_list = [] # For MOMENT's input_mask (padding)
        current_batch_target_list = []
        num_series_in_batch = 0

        for i, loader_iter in enumerate(list_of_loaders):
            try:
                single_x_series, placeholder_val, labels = next(loader_iter) # placeholder_val is the '1'
            except StopIteration:
                continue

            median_for_series = list_of_medians[i]
            labels_bool = labels.bool()

            target_x_series = single_x_series.clone().detach()
            target_x_series[labels_bool] = median_for_series
            
            # MOMENT expects [batch, channels, context_len] -> [1, 1, context_len] for a single series
            current_batch_x_list.append(single_x_series.unsqueeze(0).unsqueeze(0)) 
            current_batch_target_list.append(target_x_series.unsqueeze(0).unsqueeze(0))
            
            # input_mask for MOMENT (1s for real data, 0s for padding - here, all real)
            # placeholder_val was the '1' from TimeDataset, effectively a tensor of ones for input_mask
            input_mask_series = torch.ones_like(single_x_series, dtype=torch.bool).unsqueeze(0) # [1, context_len]
            current_batch_input_masks_list.append(input_mask_series)
            num_series_in_batch += 1

            if (num_series_in_batch % batch_size_script == 0) or (i == len(list_of_loaders) - 1 and num_series_in_batch > 0):
                if not current_batch_x_list: continue

                batch_x_enc = torch.cat(current_batch_x_list, dim=0).to(device).float()
                batch_target_rec = torch.cat(current_batch_target_list, dim=0).to(device).float()
                batch_input_masks = torch.cat(current_batch_input_masks_list, dim=0).to(device).bool() # [N, context_len]

                # Generate the random mask for the pre-training objective
                # The generated mask is [N, context_len]
                reconstruction_objective_mask = mask_generator.generate_mask(x=batch_x_enc).to(device) # bool

                optimizer.zero_grad()
                # MOMENT model forward pass
                # x_enc: [N, C, L], input_mask: [N, L], mask: [N, L] for reconstruction task
                output = model(x_enc=batch_x_enc, 
                               input_mask=batch_input_masks, # Identifies valid data vs padding
                               mask=reconstruction_objective_mask)  # Identifies which tokens to reconstruct

                loss = loss_fn(output.reconstruction, batch_target_rec) # Loss on reconstructed values

                if torch.isnan(loss):
                    print("NaN loss encountered during MOMENT training! Skipping backward pass.")
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                    optimizer.step()
                    epoch_losses.append(loss.item())
                
                current_batch_x_list = []
                current_batch_input_masks_list = []
                current_batch_target_list = []
                num_series_in_batch = 0
        pbar.set_postfix({"loss": np.mean(epoch_losses) if epoch_losses else 0.0})

    if scheduler: # Step scheduler after each epoch
        scheduler.step()

    avg_epoch_loss = np.mean(epoch_losses) if epoch_losses else 0.0
    print(f"Epoch MOMENT Training Loss: {avg_epoch_loss:.4f}")
    return float(avg_epoch_loss)

if __name__ == '__main__':
    print("This module provides training loop functions (train_vae_epoch, train_moment_epoch).")
    print("It is intended to be imported and used by training scripts.")
    # Add more detailed examples or tests if necessary,
    # but these functions require significant setup (model, data, optimizer).