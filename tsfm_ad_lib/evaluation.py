# tsfm_ad_lib/evaluation.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import pandas as pd
from typing import List, Any, Callable, Optional

from .data_loader import TimeDataset # Assuming TimeDataset is the primary dataset type

# Placeholder for adjbestf1 - this should come from momentfm or be reimplemented
# For now, let's define a dummy one for the code to run.
# In a real setup, ensure momentfm is installed and importable, or provide the actual implementation.
try:
    from momentfm.utils.anomaly_detection_metrics import adjbestf1
except ImportError:
    print("Warning: momentfm.utils.anomaly_detection_metrics.adjbestf1 not found. Using a dummy adjbestf1.")
    def adjbestf1(y_true: np.ndarray, y_scores: np.ndarray) -> float:
        """Dummy adjbestf1. Replace with actual implementation or import."""
        if len(y_true) == 0 or len(y_scores) == 0: return 0.0
        # This is a placeholder. A real F1 calculation would involve thresholds.
        # For adjbestf1, it finds the best threshold on y_scores.
        # Here, just returning a random value or simple F1 for placeholder.
        from sklearn.metrics import f1_score
        # Simple thresholding for dummy:
        threshold = np.median(y_scores)
        y_pred = (y_scores > threshold).astype(int)
        return f1_score(y_true, y_pred, zero_division=0)

def evaluate_vae_model(
    model: nn.Module, # VarEncoderDecoder instance
    df_val_fold: pd.DataFrame,
    val_building_ids: List[Any],
    device: torch.device,
    context_len: int, # Max sequence length for padding
    meter_reading_col: str = 'meter_reading',
    anomaly_col: str = 'anomaly',
    building_id_col: str = 'building_id'
) -> float:
    """
    Evaluates the VAE model on validation data for specified building IDs.
    Calculates anomaly scores based on reconstruction error and uses adjbestf1.

    Args:
        model: The trained VAE model instance.
        df_val_fold: DataFrame containing validation data for the current fold.
        val_building_ids: List of building IDs to evaluate.
        device: The torch device.
        context_len: The sequence length expected by the model; shorter sequences
                     will be padded to this length.
        meter_reading_col (str): Name of the meter reading column.
        anomaly_col (str): Name of the anomaly label column.
        building_id_col (str): Name of the building ID column.
        
    Returns:
        float: The average adjbestf1 score across all evaluated building IDs.
    """
    model.eval()
    all_building_scores = []

    for b_id in val_building_ids:
        building_df = df_val_fold[df_val_fold[building_id_col] == b_id].copy()
        if building_df.empty:
            continue
        building_df.reset_index(drop=True, inplace=True)
        
        dataset = TimeDataset(building_df, meter_reading_col=meter_reading_col, anomaly_col=anomaly_col)
        # Original script used batch_size=512 for VAE testing, shuffle=False, drop_last=False
        loader = DataLoader(dataset, batch_size=context_len, shuffle=False, drop_last=False) 

        building_true_readings = []
        building_reconstructed_readings = []
        building_true_labels = []

        with torch.no_grad():
            for batch_x_series, _, batch_labels_series in loader: # tqdm(loader, desc=f"Eval VAE Bldg {b_id}", leave=False)
                # batch_x_series is [N, context_len] where N <= context_len (last batch)
                # or [actual_batch_size_from_loader, context_len]
                # The original script's test_model used batch_size=512 for loader, but context_len for model input.
                # It then unsqueezed batch_x to [1, 512] and padded if length < 512.
                # This implies it processes one full (or padded) sequence at a time for eval.
                # Let's refine based on that: each item from loader is a full sequence.

                current_seq_len = batch_x_series.shape[0] # Should be context_len unless it's the last partial batch
                                                        # Wait, the original loader batch_size was 512 for testing.
                                                        # And context_len was 512 for model.
                                                        # This means each item from loader *is* the batch_x for model
                                                        # if batch_size=context_len for loader.

                # Replicate original logic: pad if current sequence from TimeDataset is shorter than context_len
                # This typically happens on the *last* batch from DataLoader if drop_last=False
                padded_batch_x = batch_x_series.clone().detach()
                if current_seq_len < context_len:
                    padding = torch.zeros(context_len - current_seq_len, device=device)
                    padded_batch_x = torch.cat([padded_batch_x, padding], dim=0)

                # Model expects [batch_size (1 for this eval style), seq_len]
                input_to_model = padded_batch_x.unsqueeze(0).to(device)
                # If model is .double(), input_to_model = input_to_model.double() else .float()
                input_to_model = input_to_model.float()


                reconstructed_x, _, _ = model(input_to_model) # mu, log_var not needed for score here
                
                # Store only the original, non-padded parts
                building_true_readings.extend(input_to_model.squeeze(0)[:current_seq_len].cpu().numpy())
                building_reconstructed_readings.extend(reconstructed_x.squeeze(0)[:current_seq_len].cpu().numpy())
                building_true_labels.extend(batch_labels_series.cpu().numpy()) # batch_labels_series corresponds to current_seq_len

        if not building_true_labels: # No data processed for this building
            all_building_scores.append(0.0) # Or handle as NaN / skip
            continue

        true_readings_np = np.array(building_true_readings)
        reconstructed_readings_np = np.array(building_reconstructed_readings)
        true_labels_np = np.array(building_true_labels)
        
        # Anomaly score is squared reconstruction error
        anomaly_scores = (true_readings_np - reconstructed_readings_np)**2
        
        score = adjbestf1(y_true=true_labels_np, y_scores=anomaly_scores)
        all_building_scores.append(score)

    avg_score = np.mean(all_building_scores) if all_building_scores else 0.0
    print(f"Average VAE Validation adjBestF1: {avg_score:.4f}")
    return float(avg_score)


def evaluate_moment_model(
    model: nn.Module, # MOMENTPipeline instance
    df_val_fold: pd.DataFrame,
    val_building_ids: List[Any],
    device: torch.device,
    context_len: int, # Max sequence length for padding
    meter_reading_col: str = 'meter_reading',
    anomaly_col: str = 'anomaly',
    building_id_col: str = 'building_id'
) -> float:
    """
    Evaluates the fine-tuned MOMENT model on validation data.
    """
    model.eval()
    all_building_scores = []

    for b_id in val_building_ids:
        building_df = df_val_fold[df_val_fold[building_id_col] == b_id].copy()
        if building_df.empty:
            continue
        building_df.reset_index(drop=True, inplace=True)

        dataset = TimeDataset(building_df, meter_reading_col=meter_reading_col, anomaly_col=anomaly_col)
        # Original script used batch_size=512 for MOMENT testing
        loader = DataLoader(dataset, batch_size=context_len, shuffle=False, drop_last=False)

        building_true_readings = []
        building_reconstructed_readings = []
        building_true_labels = []

        with torch.no_grad():
            for batch_x_series, _, batch_labels_series in loader: # tqdm(loader, desc=f"Eval MOMENT Bldg {b_id}", leave=False)
                current_seq_len = batch_x_series.shape[0]

                # Prepare input for MOMENT: x_enc and input_mask
                padded_batch_x_series = batch_x_series.clone().detach()
                # The input_mask indicates valid data points (1s for real, 0s for padding)
                current_input_mask = torch.ones(current_seq_len, dtype=torch.bool, device=device)

                if current_seq_len < context_len:
                    padding_x = torch.zeros(context_len - current_seq_len, device=device)
                    padded_batch_x_series = torch.cat([padded_batch_x_series, padding_x], dim=0)
                    
                    padding_mask = torch.zeros(context_len - current_seq_len, dtype=torch.bool, device=device)
                    current_input_mask = torch.cat([current_input_mask, padding_mask], dim=0)

                # MOMENT expects x_enc: [N, C, L], input_mask: [N, L]
                # For eval, N=1 (one sequence at a time), C=1
                x_enc_model = padded_batch_x_series.unsqueeze(0).unsqueeze(0).to(device).float()
                input_mask_model = current_input_mask.unsqueeze(0).to(device).bool()
                
                output = model(x_enc=x_enc_model, input_mask=input_mask_model) # No random 'mask' for eval

                building_true_readings.extend(x_enc_model.squeeze(0).squeeze(0)[:current_seq_len].cpu().numpy())
                building_reconstructed_readings.extend(output.reconstruction.squeeze(0).squeeze(0)[:current_seq_len].cpu().numpy())
                building_true_labels.extend(batch_labels_series.cpu().numpy())
        
        if not building_true_labels:
            all_building_scores.append(0.0)
            continue

        true_readings_np = np.array(building_true_readings)
        reconstructed_readings_np = np.array(building_reconstructed_readings)
        true_labels_np = np.array(building_true_labels)
        
        anomaly_scores = (true_readings_np - reconstructed_readings_np)**2
        score = adjbestf1(y_true=true_labels_np, y_scores=anomaly_scores)
        all_building_scores.append(score)

    avg_score = np.mean(all_building_scores) if all_building_scores else 0.0
    print(f"Average MOMENT Validation adjBestF1: {avg_score:.4f}")
    return float(avg_score)

if __name__ == '__main__':
    print("This module provides evaluation functions (evaluate_vae_model, evaluate_moment_model).")
    print("It is intended to be imported and used by training/evaluation scripts.")
    # Add examples if simple model mockups can be created for testing.