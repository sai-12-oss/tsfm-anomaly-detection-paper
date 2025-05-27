# tsfm_ad_lib/models/statistical.py
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, precision_score, recall_score
from typing import Dict, List, Union, Tuple, Optional

from .. import config # To access default search ranges

# --- IQR Method ---

def _calculate_iqr_bounds_on_series(
    data_series: pd.Series, 
    k_multiplier: float
) -> Tuple[float, float]:
    """
    Helper function to calculate IQR bounds for a 1D pandas Series.
    
    Args:
        data_series (pd.Series): The 1D data series.
        k_multiplier (float): The multiplier for the IQR.

    Returns:
        Tuple[float, float]: The lower and upper bounds.
    """
    if not isinstance(data_series, pd.Series):
        raise TypeError("data_series must be a pandas Series.")
    if data_series.empty or data_series.isnull().all():
        # Handle empty or all-NaN series to avoid errors in percentile
        return -np.inf, np.inf # Or raise error, or return (np.nan, np.nan)

    q1 = data_series.quantile(0.25)
    q3 = data_series.quantile(0.75)
    iqr_value = q3 - q1

    # Handle cases where IQR is zero (e.g., all values are the same)
    if iqr_value == 0:
        # If IQR is 0, bounds are typically just the q1/q3 or the value itself.
        # For anomaly detection, this means only exact deviations would be caught.
        # Or, one might define a minimum spread. For now, use q1/q3.
        return q1, q3 
        
    lower_bound = q1 - k_multiplier * iqr_value
    upper_bound = q3 + k_multiplier * iqr_value
    return lower_bound, upper_bound

def find_best_iqr_k_for_building(
    building_data: pd.DataFrame, 
    meter_reading_col: str = 'meter_reading',
    anomaly_col: str = 'anomaly',
    k_search_values: Optional[List[float]] = None,
    scale_data: bool = True
) -> Dict[str, Union[float, List[float]]]:
    """
    Applies the Interquartile Range (IQR) method to a single building's data
    to find the best K multiplier for defining outlier bounds. The best K is
    selected by maximizing the F1 score against true anomaly labels.

    Args:
        building_data (pd.DataFrame): DataFrame for a single building.
        meter_reading_col (str): Column name for meter readings.
        anomaly_col (str): Column name for true anomaly labels (0 or 1).
        k_search_values (Optional[List[float]]): A list of K multipliers to test.
            If None, uses DEFAULT_IQR_K_SEARCH_VALUES from config.
        scale_data (bool): If True, meter readings are scaled using StandardScaler
                           before applying IQR. Defaults to True as in original script.

    Returns:
        Dict[str, Union[float, List[float]]]: A dictionary containing the best F1 score,
            corresponding precision, recall, and the K value that achieved it.
            Example: {"f1": 0.8, "precision": 0.7, "recall": 0.9, "k": 1.5}
    """
    if not isinstance(building_data, pd.DataFrame) or building_data.empty:
        raise ValueError("building_data must be a non-empty pandas DataFrame.")
    if meter_reading_col not in building_data.columns:
        raise ValueError(f"Column '{meter_reading_col}' not found in building_data.")
    if anomaly_col not in building_data.columns:
        raise ValueError(f"Column '{anomaly_col}' not found in building_data.")

    readings_series = building_data[meter_reading_col].copy()
    y_true = building_data[anomaly_col].values # Ensure numpy array for scikit-learn metrics

    if scale_data:
        if readings_series.isnull().all(): # All NaNs, scaler will fail or produce NaNs
             print(f"Warning: All '{meter_reading_col}' are NaN for building. Skipping scaling, F1 will be 0.")
             # Return 0 scores as no meaningful bounds can be found
             return {"f1": 0.0, "precision": 0.0, "recall": 0.0, "k": np.nan}

        scaler = StandardScaler()
        # StandardScaler expects 2D input, reshape Series to [n_samples, 1]
        readings_scaled_np = scaler.fit_transform(readings_series.values.reshape(-1, 1)).flatten()
        data_for_iqr = pd.Series(readings_scaled_np, name=meter_reading_col)
    else:
        data_for_iqr = readings_series
    
    # Handle case where all values are NaN after potential scaling attempt (if original was all NaN)
    if data_for_iqr.isnull().all():
        print(f"Warning: All '{meter_reading_col}' are NaN after processing for building. F1 will be 0.")
        return {"f1": 0.0, "precision": 0.0, "recall": 0.0, "k": np.nan}


    best_metrics = {"f1": 0.0, "precision": 0.0, "recall": 0.0, "k": np.nan}
    current_k_search = k_search_values if k_search_values is not None else config.DEFAULT_IQR_K_SEARCH_VALUES

    for k_val in current_k_search:
        lower_b, upper_b = _calculate_iqr_bounds_on_series(data_for_iqr, k_val)
        
        anomalies_predicted = (data_for_iqr < lower_b) | (data_for_iqr > upper_b)
        y_pred_binary = anomalies_predicted.astype(int).values

        try:
            current_f1 = f1_score(y_true, y_pred_binary, zero_division=0)
            if current_f1 > best_metrics["f1"]:
                best_metrics["f1"] = current_f1
                best_metrics["precision"] = precision_score(y_true, y_pred_binary, zero_division=0)
                best_metrics["recall"] = recall_score(y_true, y_pred_binary, zero_division=0)
                best_metrics["k"] = k_val
        except ValueError as e: # Should not happen if y_true and y_pred_binary are valid 0/1 arrays
            print(f"Warning: Error calculating F1 score for K={k_val}. {e}. Skipping this K.")
            continue
            
    return best_metrics

# --- Modified Z-score Method ---

def _calculate_modified_zscore_on_series(
    data_series: pd.Series, 
    use_absolute_for_final_score: bool = True,
    consistency_constant: float = 1.4826
) -> pd.Series:
    """
    Helper function to calculate Modified Z-scores for a 1D pandas Series.

    Args:
        data_series (pd.Series): The 1D data series.
        use_absolute_for_final_score (bool): If True, the final Z-score numerator
            uses the absolute difference from the median (as in original script).
            If False, it uses the signed difference, preserving direction.
        consistency_constant (float): Constant used to scale MAD to an estimate of
                                      standard deviation (commonly 1.4826 for normal data).
    Returns:
        pd.Series: Series containing the Modified Z-scores.
    """
    if not isinstance(data_series, pd.Series):
        raise TypeError("data_series must be a pandas Series.")
    if data_series.empty or data_series.isnull().all():
        return pd.Series([np.nan] * len(data_series), index=data_series.index)


    values_1d = data_series.values.flatten() # Ensure 1D numpy array
    median_val = np.nanmedian(values_1d) # Use nanmedian to be robust to NaNs

    if np.isnan(median_val): # All values were NaN
        return pd.Series([np.nan] * len(values_1d), index=data_series.index)

    # For MAD calculation, differences are absolute from the (nan)median
    mad_diff = np.abs(values_1d - median_val)
    mad = np.nanmedian(mad_diff) # Median Absolute Deviation
    
    epsilon = 1e-9 # Avoid division by zero if MAD is 0
    sigma_from_mad = consistency_constant * mad + epsilon
    
    if sigma_from_mad == epsilon: # MAD was 0
        # If MAD is 0, all non-median values are infinitely far.
        # Return 0 for values equal to median, inf for others.
        # Or, based on problem, could return all 0s. Let's make diffs 0 where value is median.
        # For now, if MAD is 0, z-scores will be 0 for points at median, large for others.
        # This case makes z-scores very sensitive.
        # Original script added epsilon to sigma_mad, so this path is fine.
        pass


    if use_absolute_for_final_score:
        final_z_scores_numerator = np.abs(values_1d - median_val)
    else:
        final_z_scores_numerator = values_1d - median_val
            
    modified_z_scores = final_z_scores_numerator / sigma_from_mad
    return pd.Series(modified_z_scores, index=data_series.index)

def find_best_mzscore_k_for_building(
    building_data: pd.DataFrame, 
    meter_reading_col: str = 'meter_reading',
    anomaly_col: str = 'anomaly',
    k_search_values: Optional[List[float]] = None,
    scale_data: bool = True,
    use_absolute_zscore_in_calc: bool = True
) -> Dict[str, Union[float, List[float]]]:
    """
    Applies the Modified Z-score method to a single building's data to find
    the best K threshold for defining outliers. The best K is selected by
    maximizing the F1 score against true anomaly labels.

    Args:
        building_data (pd.DataFrame): DataFrame for a single building.
        meter_reading_col (str): Column name for meter readings.
        anomaly_col (str): Column name for true anomaly labels (0 or 1).
        k_search_values (Optional[List[float]]): A list of K thresholds to test.
            If None, uses DEFAULT_MZSCORE_K_SEARCH_VALUES from config.
        scale_data (bool): If True, meter readings are scaled using StandardScaler
                           before applying Modified Z-score. Defaults to True.
        use_absolute_zscore_in_calc (bool): Passed to _calculate_modified_zscore.
            If True (default, matches original script), Z-scores are always positive.

    Returns:
        Dict[str, Union[float, List[float]]]: A dictionary containing the best F1 score,
            corresponding precision, recall, and the K value that achieved it.
    """
    if not isinstance(building_data, pd.DataFrame) or building_data.empty:
        raise ValueError("building_data must be a non-empty pandas DataFrame.")
    # ... (add column checks as in IQR function) ...

    readings_series = building_data[meter_reading_col].copy()
    y_true = building_data[anomaly_col].values

    if scale_data:
        if readings_series.isnull().all():
             print(f"Warning: All '{meter_reading_col}' are NaN for building. Skipping scaling, F1 will be 0.")
             return {"f1": 0.0, "precision": 0.0, "recall": 0.0, "k": np.nan}
        scaler = StandardScaler()
        readings_scaled_np = scaler.fit_transform(readings_series.values.reshape(-1, 1)).flatten()
        data_for_mz = pd.Series(readings_scaled_np, name=meter_reading_col)
    else:
        data_for_mz = readings_series

    if data_for_mz.isnull().all():
        print(f"Warning: All '{meter_reading_col}' are NaN after processing for building. F1 will be 0.")
        return {"f1": 0.0, "precision": 0.0, "recall": 0.0, "k": np.nan}

    mz_scores = _calculate_modified_zscore_on_series(
        data_for_mz, 
        use_absolute_for_final_score=use_absolute_zscore_in_calc
    )
    
    if mz_scores.isnull().all(): # If mz_score calculation resulted in all NaNs
        print(f"Warning: All Modified Z-scores are NaN for building. F1 will be 0.")
        return {"f1": 0.0, "precision": 0.0, "recall": 0.0, "k": np.nan}


    best_metrics = {"f1": 0.0, "precision": 0.0, "recall": 0.0, "k": np.nan}
    current_k_search = k_search_values if k_search_values is not None else config.DEFAULT_MZSCORE_K_SEARCH_VALUES

    for k_val in current_k_search:
        # Anomalies if |mz_score| > k_val.
        # If use_absolute_zscore_in_calc is True, mz_scores are already >= 0,
        # so np.abs() is still correct and general.
        anomalies_predicted = np.abs(mz_scores.fillna(0).values) > k_val # fillna(0) for safety if some mz_scores are NaN
        y_pred_binary = anomalies_predicted.astype(int)

        try:
            current_f1 = f1_score(y_true, y_pred_binary, zero_division=0)
            if current_f1 > best_metrics["f1"]:
                best_metrics["f1"] = current_f1
                best_metrics["precision"] = precision_score(y_true, y_pred_binary, zero_division=0)
                best_metrics["recall"] = recall_score(y_true, y_pred_binary, zero_division=0)
                best_metrics["k"] = k_val
        except ValueError as e:
            print(f"Warning: Error calculating F1 score for K={k_val}. {e}. Skipping this K.")
            continue
            
    return best_metrics


if __name__ == '__main__':
    # Example Usage
    from ..data_loader import load_raw_lead_data # Relative import for example
    from ..preprocessing import preprocess_lead_data_by_building

    try:
        print("Loading raw data for statistical model examples...")
        raw_df_example = load_raw_lead_data()

        if not raw_df_example.empty and 'building_id' in raw_df_example.columns:
            example_building_id = raw_df_example['building_id'].unique()[0]
            # Using already preprocessed data (median imputed, but not scaled here) for simplicity
            # The functions themselves handle scaling if scale_data=True
            building_df_for_stat_test = raw_df_example[raw_df_example['building_id'] == example_building_id].copy()
            # Quick median imputation for the single building test data
            median_val = building_df_for_stat_test['meter_reading'].median()
            building_df_for_stat_test['meter_reading'].fillna(median_val, inplace=True)


            if building_df_for_stat_test.empty or building_df_for_stat_test['meter_reading'].isnull().all():
                print(f"Skipping examples for building {example_building_id} due to no valid data.")
            else:
                print(f"\n--- Testing IQR for building ID: {example_building_id} ---")
                iqr_results = find_best_iqr_k_for_building(
                    building_df_for_stat_test,
                    scale_data=True # As in original script
                )
                print(f"IQR Results: {iqr_results}")

                print(f"\n--- Testing Modified Z-score for building ID: {example_building_id} ---")
                mz_results = find_best_mzscore_k_for_building(
                    building_df_for_stat_test,
                    scale_data=True, # As in original script
                    use_absolute_zscore_in_calc=True # As in original script
                )
                print(f"Modified Z-score Results: {mz_results}")
        else:
            print("Raw data is empty or 'building_id' missing, skipping statistical model examples.")

    except FileNotFoundError as e:
        print(e)
    except Exception as e:
        print(f"An unexpected error occurred during statistical model example usage: {e}")
        import traceback
        traceback.print_exc()