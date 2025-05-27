# tsfm_ad_lib/models/neighbors.py
import numpy as np
import pandas as pd
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, precision_score, recall_score
from typing import Dict, List, Union, Optional, Any

from .. import config # To access default search ranges and params

def find_best_lof_params_for_building(
    building_data: pd.DataFrame, 
    meter_reading_col: str = 'meter_reading',
    anomaly_col: str = 'anomaly',
    base_lof_params: Optional[Dict[str, Any]] = None,
    contamination_search_values: Optional[List[float]] = None,
    scale_data: bool = True
) -> Dict[str, Union[float, List[float]]]:
    """
    Applies the Local Outlier Factor (LOF) method to a single building's data 
    to find the best 'contamination' parameter. The best contamination is selected 
    by maximizing the F1 score against true anomaly labels. Other LOF parameters
    are taken from `base_lof_params`. Assumes `novelty=False` for LOF.

    Args:
        building_data (pd.DataFrame): DataFrame for a single building.
        meter_reading_col (str): Column name for meter readings.
        anomaly_col (str): Column name for true anomaly labels (0 or 1).
        base_lof_params (Optional[Dict[str, Any]]): Base parameters for LocalOutlierFactor
            (e.g., n_neighbors). If None, uses defaults from config. 
            'novelty' will be forced to False.
        contamination_search_values (Optional[List[float]]): A list of contamination
            values to test. If None, uses a default range derived from config.
        scale_data (bool): If True, meter readings are scaled using StandardScaler
                           before fitting LOF. Defaults to True.

    Returns:
        Dict[str, Union[float, List[float]]]: A dictionary containing the best F1 score,
            corresponding precision, recall, and the contamination value that achieved it.
    """
    if not isinstance(building_data, pd.DataFrame) or building_data.empty:
        raise ValueError("building_data must be a non-empty pandas DataFrame.")
    # ... (add column checks) ...

    readings_series = building_data[meter_reading_col].copy()
    y_true = building_data[anomaly_col].values

    data_for_lof_np: np.ndarray
    if scale_data:
        if readings_series.isnull().all():
             print(f"Warning: All '{meter_reading_col}' are NaN for building. Skipping scaling, F1 will be 0.")
             return {"f1": 0.0, "precision": 0.0, "recall": 0.0, "contamination": np.nan}
        scaler = StandardScaler()
        data_for_lof_np = scaler.fit_transform(readings_series.values.reshape(-1, 1))
    else:
        # LOF also expects no NaNs typically
        if readings_series.isnull().any():
             print(f"Warning: NaNs found in '{meter_reading_col}' for building ID when scale_data=False. "
                  "LOF might error or behave unexpectedly. Consider imputing first.")
        data_for_lof_np = readings_series.values.reshape(-1, 1)
        
    if np.all(np.isnan(data_for_lof_np)):
        print(f"Warning: All '{meter_reading_col}' are NaN after processing for building. F1 will be 0.")
        return {"f1": 0.0, "precision": 0.0, "recall": 0.0, "contamination": np.nan}

    best_metrics = {"f1": 0.0, "precision": 0.0, "recall": 0.0, "contamination": np.nan}
    
    current_base_params = base_lof_params if base_lof_params is not None else config.DEFAULT_LOF_BASE_PARAMS.copy()
    current_base_params['novelty'] = False # Ensure novelty is False for fit_predict behavior

    if contamination_search_values is None:
        start, stop, step = config.DEFAULT_CONTAMINATION_SEARCH_RANGE
        current_contamination_search = list(np.arange(start, stop, step))
    else:
        current_contamination_search = contamination_search_values
        
    for cont_val in current_contamination_search:
        if not (0 < cont_val <= 0.5): # Valid Sklearn LOF contamination range
            # print(f"Warning: LOF Contamination value {cont_val:.4f} is outside (0, 0.5]. Skipping.")
            # As with IForest, original script's range was valid.
            pass

        run_params = current_base_params.copy()
        run_params['contamination'] = cont_val
        
        try:
            # n_samples must be greater than n_neighbors for LOF to work properly
            if data_for_lof_np.shape[0] <= run_params.get('n_neighbors', 20): # Default n_neighbors is 20 in sklearn
                print(f"Warning: n_samples ({data_for_lof_np.shape[0]}) <= n_neighbors "
                      f"({run_params.get('n_neighbors', 20)}) for LOF. Skipping building or using smaller n_neighbors.")
                # Could try to adapt n_neighbors, or skip. For now, let it potentially error or warn from sklearn.
                # A robust approach would be: run_params['n_neighbors'] = max(1, data_for_lof_np.shape[0] - 1)
                if data_for_lof_np.shape[0] <=1: continue # Cannot run LOF
                run_params['n_neighbors'] = max(1, data_for_lof_np.shape[0] -1)


            model = LocalOutlierFactor(**run_params)
            # fit_predict returns -1 for outliers (anomalies), 1 for inliers.
            y_pred_lof = model.fit_predict(data_for_lof_np)
            y_pred_binary = np.where(y_pred_lof == -1, 1, 0) # Convert to 0 (normal), 1 (anomaly)
        except Exception as e:
            print(f"Error during LocalOutlierFactor fit/predict for contamination {cont_val:.4f}: {e}. Skipping.")
            continue

        try:
            current_f1 = f1_score(y_true, y_pred_binary, zero_division=0)
            if current_f1 > best_metrics["f1"]:
                best_metrics["f1"] = current_f1
                best_metrics["precision"] = precision_score(y_true, y_pred_binary, zero_division=0)
                best_metrics["recall"] = recall_score(y_true, y_pred_binary, zero_division=0)
                best_metrics["contamination"] = cont_val
        except ValueError as e:
            print(f"Warning: Error calculating F1 score for contamination {cont_val:.4f}. {e}. Skipping this K.")
            continue
            
    return best_metrics

if __name__ == '__main__':
    from ..data_loader import load_raw_lead_data # Relative import for example
    
    try:
        print("Loading raw data for LOF example...")
        raw_df_example = load_raw_lead_data()

        if not raw_df_example.empty and 'building_id' in raw_df_example.columns:
            example_building_id = raw_df_example['building_id'].unique()[0]
            building_df_for_test = raw_df_example[raw_df_example['building_id'] == example_building_id].copy()
            median_val = building_df_for_test['meter_reading'].median() # Quick imputation for example
            building_df_for_test['meter_reading'].fillna(median_val, inplace=True)

            if building_df_for_test.empty or building_df_for_test['meter_reading'].isnull().all():
                 print(f"Skipping LOF example for building {example_building_id} due to no valid data.")
            else:
                print(f"\n--- Testing Local Outlier Factor for building ID: {example_building_id} ---")
                # Using default LOF params from config, which might be aggressive for small datasets
                lof_results = find_best_lof_params_for_building(
                    building_df_for_test,
                    scale_data=True # As in original script
                )
                print(f"LOF Results: {lof_results}")
        else:
            print("Raw data is empty or 'building_id' missing, skipping LOF example.")

    except FileNotFoundError as e:
        print(e)
    except Exception as e:
        print(f"An unexpected error occurred during LOF example usage: {e}")
        import traceback
        traceback.print_exc()