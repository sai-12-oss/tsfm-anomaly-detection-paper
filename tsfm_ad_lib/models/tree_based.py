# tsfm_ad_lib/models/tree_based.py
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, precision_score, recall_score
from typing import Dict, List, Union, Optional, Any

from .. import config # To access default search ranges and params

def find_best_iforest_params_for_building(
    building_data: pd.DataFrame, 
    meter_reading_col: str = 'meter_reading',
    anomaly_col: str = 'anomaly',
    base_iforest_params: Optional[Dict[str, Any]] = None,
    contamination_search_values: Optional[List[float]] = None,
    scale_data: bool = True
) -> Dict[str, Union[float, List[float]]]:
    """
    Applies the Isolation Forest method to a single building's data to find
    the best 'contamination' parameter. The best contamination is selected by 
    maximizing the F1 score against true anomaly labels. Other IForest parameters
    are taken from `base_iforest_params`.

    Args:
        building_data (pd.DataFrame): DataFrame for a single building.
        meter_reading_col (str): Column name for meter readings.
        anomaly_col (str): Column name for true anomaly labels (0 or 1).
        base_iforest_params (Optional[Dict[str, Any]]): Base parameters for IsolationForest
            (e.g., n_estimators, max_samples). If None, uses defaults from config.
        contamination_search_values (Optional[List[float]]): A list of contamination
            values to test. If None, uses a default range derived from config.
        scale_data (bool): If True, meter readings are scaled using StandardScaler
                           before fitting Isolation Forest. Defaults to True.

    Returns:
        Dict[str, Union[float, List[float]]]: A dictionary containing the best F1 score,
            corresponding precision, recall, and the contamination value that achieved it.
    """
    if not isinstance(building_data, pd.DataFrame) or building_data.empty:
        raise ValueError("building_data must be a non-empty pandas DataFrame.")
    # ... (add column checks as in statistical.py functions) ...

    readings_series = building_data[meter_reading_col].copy()
    y_true = building_data[anomaly_col].values

    data_for_iforest_np: np.ndarray
    if scale_data:
        if readings_series.isnull().all():
             print(f"Warning: All '{meter_reading_col}' are NaN for building. Skipping scaling, F1 will be 0.")
             return {"f1": 0.0, "precision": 0.0, "recall": 0.0, "contamination": np.nan}
        scaler = StandardScaler()
        data_for_iforest_np = scaler.fit_transform(readings_series.values.reshape(-1, 1))
    else:
        # IsolationForest can handle NaNs if underlying trees can, but it's safer to ensure no NaNs
        # For consistency with scaled data, let's ensure it's a 2D numpy array
        if readings_series.isnull().any():
            print(f"Warning: NaNs found in '{meter_reading_col}' for building ID when scale_data=False. "
                  "Isolation Forest might behave unexpectedly. Consider imputing first.")
            # IForest might handle them or error depending on sklearn version and underlying tree.
            # For safety, one might impute here, or rely on prior global imputation.
            # For now, pass as is, but be aware.
        data_for_iforest_np = readings_series.values.reshape(-1, 1)
    
    if np.all(np.isnan(data_for_iforest_np)): # Check after potential scaling
        print(f"Warning: All '{meter_reading_col}' are NaN after processing for building. F1 will be 0.")
        return {"f1": 0.0, "precision": 0.0, "recall": 0.0, "contamination": np.nan}


    best_metrics = {"f1": 0.0, "precision": 0.0, "recall": 0.0, "contamination": np.nan}
    
    current_base_params = base_iforest_params if base_iforest_params is not None else config.DEFAULT_IFOREST_BASE_PARAMS.copy()
    
    if contamination_search_values is None:
        start, stop, step = config.DEFAULT_CONTAMINATION_SEARCH_RANGE
        current_contamination_search = list(np.arange(start, stop, step))
    else:
        current_contamination_search = contamination_search_values

    for cont_val in current_contamination_search:
        # Ensure contamination is within valid Sklearn range (0, 0.5]
        # Sklearn IForest contamination: "The amount of contamination of the data set, 
        # i.e. the proportion of outliers in the data set. Used when fitting to define the threshold 
        # on the scores of the samples. Should be in the interval (0, 0.5]."
        # Some of the original script's cont values (e.g. 0.001) are fine.
        # Values like 0 or > 0.5 would error or warn.
        if not (0 < cont_val <= 0.5):
            # print(f"Warning: Contamination value {cont_val:.4f} is outside (0, 0.5]. Skipping.")
            # Silently skip or adjust, for now, let sklearn handle it or error if invalid.
            # The original loop `range(1, 87, 5) / 1000` produced values like 0.001 to 0.086, which are valid.
            pass

        run_params = current_base_params.copy()
        run_params['contamination'] = cont_val
        
        try:
            model = IsolationForest(**run_params)
            # Fit on the (potentially scaled) meter readings
            model.fit(data_for_iforest_np) 
            # predict returns -1 for outliers (anomalies), 1 for inliers.
            y_pred_if = model.predict(data_for_iforest_np)
            y_pred_binary = np.where(y_pred_if == -1, 1, 0) # Convert to 0 (normal), 1 (anomaly)
        except Exception as e:
            print(f"Error during IsolationForest fit/predict for contamination {cont_val:.4f}: {e}. Skipping.")
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
        print("Loading raw data for Isolation Forest example...")
        raw_df_example = load_raw_lead_data()

        if not raw_df_example.empty and 'building_id' in raw_df_example.columns:
            example_building_id = raw_df_example['building_id'].unique()[0]
            building_df_for_test = raw_df_example[raw_df_example['building_id'] == example_building_id].copy()
            # Quick median imputation for the single building test data
            median_val = building_df_for_test['meter_reading'].median()
            building_df_for_test['meter_reading'].fillna(median_val, inplace=True)

            if building_df_for_test.empty or building_df_for_test['meter_reading'].isnull().all():
                 print(f"Skipping IForest example for building {example_building_id} due to no valid data.")
            else:
                print(f"\n--- Testing Isolation Forest for building ID: {example_building_id} ---")
                iforest_results = find_best_iforest_params_for_building(
                    building_df_for_test,
                    scale_data=True # As in original script
                )
                print(f"Isolation Forest Results: {iforest_results}")
        else:
            print("Raw data is empty or 'building_id' missing, skipping Isolation Forest example.")

    except FileNotFoundError as e:
        print(e)
    except Exception as e:
        print(f"An unexpected error occurred during Isolation Forest example usage: {e}")
        import traceback
        traceback.print_exc()