# tsfm_ad_lib/preprocessing.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from typing import List, Optional
from typing import List, Optional, Any, Union # Add Union if used elsewhere
def preprocess_lead_data_by_building(
    raw_df: pd.DataFrame,
    building_id_col: str = 'building_id',
    meter_reading_col: str = 'meter_reading',
    scale_meter_reading: bool = False,
    building_ids_to_process: Optional[List[Any]] = None
) -> pd.DataFrame:
    """
    Preprocesses the LEAD dataset by performing median imputation for 'meter_reading'
    for each building. Optionally scales the 'meter_reading' column using StandardScaler
    for each building.

    Args:
        raw_df (pd.DataFrame): The raw input DataFrame.
        building_id_col (str): Name of the column identifying unique buildings.
        meter_reading_col (str): Name of the column containing meter readings.
        scale_meter_reading (bool): If True, applies StandardScaler to 'meter_reading'
                                    for each building after imputation.
        building_ids_to_process (Optional[List[Any]]): A list of building IDs to process.
                                                       If None, all unique building IDs
                                                       in raw_df will be processed.

    Returns:
        pd.DataFrame: A new DataFrame with processed 'meter_reading' data, concatenated
                      for all processed buildings.

    Raises:
        ValueError: If essential columns are missing from raw_df.
    """
    if not isinstance(raw_df, pd.DataFrame):
        raise TypeError("raw_df must be a pandas DataFrame.")
    if building_id_col not in raw_df.columns:
        raise ValueError(f"Building ID column '{building_id_col}' not found in DataFrame.")
    if meter_reading_col not in raw_df.columns:
        raise ValueError(f"Meter reading column '{meter_reading_col}' not found in DataFrame.")

    processed_dfs = []
    
    if building_ids_to_process is None:
        unique_ids = sorted(raw_df[building_id_col].unique()) # Sort for consistent order
    else:
        unique_ids = sorted(list(set(building_ids_to_process))) # Ensure unique and sorted

    if not unique_ids:
        print("Warning: No building IDs to process.")
        return pd.DataFrame(columns=raw_df.columns)

    print(f"Starting preprocessing for {len(unique_ids)} building(s)...")
    for b_id in unique_ids:
        # Filter data for the current building
        building_df_slice = raw_df[raw_df[building_id_col] == b_id]
        
        if building_df_slice.empty:
            print(f"Warning: No data found for building ID {b_id}. Skipping.")
            continue
            
        # Make a copy to avoid SettingWithCopyWarning
        building_df = building_df_slice.copy()

        # 1. Median Imputation for NaNs in meter_reading
        median_val = building_df[meter_reading_col].median()
        if pd.isna(median_val): 
            # If all values are NaN for this building, median is NaN.
            # Fill with 0 or handle as per a defined strategy.
            # For now, if median is NaN, NaNs might persist or be filled with 0.
            # print(f"Warning: Median for building {b_id} is NaN. Filling NaNs with 0 for {meter_reading_col}.")
            # building_df[meter_reading_col] = building_df[meter_reading_col].fillna(0)
            # Let's replicate original behavior which would effectively leave NaNs if median is NaN,
            # or if all values were NaN, StandardScaler would fail.
            # A robust strategy would be to check if median_val is NaN and decide.
            # Original scripts seemed to assume median would be valid.
             pass # Let it fill with NaN median if that's the case; scaler will handle it or error

        building_df[meter_reading_col] = building_df[meter_reading_col].fillna(median_val)
        
        # 2. Optional Scaling
        if scale_meter_reading:
            if building_df[meter_reading_col].isnull().any():
                # This case should ideally not happen if median_val was valid and filled.
                # If median_val itself was NaN (e.g., all-NaN column for a building),
                # then NaNs would persist. StandardScaler cannot handle NaNs.
                print(f"Warning: NaNs found in '{meter_reading_col}' for building {b_id} before scaling. Filling with 0.")
                building_df[meter_reading_col] = building_df[meter_reading_col].fillna(0)

            scaler = StandardScaler()
            # StandardScaler expects 2D array: [n_samples, n_features]
            scaled_values = scaler.fit_transform(building_df[[meter_reading_col]])
            building_df[meter_reading_col] = scaled_values.flatten() # Assign back as 1D

        building_df.reset_index(drop=True, inplace=True)
        processed_dfs.append(building_df)

    if not processed_dfs:
        print("Warning: No dataframes were processed.")
        return pd.DataFrame(columns=raw_df.columns) # Return empty DF with same columns

    concatenated_df = pd.concat(processed_dfs, ignore_index=True)
    print(f"Preprocessing complete. Shape of processed data: {concatenated_df.shape}")
    return concatenated_df


if __name__ == '__main__':
    # Example usage:
    from .data_loader import load_raw_lead_data # Relative import for sibling module

    try:
        print("Loading raw data for preprocessing example...")
        raw_df_example = load_raw_lead_data()

        if not raw_df_example.empty:
            # Example 1: Preprocessing without scaling (like for MOMENT, IForest, etc.)
            print("\n--- Preprocessing without scaling ---")
            imputed_only_df = preprocess_lead_data_by_building(
                raw_df_example, 
                scale_meter_reading=False
            )
            print(f"Imputed data head:\n{imputed_only_df.head()}")
            if 'meter_reading' in imputed_only_df.columns:
                 print(f"NaNs in meter_reading after imputation: {imputed_only_df['meter_reading'].isnull().sum()}")


            # Example 2: Preprocessing with scaling (like for VAE)
            print("\n--- Preprocessing with scaling ---")
            scaled_df = preprocess_lead_data_by_building(
                raw_df_example,
                scale_meter_reading=True
            )
            print(f"Scaled data head:\n{scaled_df.head()}")
            if 'meter_reading' in scaled_df.columns:
                print(f"NaNs in meter_reading after scaling: {scaled_df['meter_reading'].isnull().sum()}")
                print(f"Mean of scaled meter_reading (should be close to 0 if many buildings): {scaled_df['meter_reading'].mean():.4f}")
                print(f"Std of scaled meter_reading (should be close to 1 if many buildings): {scaled_df['meter_reading'].std():.4f}")

            # Example 3: Processing only specific building IDs
            if 'building_id' in raw_df_example.columns and len(raw_df_example['building_id'].unique()) > 2:
                some_ids = raw_df_example['building_id'].unique()[:2].tolist()
                print(f"\n--- Preprocessing for specific building IDs: {some_ids} ---")
                partial_df = preprocess_lead_data_by_building(
                    raw_df_example,
                    scale_meter_reading=True,
                    building_ids_to_process=some_ids
                )
                print(f"Partial processed data head:\n{partial_df.head()}")
                print(f"Unique building IDs in partial_df: {partial_df['building_id'].unique()}")

        else:
            print("Raw data is empty, skipping preprocessing examples.")
            
    except FileNotFoundError as e:
        print(e)
    except Exception as e:
        print(f"An unexpected error occurred during example usage: {e}")