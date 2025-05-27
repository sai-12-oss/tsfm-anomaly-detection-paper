# tsfm_ad_lib/data_loader.py
import pandas as pd
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Tuple, Any, Optional

from . import config 

class TimeDataset(Dataset):
    """
    A PyTorch Dataset for time series data from the LEAD competition.
    Assumes the DataFrame has 'meter_reading' and 'anomaly' columns.
    """
    def __init__(self, building_df: pd.DataFrame, meter_reading_col: str = 'meter_reading', anomaly_col: str = 'anomaly'):
        """
        Args:
            building_df (pd.DataFrame): DataFrame containing data for a single building.
                                        Must include meter_reading_col and anomaly_col.
            meter_reading_col (str): Name of the column with meter readings.
            anomaly_col (str): Name of the column with anomaly labels (0 or 1).
        """
        super().__init__()
        if not isinstance(building_df, pd.DataFrame):
            raise TypeError("building_df must be a pandas DataFrame.")
        if meter_reading_col not in building_df.columns:
            raise ValueError(f"Column '{meter_reading_col}' not found in DataFrame.")
        if anomaly_col not in building_df.columns:
            raise ValueError(f"Column '{anomaly_col}' not found in DataFrame.")
        
        self.df = building_df
        self.meter_reading_col = meter_reading_col
        self.anomaly_col = anomaly_col
        
    def __len__(self) -> int:
        """Returns the number of samples in the dataset."""
        return len(self.df)
    
    def __getitem__(self, idx: int) -> Tuple[Any, int, Any]:
        """
        Retrieves a sample from the dataset.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            Tuple[Any, int, Any]: A tuple containing:
                - meter_reading (float or appropriate type from DataFrame)
                - placeholder_value (int): Always 1, as seen in original scripts.
                                         Its purpose is unclear but replicated.
                - anomaly_label (int or appropriate type from DataFrame)
        """
        if not 0 <= idx < len(self.df):
            raise IndexError(f"Index {idx} out of bounds for dataset of length {len(self.df)}")
            
        row = self.df.iloc[idx]
        meter_reading = row[self.meter_reading_col]
        anomaly_label = row[self.anomaly_col]
        # doubt 
        placeholder_value = 1 
        
        return meter_reading, placeholder_value, anomaly_label

def load_raw_lead_data(csv_file_path: Optional[str] = None) -> pd.DataFrame:
    """
    Loads the raw LEAD dataset from a CSV file.

    Args:
        csv_file_path (Optional[str]): Path to the CSV file. 
                                       If None, uses DEFAULT_DATA_CSV_PATH from config.

    Returns:
        pd.DataFrame: The loaded DataFrame.

    Raises:
        FileNotFoundError: If the CSV file cannot be found.
        pd.errors.EmptyDataError: If the CSV file is empty.
        Exception: For other pandas read_csv errors.
    """
    if csv_file_path is None:
        csv_file_path = config.DEFAULT_DATA_CSV_PATH
    
    file_path = Path(csv_file_path)
    if not file_path.is_file():
        raise FileNotFoundError(f"Data file not found at {file_path}")

    try:
        df = pd.read_csv(file_path)
        if df.empty:
            raise pd.errors.EmptyDataError(f"No data found in {file_path}. File is empty.")
        print(f"Successfully loaded data from {file_path}. Shape: {df.shape}")
        return df
    except pd.errors.EmptyDataError as e:
        # Already handled above, but good to be explicit or log
        print(f"Error: {e}")
        raise
    except Exception as e:
        print(f"An error occurred while reading {file_path}: {e}")
        raise

if __name__ == '__main__':
    try:
        # Test loading default data
        print("Attempting to load data from default path...")
        raw_df = load_raw_lead_data()
        print(f"Loaded DataFrame head:\n{raw_df.head()}")

        # Example of using TimeDataset (requires a preprocessed 'building_df')
        if not raw_df.empty and 'building_id' in raw_df.columns:
            # Create a dummy building_df for testing TimeDataset
            # In real use, this df would be preprocessed
            example_building_id = raw_df['building_id'].unique()[0]
            dummy_building_df = raw_df[raw_df['building_id'] == example_building_id].copy()
            
            # Ensure required columns exist (they should from train.csv)
            if 'meter_reading' in dummy_building_df.columns and 'anomaly' in dummy_building_df.columns:
                print(f"\nCreating TimeDataset for building ID: {example_building_id}")
                time_series_dataset = TimeDataset(dummy_building_df)
                print(f"Dataset length: {len(time_series_dataset)}")
                if len(time_series_dataset) > 0:
                    sample_meter_reading, sample_placeholder, sample_anomaly = time_series_dataset[0]
                    print(f"First sample: Meter Reading={sample_meter_reading}, Placeholder={sample_placeholder}, Anomaly={sample_anomaly}")
            else:
                print("Skipping TimeDataset example: 'meter_reading' or 'anomaly' column missing in the dummy DataFrame.")
        else:
            print("Skipping TimeDataset example: DataFrame is empty or 'building_id' column is missing.")

    except FileNotFoundError as e:
        print(e)
    except Exception as e:
        print(f"An unexpected error occurred during example usage: {e}")