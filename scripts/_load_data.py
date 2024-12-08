import os
import pandas as pd


def load_data(data_dir="data"):
    """
    Loads accident and vehicle data from CSV files, merges them, and returns the merged DataFrame.

    Parameters:
        data_dir (str): Path to the directory containing the data files. Defaults to 'data'.

    Returns:
        pd.DataFrame: A merged DataFrame containing both accident and vehicle data.
    """
    # Define file paths
    accident_file = os.path.join(data_dir, "accident_data.csv")

    # Check if files exist
    if not os.path.exists(accident_file):
        raise FileNotFoundError(f"Accident data file not found in {accident_file}")

    # Load data with encoding handling
    df = pd.read_csv(accident_file, encoding="utf-8")  # to handle encoding

    return df
