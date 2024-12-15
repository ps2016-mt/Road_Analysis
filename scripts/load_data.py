import os
import pandas as pd


def load_data(data_dir="data"):
    """
    Loads accident data from CSV files, and returns a DataFrame.
    """
    # Define file paths
    accident_file = os.path.join(data_dir, "accident_data.csv")

    # Check if files exist
    if not os.path.exists(accident_file):
        raise FileNotFoundError(f"File not found in {accident_file}")

    # Load data with encoding handling
    df = pd.read_csv(accident_file, encoding="utf-8")  # to handle encoding

    return df
