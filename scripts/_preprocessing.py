import pandas as pd
import numpy as np


def drop_unneeded_columns(df, columns):
    """
    Drop unnecessary columns from the dataset.
    """
    return df.drop(columns=columns, axis=1)


def handle_missing_values(df, drop_na_columns=None):
    """
    Drop rows with missing values in the dataset.

    Parameters:
        df (pd.DataFrame): The input DataFrame.
        drop_na_columns (list): List of columns where rows with missing values should be dropped.
                                 If None, rows with missing values in any column will be dropped.

    Returns:
        pd.DataFrame: The DataFrame with rows containing missing values dropped.
    """
    if drop_na_columns:
        # Drop rows with missing values in specified columns
        df = df.dropna(subset=drop_na_columns).copy()
    else:
        # Drop rows with missing values in any column
        df = df.dropna().copy()

    return df


def handle_outliers(df, numerical_columns):
    """
    Handle outliers in the dataset using IQR.

    Parameters:
        df (pd.DataFrame): The input DataFrame.
        numerical_columns (list): List of numerical columns to handle outliers for.

    Returns:
        pd.DataFrame: The DataFrame with outliers handled.
    """
    for col in numerical_columns:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
    return df
