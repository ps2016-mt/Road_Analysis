import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder


def analyse_missing_values(df):
    """
    Analyse and visualise missing values in a DataFrame.
    """
    # Calculate missing values and their percentage
    missing_values = df.isnull().sum().sort_values(ascending=False)
    missing_percentage = (missing_values / len(df)) * 100
    missing_summary = pd.DataFrame(
        {"Missing Values": missing_values, "Percentage": missing_percentage}
    )

    # Print missing values summary
    print("Missing Values Summary:")
    print(missing_summary)

    # Visualize missing data as a heatmap
    plt.figure(figsize=(12, 6))
    sns.heatmap(df.isnull(), cbar=False, cmap="viridis")
    plt.title("Missing Values Heatmap")
    plt.show()

    return missing_summary


def drop_unneeded_columns(df, columns):
    """
    Drop unnecessary columns from the dataset.
    """
    return df.drop(columns=columns, axis=1)


def preprocess_and_plot_correlation(
    df, target_column, categorical_columns, severity_mapping
):
    """
    Preprocess the dataframe and plot a correlation heatmap.

    Parameters:
        df (pd.DataFrame): Original dataframe (remains unaffected).
        target_column (str): Column name of the target variable to encode.
        categorical_columns (list): List of categorical columns to encode.
        severity_mapping (dict): Mapping for target variable encoding.

    Returns:
        None
    """
    # Create a copy of the dataframe to avoid modifying the original
    df_copy = df.copy()

    # Encode target variable
    if target_column in df_copy.columns:
        df_copy[target_column] = df_copy[target_column].map(severity_mapping)

    # Encode categorical variables
    label_encoders = {}
    for col in categorical_columns:
        le = LabelEncoder()
        df_copy[col] = le.fit_transform(
            df_copy[col].astype(str)
        )  # Convert to string for consistency
        label_encoders[col] = le  # Save encoder for future use (if needed)

    # Handle missing values (drop rows with missing values)
    df_copy = df_copy.dropna()

    # Draw correlation heatmap
    plt.figure(figsize=(15, 12))  # Increased figure size for readability
    correlation_matrix = df_copy.corr()

    # Create heatmap
    sns.heatmap(
        correlation_matrix,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        cbar_kws={"label": "Correlation Coefficient"},
        linewidths=0.5,
    )

    plt.xticks(fontsize=10, rotation=45, ha="right")
    plt.yticks(fontsize=10)
    plt.title("Correlation Heatmap of Features", fontsize=16)
    plt.tight_layout()
    plt.show()


def check_unique_values(df, column=None):
    """
    Check for unique values in a specified column
    or all columns in the DataFrame.
    """
    if column:
        # Check for unique values in the specified column
        if column in df.columns:
            unique_values = df[column].unique()
            print(f"Unique values in column '{column}': {unique_values}")
            return unique_values
        else:
            raise ValueError(f"Column '{column}' not found in the DataFrame.")
    else:
        # Check for unique values in all columns
        unique_values_dict = {col: df[col].unique() for col in df.columns}
        for col, values in unique_values_dict.items():
            print(f"Unique values in column '{col}': {values}")
        return unique_values_dict


def drop_missing_values(df):
    """
    Remove all rows with missing values from the DataFrame.
    """
    cleaned_df = df.dropna()
    print(f"Original DataFrame had {df.isnull().sum().sum()} missing values.")
    return cleaned_df


def encode_target(df, target_column):
    """
    Encode the target variable for use in machine learning models.

    Parameters:
        df (pd.DataFrame): The dataset containing the target column.
        target_column (str): The name of the target column to encode.

    Returns:
        pd.DataFrame: Dataset with the encoded target column.
        dict: Mapping of original target labels to numerical values.
    """
    # Define the mapping
    target_mapping = {"Slight": 1, "Serious": 2, "Fatal": 3}

    # Encode the target column
    df[target_column] = df[target_column].map(target_mapping)

    return df, target_mapping
