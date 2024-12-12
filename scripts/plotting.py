import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def plot_distribution_share(df, column, figsize=(10, 6), rotation=45):
    """
    Plot the distribution of a categorical variable as a share (percentage)
    with improved axis readability.
    Parameters: df (pd.DataFrame): The dataframe containing the data.
                column (str): The column to plot.
                figsize (tuple): Size of the figure.
                rotation (int): Angle for rotating x-axis labels.
    """
    # Calculate the percentage distribution
    value_counts = df[column].value_counts(normalize=True) * 100
    percentages = value_counts.sort_index()

    plt.figure(figsize=figsize)
    sns.barplot(x=percentages.index,
                y=percentages.values,
                order=value_counts.index)
    plt.title(f"Distribution of {column} (as Share)")
    plt.xlabel(column)
    plt.ylabel("Percentage")
    plt.xticks(rotation=rotation,
               ha="right")
    plt.tight_layout()  # Adjust layout to fit labels
    plt.show()


def plot_missing_data(df):
    """
    Visualize missing data as a heatmap.
    """
    plt.figure(figsize=(12, 8))
    sns.heatmap(df.isnull(), cbar=False, cmap="viridis")
    plt.title("Missing Data Heatmap")
    plt.show()


def plot_feature_vs_target(df, feature, target, kind="box", rotation=45):
    """
    Plot the relationship between a feature and the target variable.

    Parameters:
        df (pd.DataFrame): The dataset.
        feature (str): The feature to plot.
        target (str): The target variable to compare against.
        kind (str): The type of plot.
        rotation (int): The rotation angle for the x-axis labels.
    """
    plt.figure(figsize=(10, 6))
    if kind == "box":
        sns.boxplot(x=target, y=feature, data=df)
    elif kind == "bar":
        sns.countplot(x=feature, hue=target, data=df)

    plt.title(f"{feature} vs {target}")
    plt.xticks(rotation=rotation, ha="right")  # Rotate x-axis labels
    plt.tight_layout()  # Adjust layout to prevent overlap
    plt.show()


def plot_accidents_by_day_of_week(data, datetime_col="DateTime"):
    """
    Plot the distribution of accidents by day of the week.

    Parameters:
        data (pd.DataFrame): The input DataFrame.
        datetime_col (str): The name of the datetime column.
    """
    if datetime_col not in data.columns:
        raise KeyError(f"{datetime_col} column not found in DataFrame.")

    # Convert to datetime if not already
    if not pd.api.types.is_datetime64_any_dtype(data[datetime_col]):
        print(f"Converting {datetime_col} to pandas datetime format...")
        data[datetime_col] = pd.to_datetime(data[datetime_col],
                                            errors="coerce")

    # Drop rows with null DateTime values
    data = data.dropna(subset=[datetime_col])

    # Extract day of the week
    data["Day_of_Week"] = data[datetime_col].dt.dayofweek

    # Ensure "Day_of_Week" exists in DataFrame
    if "Day_of_Week" not in data.columns:
        raise KeyError("Column 'Day_of_Week' not found after transformation.")

    # Plot
    plt.figure(figsize=(10, 5))
    sns.countplot(x="Day_of_Week", data=data, order=range(7))
    plt.title("Accidents Distribution by Day")
    plt.xticks(range(7), labels=["Mon", "Tue",
                                 "Wed", "Thu",
                                 "Fri", "Sat", "Sun"])
    plt.xlabel("Day of Week")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()


def plot_accidents_by_month(data, datetime_col="DateTime"):
    """
    Plot the distribution of accidents by month.

    Parameters:
        data (pd.DataFrame): The input DataFrame.
        datetime_col (str): The name of the datetime column.
    """
    if datetime_col not in data.columns:
        raise KeyError(f"{datetime_col} column not found in DataFrame.")

    # Convert to datetime if not already
    if not pd.api.types.is_datetime64_any_dtype(data[datetime_col]):
        print(f"Converting {datetime_col} to pandas datetime format...")
        data[datetime_col] = pd.to_datetime(data[datetime_col],
                                            errors="coerce")

    # Drop rows with null DateTime values
    data = data.dropna(subset=[datetime_col])

    # Extract month
    data["Month"] = data[datetime_col].dt.month

    # Ensure "Month" exists in DataFrame
    if "Month" not in data.columns:
        raise KeyError("Column 'Month' not found after transformation.")

    # Plot
    plt.figure(figsize=(12, 6))
    sns.countplot(x="Month", data=data, order=range(1, 13))
    plt.title("Accidents Distribution by Month")
    plt.xticks(
        ticks=range(12),
        labels=[
            "Jan",
            "Feb",
            "Mar",
            "Apr",
            "May",
            "Jun",
            "Jul",
            "Aug",
            "Sep",
            "Oct",
            "Nov",
            "Dec",
        ],
    )
    plt.xlabel("Month")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()


def plot_accidents_by_time(data, datetime_col="DateTime"):
    """
    Plot the distribution of accidents by time of day (hour).

    Parameters:
        data (pd.DataFrame): The input DataFrame.
        datetime_col (str): The name of the datetime column.
    """
    if datetime_col not in data.columns:
        raise KeyError(f"{datetime_col} column not found in DataFrame.")

    # Convert to datetime if not already
    if not pd.api.types.is_datetime64_any_dtype(data[datetime_col]):
        print(f"Converting {datetime_col} to datetime format...")
        data[datetime_col] = pd.to_datetime(data[datetime_col],
                                            errors="coerce")

    # Drop rows with null DateTime values
    data = data.dropna(subset=[datetime_col])

    # Extract hour
    data["Hour"] = data[datetime_col].dt.hour

    # Ensure "Hour" exists in DataFrame
    if "Hour" not in data.columns:
        raise KeyError("Column 'Hour' not found after transformation.")

    # Plot
    plt.figure(figsize=(10, 5))
    sns.countplot(x="Hour", data=data, order=range(0, 24))
    plt.title("Accidents Distribution by Hour")
    plt.xticks(range(0, 24))
    plt.xlabel("Hour of Day")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()
