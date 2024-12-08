import matplotlib.pyplot as plt
import seaborn as sns


def plot_distribution(df, column, figsize=(10, 6), rotation=45):
    """
    Plot the distribution of a categorical variable with improved axis readability.

    Parameters:
        df (pd.DataFrame): The dataframe containing the data.
        column (str): The column to plot.
        figsize (tuple): Size of the figure.
        rotation (int): Angle for rotating x-axis labels.
    """
    plt.figure(figsize=figsize)
    sns.countplot(x=column, data=df, order=df[column].value_counts().index)
    plt.title(f"Distribution of {column}")
    plt.xlabel(column)
    plt.ylabel("Count")
    plt.xticks(rotation=rotation, ha="right")  # Rotate x-axis labels for readability
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
        feature (str): The feature (independent variable) to plot.
        target (str): The target variable (dependent variable) to compare against.
        kind (str): The type of plot ("box" for boxplot, "bar" for barplot). Default is "box".
        rotation (int): The rotation angle for the x-axis labels. Default is 45.
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


def plot_feature_distribution(df, feature):
    """
    Plot the distribution of a numerical feature.
    """
    plt.figure(figsize=(8, 5))
    sns.histplot(df[feature].dropna(), kde=True, bins=30)
    plt.title(f"Distribution of {feature}")
    plt.xlabel(feature)
    plt.ylabel("Frequency")
    plt.show()


def plot_accidents_by_day_of_week(data):
    """
    Plot the distribution of accidents by day of the week.
    """
    plt.figure(figsize=(10, 5))
    data["Day_of_Week"] = data["DateTime"].dt.dayofweek  # 0 = Monday, 6 = Sunday
    sns.countplot(x="Day_of_Week", data=data, order=range(7))
    plt.title("Accidents Distribution by Day of Week")
    plt.xticks(range(7), labels=["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])
    plt.xlabel("Day of Week")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()


def plot_accidents_by_month(data):
    """
    Plot the distribution of accidents by month with proper alignment and axis labeling.
    """
    plt.figure(figsize=(12, 6))
    # Ensure the 'Month' column is extracted
    data["Month"] = data["DateTime"].dt.month

    # Use sns.countplot with fixed order
    sns.countplot(x="Month", data=data, order=range(1, 13))

    # Properly label the x-axis
    plt.title("Accidents Distribution by Month", fontsize=16)
    plt.xticks(
        ticks=range(12),  # Ensure alignment with the month numbers
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
        fontsize=12,
    )
    plt.xlabel("Month", fontsize=14)
    plt.ylabel("Count", fontsize=14)
    plt.tight_layout()
    plt.show()


def plot_accidents_by_time(data):
    """
    Plot the distribution of accidents by time of day (hour).
    """
    plt.figure(figsize=(10, 5))
    data["Hour"] = data["DateTime"].dt.hour
    sns.countplot(x="Hour", data=data, order=range(0, 24))
    plt.title("Accidents Distribution by Hour")
    plt.xticks(range(0, 24))
    plt.xlabel("Hour of Day")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()
