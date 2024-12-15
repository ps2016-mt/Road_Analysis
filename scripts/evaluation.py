from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import dalex as dx
import numpy as np
import pandas as pd


def evaluate_model(model, test_data, target_column="Accident_Severity"):
    """
    Evaluate a model on the test data and print accuracy.

    Parameters:
        model: Trained model pipeline.

    Returns:
        pd.DataFrame: DataFrame with actual and predicted values.
    """
    # Split features and target
    X_test = test_data.drop(columns=[target_column, "Accident_Index"])
    y_test = test_data[target_column]

    # Predict and calculate accuracy
    predictions = model.predict(X_test)

    # Confusion matrix
    cm = confusion_matrix(y_test, predictions, labels=np.unique(y_test))
    ConfusionMatrixDisplay(confusion_matrix=cm,
                           display_labels=np.unique(y_test)).plot()
    plt.title("Confusion Matrix")
    plt.show()

    return pd.DataFrame({"Actual": y_test, "Predicted": predictions})


def plot_predicted_vs_actual_bar(df, actual_col="Actual",
                                 predicted_col="Predicted"):
    """
    Parameters:
        df (pd.DataFrame): DataFrame containing actual and predicted columns.
    """
    # Count occurrences of actual and predicted values
    actual_counts = df[actual_col].value_counts().sort_index()
    predicted_counts = df[predicted_col].value_counts().sort_index()

    # Define bar width and x-axis ticks
    width = 0.35
    x = np.arange(len(actual_counts))

    # Plot the grouped bar chart
    plt.figure(figsize=(8, 6))
    plt.bar(x - width/2, actual_counts, width, label="Actual")
    plt.bar(x + width/2, predicted_counts, width, label="Predicted")

    plt.xticks(x, actual_counts.index, fontsize=10)
    plt.xlabel("Accident Severity", fontsize=12)
    plt.ylabel("Count", fontsize=12)
    plt.title("Actual vs Predicted Distribution", fontsize=14)
    plt.legend()
    plt.tight_layout()
    plt.show()


def feature_analysis(
    model_pipeline,
    X_train,
    y_train,
    top_n=5
):
    """
    Perform feature analysis and extract the top N important features.

    Parameters:
        model_pipeline (Pipeline): The trained model pipeline.
        top_n (int): Number of top important features to extract.

    Returns:
        pd.DataFrame: Top N important features and their importance values.
    """
    # Initialise Dalex Explainer
    explainer = dx.Explainer(
        model_pipeline, X_train, y_train, label="Model Feature Analysis"
    )

    # Calculate feature importance
    importance = explainer.model_parts()

    # Filter out the '_full_model_' row
    filtered_importance = importance.result[importance
                                            .result["variable"] != "_full_model_"]

    # Extract top N features
    top_features = (
        filtered_importance.sort_values(by="dropout_loss", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )

    print("\nTop Features by Importance:")
    print(top_features[["variable", "dropout_loss"]])

    # Plot feature importance
    top_features.plot.bar(x="variable", y="dropout_loss", legend=False)
    plt.title("Top Feature Importance (Dropout Loss)")
    plt.xlabel("Features")
    plt.ylabel("Importance (Dropout Loss)")
    plt.tight_layout()
    plt.show()
