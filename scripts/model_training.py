from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier
from glum import GeneralizedLinearRegressor, TweedieDistribution
from scripts.feature_engineering import LogTransform
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline

def load_and_split_data(
    data_path=None,
    data_dir="data",
    file_name="processed_data.parquet",
    id_column="Accident_Index",
    test_size=0.1,
    random_state=42
):
    """
    Load processed data and split into train and test sets using ID-based splitting.

    Parameters:
        data_path (str or Path, optional): Full path to the data file. Overrides `data_dir` and `file_name` if provided.
        data_dir (str): Directory where the data file is stored (used if `data_path` is not provided).
        file_name (str): Name of the processed data file (used if `data_path` is not provided).
        id_column (str): Column to base splitting on.
    Returns:
        pd.DataFrame: Train data.
        pd.DataFrame: Test data.
    """
    # Resolve the path dynamically if data_path is not provided
    if data_path is None:
        data_path = Path(data_dir) / file_name

    # Ensure data_path is a Path object
    data_path = Path(data_path)

    # Check if the file exists
    if not data_path.is_file():
        raise FileNotFoundError(f"The data file {data_path} does not exist.")

    # Load data
    data = pd.read_parquet(data_path)

    # Perform ID-based splitting
    unique_ids = data[id_column].unique()
    train_ids, test_ids = train_test_split(unique_ids, test_size=test_size, random_state=random_state)

    train_data = data[data[id_column].isin(train_ids)]
    test_data = data[data[id_column].isin(test_ids)]

    return train_data, test_data

def build_glm_pipeline():
    """
    Build a pipeline for GLM with feature preprocessing and logistic regression.
    """
    # Define feature columns
    glm_numerical_features = ["Speed_limit",
                              "Number_of_Casualties",
                              "Number_of_Vehicles",
                              ]
    glm_categorical_features = [
        "Did_Police_Officer_Attend_Scene_of_Accident",
        "Junction_Control",
        "Junction_Detail",
        "Light_Conditions",
        "Pedestrian_Crossing-Human_Control",
        "Pedestrian_Crossing-Physical_Facilities",
        "Road_Type",
        "Urban_or_Rural_Area",
        "Weather_Conditions",
        "Road_Surface_Conditions",
        "Month",
        "Day_of_Week",
        "Hour_of_Day"
    ]

    # Log transformation for numerical features
    log_transformer = LogTransform(columns=glm_numerical_features)

    numerical_transformer = Pipeline([
        ("log_transform", log_transformer),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = OneHotEncoder(handle_unknown="ignore")

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_transformer, glm_numerical_features),
            ("cat", categorical_transformer, glm_categorical_features),
        ]
    )

    glm_pipeline = Pipeline([
    ("preprocessor", preprocessor),  # Preprocessing step (scaling, encoding, etc.)
    ("classifier", GeneralizedLinearRegressor(
        alpha=0.5,  # Regularization strength (to be tuned during hyperparameter optimization)
        family=TweedieDistribution(power=1.5),  # Adjust power as needed
        fit_intercept=True,
        max_iter=1000  # Increase if convergence issues occur
    ))
    ])
    return glm_pipeline

def build_lgbm_pipeline():
    """
    Build a pipeline for LGBM with feature preprocessing and LightGBM model.
    """
    # Define feature columns
    lgbm_numerical_features = ["Speed_limit",
                              "Number_of_Casualties",
                              "Number_of_Vehicles",
                              ]
    lgbm_categorical_features = [
        "Did_Police_Officer_Attend_Scene_of_Accident",
        "Junction_Control",
        "Junction_Detail",
        "Light_Conditions",
        "Pedestrian_Crossing-Human_Control",
        "Pedestrian_Crossing-Physical_Facilities",
        "Road_Type",
        "Urban_or_Rural_Area",
        "Weather_Conditions",
        "Road_Surface_Conditions",
        "Month",
        "Day_of_Week",
        "Hour_of_Day"
    ]

    log_transformer = LogTransform(columns=lgbm_numerical_features)

    numerical_transformer = Pipeline([
        ("log_transform", log_transformer),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = OneHotEncoder(handle_unknown="ignore")

    lgbm_preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_transformer, lgbm_numerical_features),
            ("cat", categorical_transformer, lgbm_categorical_features),
        ]
    )

    # LGBM pipeline
    lgbm_pipeline = Pipeline([
        ("preprocessor", lgbm_preprocessor),
        ("classifier", LGBMClassifier(n_estimators=100,
                                      objective="multiclass",
                                      random_state=42))
    ])

    return lgbm_pipeline

def tune_glm_pipeline(glm_pipeline, X_train, y_train):
    """
    Tune the GLM pipeline using GridSearchCV with manual handling of variance_power.

    Parameters:
        glm_pipeline (Pipeline): The GLM pipeline.
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training target.

    Returns:
        Pipeline: The best estimator after tuning.
        dict: The best parameters.
    """
    # Define alpha values to tune
    alpha_values = [0.1, 0.5, 1.0]

    # Define variance power values to tune for TweedieDistribution
    variance_power_values = [1.0, 1.5, 2.0]

    # Store the results
    best_score = -float("inf")
    best_pipeline = None
    best_params = {}

    # Iterate over variance_power
    for variance_power in variance_power_values:
        # Create a new pipeline with the updated TweedieDistribution
        glm_pipeline.set_params(classifier=GeneralizedLinearRegressor(
            alpha=0.5,  # Placeholder, will be tuned below
            family=TweedieDistribution(power=variance_power),
            fit_intercept=True,
            max_iter=1000
        ))

        # Define the parameter grid for alpha
        param_grid = {
            "classifier__alpha": alpha_values,
        }

        # Perform GridSearchCV
        grid_search = GridSearchCV(glm_pipeline, param_grid, cv=5, scoring="accuracy", n_jobs=-1)
        grid_search.fit(X_train, y_train)

        # Update the best pipeline if the score improves
        if grid_search.best_score_ > best_score:
            best_score = grid_search.best_score_
            best_pipeline = grid_search.best_estimator_
            best_params = grid_search.best_params_
            best_params["variance_power"] = variance_power  # Add variance_power to the best params

    return best_pipeline, best_params


def tune_lgbm_pipeline(lgbm_pipeline, X_train, y_train):
    param_grid = {
        "classifier__learning_rate": [0.01, 0.1, 0.2],
        "classifier__n_estimators": [50, 100, 200],
        "classifier__n_leaves": [31, 50, 70],
        "classifier__min_child_weight": [1, 5, 10]
    }
    grid_search = GridSearchCV(lgbm_pipeline, param_grid, cv=5, scoring="accuracy", n_jobs=-1)
    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_

def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    print(f"Model Accuracy: {accuracy:.2f}")
