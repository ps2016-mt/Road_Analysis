from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score


def load_and_split_data(
    data_path=None,
    data_dir="data",
    file_name="processed_data.parquet",
    id_column="Accident_Index",
    test_size=0.2,
    random_state=42,
):
    """
    Split into train and test sets using ID-based splitting.

    Parameters:
        data_path (str or Path, optional): Full path to the data file.
        data_dir (str): Directory where the data file is stored.
        file_name (str): Used if `data_path` is not provided.
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

    # ID-based splitting
    unique_ids = data[id_column].unique()
    train_ids, test_ids = train_test_split(
        unique_ids, test_size=test_size, random_state=random_state
    )

    train_data = data[data[id_column].isin(train_ids)]
    test_data = data[data[id_column].isin(test_ids)]

    return train_data, test_data


def pipe_preprocessing(numerical_features, categorical_features):
    """
    Create a ColumnTransformer for GLM & LGBM preprocessing.

    Returns:
        ColumnTransformer: A preprocessor for all features.
    """
    # Preprocessing for numerical features
    num_preprocessor = StandardScaler()

    # Preprocessing for categorical features
    cat_preprocessor = OneHotEncoder(handle_unknown="ignore")

    # Combine preprocessors in a ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_preprocessor, numerical_features),
            ("cat", cat_preprocessor, categorical_features),
        ]
    )

    return preprocessor


def build_glm_pipeline(
    train_data,
    test_data,
    target_column,
    preprocessor
):
    """
    Build, train, and evaluate a GLM pipeline using Logistic Regression.

    Parameters:
        preprocessor (ColumnTransformer): Preprocessor for the pipeline.

    Returns:
        Pipeline: The trained GLM pipeline.
        float: Accuracy score on the test dataset.
    """
    # Define Logistic Regression model
    model = LogisticRegression(
        multi_class="multinomial",
        max_iter=1000,
        random_state=42
    )

    # Create pipeline
    glm_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", model)
    ])

    # Split features and target
    X_train = train_data.drop(columns=[target_column, "Accident_Index"])
    y_train = train_data[target_column]
    X_test = test_data.drop(columns=[target_column, "Accident_Index"])
    y_test = test_data[target_column]

    # Train the pipeline
    glm_pipeline.fit(X_train, y_train)

    # Evaluate the pipeline
    y_pred = glm_pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"GLM Model Accuracy: {accuracy:.4f}")

    return glm_pipeline, accuracy


def build_lgbm_pipeline(
    train_data,
    test_data,
    target_column,
    preprocessor
):
    """
    Build, train, and evaluate an LGBM pipeline.

    Parameters:
        target_column (str): Name of the target variable.
        preprocessor: Preprocessor for the pipeline.

    Returns:
        Pipeline: The trained LGBM pipeline.
        float: Accuracy score on the test dataset.
    """
    # Define LGBM classifier
    lgbm_classifier = LGBMClassifier(
        objective="multiclass",
        random_state=42,
    )

    # Combine preprocessing and classifier in a pipeline
    lgbm_pipeline = Pipeline(
        steps=[("preprocessor", preprocessor), ("classifier", lgbm_classifier)]
    )

    # Split features and target
    X_train = train_data.drop(columns=[target_column, "Accident_Index"])
    y_train = train_data[target_column]
    X_test = test_data.drop(columns=[target_column, "Accident_Index"])
    y_test = test_data[target_column]

    # Train the pipeline
    lgbm_pipeline.fit(X_train, y_train)

    # Evaluate the pipeline
    y_pred = lgbm_pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"LGBM Model Accuracy: {accuracy:.4f}")

    return lgbm_pipeline, accuracy


def tune_glm_pipeline(glm_pipeline,
                      train_data,
                      target_column="Accident_Severity"):
    """
    Tune the GLM pipeline using GridSearchCV.

    Parameters:
        glm_pipeline (Pipeline): The GLM pipeline to tune.

    Returns:
        tuple: (Best pipeline,
                Best parameters,
                Best cross-validation score)
    """
    # Split train and test data into features (X) and target (y)
    X_train = train_data.drop(columns=[target_column,
                                       "Accident_Index"])
    y_train = train_data[target_column]

    # Define hyperparameter grid for Logistic Regression - used ChatGPT here
    param_grid = {
        "classifier__C": [0.01, 0.1, 1],  # Regularisation strength
        "classifier__penalty": [
            "l2"
        ],  # Regularization type (L2 recommended for multinomial)
        "classifier__solver": [
            "lbfgs",
            "saga",
        ],  # Solvers that support multinomial logistic regression
        "classifier__multi_class": [
            "multinomial"
        ],  # Specify multinomial classification
    }

    # Define cross-validation strategy
    cv_strategy = 5

    # Create GridSearchCV instance
    grid_search = GridSearchCV(
        glm_pipeline,
        param_grid=param_grid,
        cv=cv_strategy,
        scoring="accuracy",
        n_jobs=-1,
        verbose=1,
    )

    # Fit GridSearchCV on the training data
    grid_search.fit(X_train, y_train)

    # Extract the best pipeline and parameters
    best_glm_pipeline = grid_search.best_estimator_
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_

    # Print results
    print("Best Logistic Regression Pipeline:", best_glm_pipeline)
    print("Best Hyperparameters:", best_params)
    print(f"Best Cross-Validation Accuracy: {best_score:.4f}")

    return best_glm_pipeline, best_params, best_score


def tune_lgbm_pipeline(lgbm_pipeline,
                       train_data,
                       target_column="Accident_Severity"):
    """
    Tune the LGBM pipeline using GridSearchCV.

    Parameters:
        lgbm_pipeline (Pipeline): The LightGBM pipeline to tune.

    Returns:
        Pipeline: The best pipeline after hyperparameter tuning.
        dict: The best parameters from the tuning.
    """
    # Define feature matrix and target variable
    X_train = train_data.drop(columns=[target_column,
                                       "Accident_Index"])
    y_train = train_data[target_column]

    # Define the parameter grid for tuning
    param_grid = {
        "classifier__learning_rate": [0.01,
                                      0.1,
                                      0.2],
        "classifier__n_estimators": [500],
        "classifier__n_leaves": [31,
                                 50,
                                 70],
        "classifier__min_child_weight": [1,
                                         5,
                                         10],
    }

    # Initialise GridSearchCV for hyperparameter tuning
    grid_search = GridSearchCV(
        estimator=lgbm_pipeline,
        param_grid=param_grid,
        cv=5,  # K-fold cross-validation
        scoring="accuracy",
        verbose=2,
        n_jobs=-1,
    )

    # Fit the grid search
    grid_search.fit(X_train, y_train)

    # Get the best pipeline and parameters
    best_lgbm_pipeline = grid_search.best_estimator_
    best_params = grid_search.best_params_

    # Output the results
    print(f"Best Parameters: {best_params}")
    print(f"Best Cross-Validation Accuracy: {grid_search.best_score_:.4f}")

    return best_lgbm_pipeline, best_params
