"""
BudgetPlanner - Model Training

Purpose:
    Train and evaluate machine-learning regression models
    using the processed personal-finance dataset.

Input:
    data/processed/cleaned_finance_data.csv

Output:
    models/
        budget_model.joblib

    reports/
        model_results.csv

Usage:
    From the project root:

        python src/model_training.py

Optional:
    Set the target column manually.

    Windows CMD:
        set ML_TARGET_COLUMN=desired_savings

    PowerShell:
        $env:ML_TARGET_COLUMN="desired_savings"
"""

from pathlib import Path
import os
import warnings

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)
from sklearn.model_selection import train_test_split


warnings.filterwarnings("ignore")


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = (
    PROJECT_ROOT /
    "data" /
    "processed"
)

DATA_FILE = (
    DATA_DIR /
    "cleaned_finance_data.csv"
)

MODEL_DIR = (
    PROJECT_ROOT /
    "models"
)

REPORT_DIR = (
    PROJECT_ROOT /
    "reports"
)

MODEL_FILE = (
    MODEL_DIR /
    "budget_model.joblib"
)

RESULT_FILE = (
    REPORT_DIR /
    "model_results.csv"
)


MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)

REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# CONFIGURATION
# ============================================================

TEST_SIZE = 0.20
RANDOM_STATE = 42


# ============================================================
# LOAD DATA
# ============================================================

def load_dataset(file_path):
    """
    Load the processed dataset.
    """

    if not file_path.exists():

        raise FileNotFoundError(
            f"\nProcessed dataset not found:\n"
            f"{file_path}\n\n"
            "Run the preprocessing pipeline first:\n"
            "python src/data_preprocessing.py"
        )

    print("=" * 70)
    print("LOADING DATASET")
    print("=" * 70)

    df = pd.read_csv(
        file_path
    )

    print(
        f"Rows    : {df.shape[0]}"
    )

    print(
        f"Columns : {df.shape[1]}"
    )

    return df


# ============================================================
# FIND TARGET COLUMN
# ============================================================

def find_target_column(df):
    """
    Determine the ML target column.

    Priority:
        1. ML_TARGET_COLUMN environment variable
        2. Common savings-related columns
        3. Common financial columns

    The target must be numerical.
    """

    manual_target = os.getenv(
        "ML_TARGET_COLUMN"
    )

    if manual_target:

        if manual_target not in df.columns:

            raise ValueError(
                f"\nTarget column '{manual_target}' "
                f"was not found.\n\n"
                f"Available columns:\n"
                f"{list(df.columns)}"
            )

        if not pd.api.types.is_numeric_dtype(
            df[manual_target]
        ):

            raise ValueError(
                f"Target column '{manual_target}' "
                "must be numerical."
            )

        print(
            f"\nTarget selected manually: "
            f"{manual_target}"
        )

        return manual_target

    # --------------------------------------------------------
    # Preferred savings targets
    # --------------------------------------------------------

    preferred_targets = [
        "desired_savings",
        "desired_savings_amount",
        "potential_savings",
        "savings",
        "monthly_savings",
        "actual_savings",
    ]

    for target in preferred_targets:

        if target in df.columns:

            if pd.api.types.is_numeric_dtype(
                df[target]
            ):

                print(
                    f"\nAutomatically selected target: "
                    f"{target}"
                )

                return target

    # --------------------------------------------------------
    # Search by keyword
    # --------------------------------------------------------

    for column in df.columns:

        name = column.lower()

        if (
            "savings" in name or
            "saving" in name
        ):

            if pd.api.types.is_numeric_dtype(
                df[column]
            ):

                print(
                    f"\nAutomatically selected target: "
                    f"{column}"
                )

                return column

    raise ValueError(
        "\nUnable to automatically determine "
        "a suitable target column.\n\n"
        "Available numerical columns:\n"
        + str(
            list(
                df.select_dtypes(
                    include=np.number
                ).columns
            )
        )
        + "\n\n"
        "Set the target manually using:\n"
        "ML_TARGET_COLUMN"
    )


# ============================================================
# REMOVE TARGET FROM FEATURES
# ============================================================

def prepare_features(df, target_column):
    """
    Separate input features and target.
    """

    df = df.copy()

    # Remove completely empty columns
    empty_columns = [
        column
        for column in df.columns
        if df[column].isnull().all()
    ]

    if empty_columns:

        print(
            "\nRemoving completely empty columns:"
        )

        print(
            empty_columns
        )

        df = df.drop(
            columns=empty_columns
        )

    X = df.drop(
        columns=[target_column]
    )

    y = df[target_column]

    # Remove rows where target is missing
    valid_target = y.notna()

    X = X.loc[
        valid_target
    ].reset_index(drop=True)

    y = y.loc[
        valid_target
    ].reset_index(drop=True)

    print("\n" + "=" * 70)
    print("FEATURE PREPARATION")
    print("=" * 70)

    print(
        f"Features : {X.shape[1]}"
    )

    print(
        f"Samples  : {X.shape[0]}"
    )

    print(
        f"Target   : {target_column}"
    )

    return X, y


# ============================================================
# REMOVE ID-LIKE FEATURES
# ============================================================

def remove_id_like_columns(X):
    """
    Remove columns that are likely identifiers.

    Examples:
        ID
        User_ID
        Customer_ID
    """

    X = X.copy()

    id_columns = []

    for column in X.columns:

        name = column.lower()

        if (
            name == "id" or
            name.endswith("_id") or
            name.endswith("id")
        ):

            id_columns.append(
                column
            )

    if id_columns:

        print(
            "\nRemoving ID-like columns:"
        )

        print(
            id_columns
        )

        X = X.drop(
            columns=id_columns
        )

    return X


# ============================================================
# BUILD PREPROCESSOR
# ============================================================

def build_preprocessor(X):
    """
    Build preprocessing pipeline for numerical
    and categorical features.
    """

    numerical_columns = (
        X.select_dtypes(
            include=np.number
        ).columns.tolist()
    )

    categorical_columns = (
        X.select_dtypes(
            include=[
                "object",
                "category",
                "bool"
            ]
        ).columns.tolist()
    )

    print("\n" + "=" * 70)
    print("FEATURE TYPES")
    print("=" * 70)

    print(
        f"Numerical features   : "
        f"{len(numerical_columns)}"
    )

    print(
        f"Categorical features : "
        f"{len(categorical_columns)}"
    )

    if numerical_columns:

        print("\nNumerical:")

        for column in numerical_columns:
            print(
                f"  - {column}"
            )

    if categorical_columns:

        print("\nCategorical:")

        for column in categorical_columns:
            print(
                f"  - {column}"
            )

    transformers = []

    # --------------------------------------------------------
    # Numerical pipeline
    # --------------------------------------------------------

    if numerical_columns:

        numerical_pipeline = Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(
                        strategy="median"
                    )
                ),
                (
                    "scaler",
                    StandardScaler()
                )
            ]
        )

        transformers.append(
            (
                "numerical",
                numerical_pipeline,
                numerical_columns
            )
        )

    # --------------------------------------------------------
    # Categorical pipeline
    # --------------------------------------------------------

    if categorical_columns:

        categorical_pipeline = Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(
                        strategy="most_frequent"
                    )
                ),
                (
                    "encoder",
                    OneHotEncoder(
                        handle_unknown="ignore"
                    )
                )
            ]
        )

        transformers.append(
            (
                "categorical",
                categorical_pipeline,
                categorical_columns
            )
        )

    if not transformers:

        raise ValueError(
            "No usable features were found."
        )

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop"
    )

    return preprocessor


# ============================================================
# DEFINE MODELS
# ============================================================

def get_models():
    """
    Return candidate regression models.
    """

    models = {

        "LinearRegression":
            LinearRegression(),

        "RandomForest":
            RandomForestRegressor(
                n_estimators=200,
                max_depth=None,
                min_samples_split=2,
                min_samples_leaf=1,
                random_state=RANDOM_STATE,
                n_jobs=-1
            )
    }

    return models


# ============================================================
# TRAIN AND EVALUATE MODEL
# ============================================================

def train_and_evaluate(
    model_name,
    model,
    preprocessor,
    X_train,
    X_test,
    y_train,
    y_test
):
    """
    Train one model and calculate regression metrics.
    """

    pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor
            ),
            (
                "model",
                model
            )
        ]
    )

    print("\n" + "=" * 70)
    print(
        f"TRAINING: {model_name}"
    )
    print("=" * 70)

    pipeline.fit(
        X_train,
        y_train
    )

    predictions = pipeline.predict(
        X_test
    )

    mae = mean_absolute_error(
        y_test,
        predictions
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_test,
            predictions
        )
    )

    r2 = r2_score(
        y_test,
        predictions
    )

    print(
        f"MAE  : {mae:.4f}"
    )

    print(
        f"RMSE : {rmse:.4f}"
    )

    print(
        f"R²   : {r2:.4f}"
    )

    return pipeline, {
        "model": model_name,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    }


# ============================================================
# SAVE MODEL RESULTS
# ============================================================

def save_results(results):
    """
    Save model comparison results.
    """

    results_df = pd.DataFrame(
        results
    )

    results_df = results_df.sort_values(
        by="RMSE",
        ascending=True
    )

    results_df.to_csv(
        RESULT_FILE,
        index=False
    )

    print("\n" + "=" * 70)
    print("MODEL COMPARISON")
    print("=" * 70)

    print(
        results_df.to_string(
            index=False
        )
    )

    print(
        f"\nResults saved to:\n"
        f"{RESULT_FILE}"
    )

    return results_df


# ============================================================
# SAVE BEST MODEL
# ============================================================

def save_model(
    pipeline,
    target_column,
    results
):
    """
    Save the complete preprocessing + model pipeline.
    """

    model_package = {
        "pipeline": pipeline,
        "target_column": target_column,
        "model_results": results,
        "random_state": RANDOM_STATE
    }

    joblib.dump(
        model_package,
        MODEL_FILE
    )

    print("\n" + "=" * 70)
    print("BEST MODEL SAVED")
    print("=" * 70)

    print(
        f"\nModel file:\n"
        f"{MODEL_FILE}"
    )


# ============================================================
# MAIN TRAINING PIPELINE
# ============================================================

def train_models():

    print("\n")
    print("=" * 70)
    print("BUDGETPLANNER - MODEL TRAINING")
    print("=" * 70)

    # --------------------------------------------------------
    # 1. Load dataset
    # --------------------------------------------------------

    df = load_dataset(
        DATA_FILE
    )

    # --------------------------------------------------------
    # 2. Find target
    # --------------------------------------------------------

    target_column = (
        find_target_column(
            df
        )
    )

    # --------------------------------------------------------
    # 3. Prepare X and y
    # --------------------------------------------------------

    X, y = prepare_features(
        df,
        target_column
    )

    # --------------------------------------------------------
    # 4. Remove ID columns
    # --------------------------------------------------------

    X = remove_id_like_columns(
        X
    )

    # --------------------------------------------------------
    # 5. Train/test split
    # --------------------------------------------------------

    X_train, X_test, y_train, y_test = (
        train_test_split(
            X,
            y,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE
        )
    )

    print("\n" + "=" * 70)
    print("TRAIN / TEST SPLIT")
    print("=" * 70)

    print(
        f"Training samples : "
        f"{len(X_train)}"
    )

    print(
        f"Testing samples  : "
        f"{len(X_test)}"
    )

    # --------------------------------------------------------
    # 6. Build preprocessing
    # --------------------------------------------------------

    preprocessor = (
        build_preprocessor(
            X_train
        )
    )

    # --------------------------------------------------------
    # 7. Get models
    # --------------------------------------------------------

    models = get_models()

    results = []

    trained_models = {}

    # --------------------------------------------------------
    # 8. Train candidate models
    # --------------------------------------------------------

    for model_name, model in (
        models.items()
    ):

        pipeline, metrics = (
            train_and_evaluate(
                model_name,
                model,
                preprocessor,
                X_train,
                X_test,
                y_train,
                y_test
            )
        )

        results.append(
            metrics
        )

        trained_models[
            model_name
        ] = pipeline

    # --------------------------------------------------------
    # 9. Save results
    # --------------------------------------------------------

    results_df = save_results(
        results
    )

    # --------------------------------------------------------
    # 10. Select best model
    # --------------------------------------------------------

    best_model_name = (
        results_df.iloc[0]["model"]
    )

    best_pipeline = (
        trained_models[
            best_model_name
        ]
    )

    print("\n" + "=" * 70)
    print("BEST MODEL")
    print("=" * 70)

    print(
        f"Model: {best_model_name}"
    )

    # --------------------------------------------------------
    # 11. Save best model
    # --------------------------------------------------------

    save_model(
        best_pipeline,
        target_column,
        results_df.to_dict(
            orient="records"
        )
    )

    print("\n")
    print("=" * 70)
    print("MODEL TRAINING COMPLETED")
    print("=" * 70)

    return best_pipeline


# ============================================================
# PROGRAM ENTRY POINT
# ============================================================

if __name__ == "__main__":
    train_models()