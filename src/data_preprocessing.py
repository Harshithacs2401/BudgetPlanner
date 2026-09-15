"""
BudgetPlanner - Data Preprocessing

Purpose:
    Load the raw personal finance dataset,
    clean the data, handle missing values and duplicates,
    normalize column names, and save the processed dataset.

Input:
    data/raw/indian_personal_finance.csv

Output:
    data/processed/cleaned_finance_data.csv
"""

from pathlib import Path
import pandas as pd
import numpy as np


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"

RAW_FILE = RAW_DATA_DIR / "indian_personal_finance.csv"
PROCESSED_FILE = (
    PROCESSED_DATA_DIR / "cleaned_finance_data.csv"
)


# ============================================================
# LOAD DATA
# ============================================================

def load_dataset(file_path):
    """
    Load the dataset from CSV.

    Parameters
    ----------
    file_path : Path
        Location of the raw CSV file.

    Returns
    -------
    pandas.DataFrame
    """

    if not file_path.exists():
        raise FileNotFoundError(
            f"\nDataset not found:\n{file_path}\n\n"
            "Please place the downloaded dataset inside:\n"
            "data/raw/\n"
        )

    print("=" * 70)
    print("LOADING DATASET")
    print("=" * 70)

    df = pd.read_csv(file_path)

    print(f"Dataset loaded successfully.")
    print(f"Rows    : {df.shape[0]}")
    print(f"Columns : {df.shape[1]}")

    return df


# ============================================================
# DISPLAY DATASET INFORMATION
# ============================================================

def display_dataset_info(df):
    """
    Display basic information about the dataset.
    """

    print("\n" + "=" * 70)
    print("DATASET INFORMATION")
    print("=" * 70)

    print("\nShape:")
    print(df.shape)

    print("\nColumns:")
    for column in df.columns:
        print(f"  - {column}")

    print("\nData Types:")
    print(df.dtypes)

    print("\nMissing Values:")
    missing = df.isnull().sum()

    missing = missing[missing > 0]

    if len(missing) == 0:
        print("  No missing values found.")
    else:
        print(missing)

    print("\nDuplicate Rows:")
    print(df.duplicated().sum())


# ============================================================
# CLEAN COLUMN NAMES
# ============================================================

def clean_column_names(df):
    """
    Standardize column names.

    Example:
        Desired Savings (%) -> desired_savings_percentage
    """

    df = df.copy()

    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_", regex=False)
        .str.replace("%", "percentage", regex=False)
        .str.replace("(", "", regex=False)
        .str.replace(")", "", regex=False)
        .str.replace("/", "_", regex=False)
        .str.replace("-", "_", regex=False)
    )

    print("\nStandardized column names:")
    print(list(df.columns))

    return df


# ============================================================
# REMOVE DUPLICATES
# ============================================================

def remove_duplicates(df):
    """
    Remove duplicate rows.
    """

    before = len(df)

    df = df.drop_duplicates().reset_index(drop=True)

    after = len(df)

    removed = before - after

    print(
        f"\nDuplicate rows removed: {removed}"
    )

    return df


# ============================================================
# HANDLE MISSING VALUES
# ============================================================

def handle_missing_values(df):
    """
    Handle missing values.

    Numerical columns:
        Fill using median.

    Categorical columns:
        Fill using mode.
    """

    df = df.copy()

    numerical_columns = df.select_dtypes(
        include=np.number
    ).columns

    categorical_columns = df.select_dtypes(
        exclude=np.number
    ).columns

    # Numerical values
    for column in numerical_columns:

        if df[column].isnull().any():

            median_value = df[column].median()

            df[column] = df[column].fillna(
                median_value
            )

            print(
                f"Filled missing numerical values "
                f"in '{column}' using median."
            )

    # Categorical values
    for column in categorical_columns:

        if df[column].isnull().any():

            mode_values = df[column].mode()

            if not mode_values.empty:

                mode_value = mode_values.iloc[0]

                df[column] = df[column].fillna(
                    mode_value
                )

                print(
                    f"Filled missing categorical values "
                    f"in '{column}' using mode."
                )

    return df


# ============================================================
# CLEAN NUMERICAL VALUES
# ============================================================

def clean_numeric_columns(df):
    """
    Convert numeric-looking columns into numeric data types.

    Non-convertible values are converted to NaN and then
    handled by the missing-value function.
    """

    df = df.copy()

    for column in df.columns:

        if df[column].dtype == "object":

            converted = pd.to_numeric(
                df[column],
                errors="coerce"
            )

            valid_ratio = converted.notna().mean()

            # Convert only when most values are numeric
            if valid_ratio >= 0.8:

                df[column] = converted

    return df


# ============================================================
# HANDLE IMPOSSIBLE NUMERIC VALUES
# ============================================================

def handle_invalid_values(df):
    """
    Handle negative values in financial columns.

    Financial amounts such as income and expenses should
    normally not be negative.

    Negative values are converted to NaN and handled later.
    """

    df = df.copy()

    financial_keywords = [
        "income",
        "salary",
        "rent",
        "loan",
        "insurance",
        "grocery",
        "groceries",
        "transport",
        "utility",
        "utilities",
        "healthcare",
        "education",
        "entertainment",
        "eating",
        "miscellaneous",
        "savings",
        "expense",
        "spending",
        "investment",
        "emi",
    ]

    for column in df.columns:

        column_lower = column.lower()

        if any(
            keyword in column_lower
            for keyword in financial_keywords
        ):

            if pd.api.types.is_numeric_dtype(
                df[column]
            ):

                negative_count = (
                    df[column] < 0
                ).sum()

                if negative_count > 0:

                    print(
                        f"Found {negative_count} "
                        f"negative values in '{column}'."
                    )

                    df.loc[
                        df[column] < 0,
                        column
                    ] = np.nan

    return df


# ============================================================
# REMOVE EXTREME INVALID RECORDS
# ============================================================

def remove_invalid_rows(df):
    """
    Remove rows where all values are missing.
    """

    before = len(df)

    df = df.dropna(
        how="all"
    ).reset_index(drop=True)

    after = len(df)

    print(
        f"\nCompletely empty rows removed: "
        f"{before - after}"
    )

    return df


# ============================================================
# FINAL DATA VALIDATION
# ============================================================

def validate_dataset(df):
    """
    Perform final validation checks.
    """

    print("\n" + "=" * 70)
    print("FINAL DATA VALIDATION")
    print("=" * 70)

    print(
        f"Rows    : {df.shape[0]}"
    )

    print(
        f"Columns : {df.shape[1]}"
    )

    missing_count = df.isnull().sum().sum()

    duplicate_count = df.duplicated().sum()

    print(
        f"Missing values : {missing_count}"
    )

    print(
        f"Duplicate rows : {duplicate_count}"
    )

    if missing_count == 0:
        print("✓ No missing values.")

    if duplicate_count == 0:
        print("✓ No duplicate rows.")

    print("\nFinal columns:")

    for column in df.columns:
        print(f"  - {column}")


# ============================================================
# SAVE PROCESSED DATA
# ============================================================

def save_dataset(df, file_path):
    """
    Save cleaned dataset to CSV.
    """

    file_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(
        file_path,
        index=False
    )

    print("\n" + "=" * 70)
    print("DATASET SAVED")
    print("=" * 70)

    print(
        f"\nProcessed dataset:\n{file_path}"
    )


# ============================================================
# MAIN PREPROCESSING PIPELINE
# ============================================================

def preprocess_dataset():
    """
    Execute the complete preprocessing pipeline.
    """

    print("\n")
    print("=" * 70)
    print("BUDGETPLANNER DATA PREPROCESSING")
    print("=" * 70)

    # --------------------------------------------------------
    # 1. Load
    # --------------------------------------------------------

    df = load_dataset(
        RAW_FILE
    )

    # --------------------------------------------------------
    # 2. Initial information
    # --------------------------------------------------------

    display_dataset_info(
        df
    )

    # --------------------------------------------------------
    # 3. Standardize column names
    # --------------------------------------------------------

    df = clean_column_names(
        df
    )

    # --------------------------------------------------------
    # 4. Remove duplicates
    # --------------------------------------------------------

    df = remove_duplicates(
        df
    )

    # --------------------------------------------------------
    # 5. Convert numeric columns
    # --------------------------------------------------------

    df = clean_numeric_columns(
        df
    )

    # --------------------------------------------------------
    # 6. Handle invalid values
    # --------------------------------------------------------

    df = handle_invalid_values(
        df
    )

    # --------------------------------------------------------
    # 7. Remove empty rows
    # --------------------------------------------------------

    df = remove_invalid_rows(
        df
    )

    # --------------------------------------------------------
    # 8. Handle missing values
    # --------------------------------------------------------

    df = handle_missing_values(
        df
    )

    # --------------------------------------------------------
    # 9. Final validation
    # --------------------------------------------------------

    validate_dataset(
        df
    )

    # --------------------------------------------------------
    # 10. Save
    # --------------------------------------------------------

    save_dataset(
        df,
        PROCESSED_FILE
    )

    print("\n")
    print("=" * 70)
    print("PREPROCESSING COMPLETED SUCCESSFULLY")
    print("=" * 70)

    return df


# ============================================================
# PROGRAM ENTRY POINT
# ============================================================

if __name__ == "__main__":

    preprocess_dataset()