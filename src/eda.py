"""
BudgetPlanner - Exploratory Data Analysis

Purpose:
    Perform exploratory data analysis on the cleaned
    personal finance dataset.

Input:
    data/processed/cleaned_finance_data.csv

Output:
    reports/
        dataset_summary.txt
        missing_values.csv
        numerical_summary.csv
        categorical_summary.csv

    reports/figures/
        correlation_heatmap.png
        numerical_distributions.png
        categorical_distributions.png

Usage:
    From the project root:

        python src/eda.py
"""

from pathlib import Path
import warnings

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


warnings.filterwarnings("ignore")


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

PROCESSED_DATA_DIR = (
    PROJECT_ROOT / "data" / "processed"
)

DATA_FILE = (
    PROCESSED_DATA_DIR /
    "cleaned_finance_data.csv"
)

REPORTS_DIR = (
    PROJECT_ROOT / "reports"
)

FIGURES_DIR = (
    REPORTS_DIR / "figures"
)


# ============================================================
# CREATE DIRECTORIES
# ============================================================

REPORTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)

FIGURES_DIR.mkdir(
    parents=True,
    exist_ok=True
)


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
            "Run data_preprocessing.py first."
        )

    print("=" * 70)
    print("LOADING PROCESSED DATASET")
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
# BASIC DATASET OVERVIEW
# ============================================================

def dataset_overview(df):
    """
    Generate basic dataset statistics.
    """

    print("\n" + "=" * 70)
    print("DATASET OVERVIEW")
    print("=" * 70)

    print("\nShape:")
    print(df.shape)

    print("\nColumns:")

    for index, column in enumerate(
        df.columns,
        start=1
    ):
        print(
            f"{index}. {column}"
        )

    print("\nData Types:")

    print(
        df.dtypes
    )

    print("\nFirst 5 rows:")

    print(
        df.head()
    )

    print("\nLast 5 rows:")

    print(
        df.tail()
    )


# ============================================================
# MISSING VALUE ANALYSIS
# ============================================================

def analyze_missing_values(df):
    """
    Analyze missing values.
    """

    print("\n" + "=" * 70)
    print("MISSING VALUE ANALYSIS")
    print("=" * 70)

    missing_count = (
        df.isnull()
        .sum()
    )

    missing_percentage = (
        missing_count /
        len(df) *
        100
    )

    missing_df = pd.DataFrame(
        {
            "missing_count":
                missing_count,

            "missing_percentage":
                missing_percentage
        }
    )

    missing_df = (
        missing_df
        .sort_values(
            "missing_count",
            ascending=False
        )
    )

    print(
        missing_df
    )

    missing_df.to_csv(
        REPORTS_DIR /
        "missing_values.csv"
    )

    return missing_df


# ============================================================
# DUPLICATE ANALYSIS
# ============================================================

def analyze_duplicates(df):
    """
    Analyze duplicate records.
    """

    print("\n" + "=" * 70)
    print("DUPLICATE ANALYSIS")
    print("=" * 70)

    duplicate_count = (
        df.duplicated()
        .sum()
    )

    duplicate_percentage = (
        duplicate_count /
        len(df) *
        100
    )

    print(
        f"Duplicate rows: "
        f"{duplicate_count}"
    )

    print(
        f"Duplicate percentage: "
        f"{duplicate_percentage:.2f}%"
    )

    return duplicate_count


# ============================================================
# NUMERICAL ANALYSIS
# ============================================================

def analyze_numerical_columns(df):
    """
    Generate descriptive statistics for numerical columns.
    """

    numerical_columns = (
        df.select_dtypes(
            include=np.number
        ).columns
    )

    print("\n" + "=" * 70)
    print("NUMERICAL FEATURE ANALYSIS")
    print("=" * 70)

    if len(numerical_columns) == 0:

        print(
            "No numerical columns found."
        )

        return None

    numerical_summary = (
        df[numerical_columns]
        .describe()
        .T
    )

    numerical_summary[
        "missing"
    ] = (
        df[numerical_columns]
        .isnull()
        .sum()
    )

    print(
        numerical_summary
    )

    numerical_summary.to_csv(
        REPORTS_DIR /
        "numerical_summary.csv"
    )

    return numerical_summary


# ============================================================
# CATEGORICAL ANALYSIS
# ============================================================

def analyze_categorical_columns(df):
    """
    Analyze categorical columns.
    """

    categorical_columns = (
        df.select_dtypes(
            include=["object", "category"]
        ).columns
    )

    print("\n" + "=" * 70)
    print("CATEGORICAL FEATURE ANALYSIS")
    print("=" * 70)

    if len(categorical_columns) == 0:

        print(
            "No categorical columns found."
        )

        return None

    summary = []

    for column in categorical_columns:

        unique_count = (
            df[column]
            .nunique()
        )

        most_common = (
            df[column]
            .mode()
        )

        if not most_common.empty:

            most_common_value = (
                most_common.iloc[0]
            )

        else:

            most_common_value = None

        summary.append(
            {
                "column":
                    column,

                "unique_values":
                    unique_count,

                "most_common":
                    most_common_value
            }
        )

        print(
            f"\n{column}"
        )

        print(
            f"Unique values: "
            f"{unique_count}"
        )

        print(
            "Top values:"
        )

        print(
            df[column]
            .value_counts()
            .head(10)
        )

    categorical_summary = pd.DataFrame(
        summary
    )

    categorical_summary.to_csv(
        REPORTS_DIR /
        "categorical_summary.csv",
        index=False
    )

    return categorical_summary


# ============================================================
# OUTLIER ANALYSIS
# ============================================================

def analyze_outliers(df):
    """
    Detect potential outliers using the IQR method.
    """

    numerical_columns = (
        df.select_dtypes(
            include=np.number
        ).columns
    )

    print("\n" + "=" * 70)
    print("OUTLIER ANALYSIS")
    print("=" * 70)

    results = []

    for column in numerical_columns:

        values = (
            df[column]
            .dropna()
        )

        if len(values) == 0:
            continue

        q1 = values.quantile(
            0.25
        )

        q3 = values.quantile(
            0.75
        )

        iqr = q3 - q1

        lower_bound = (
            q1 - 1.5 * iqr
        )

        upper_bound = (
            q3 + 1.5 * iqr
        )

        outliers = values[
            (values < lower_bound) |
            (values > upper_bound)
        ]

        count = len(
            outliers
        )

        percentage = (
            count /
            len(values) *
            100
        )

        results.append(
            {
                "column":
                    column,

                "outlier_count":
                    count,

                "outlier_percentage":
                    percentage,

                "lower_bound":
                    lower_bound,

                "upper_bound":
                    upper_bound
            }
        )

        print(
            f"{column}: "
            f"{count} outliers "
            f"({percentage:.2f}%)"
        )

    outlier_df = pd.DataFrame(
        results
    )

    outlier_df.to_csv(
        REPORTS_DIR /
        "outlier_analysis.csv",
        index=False
    )

    return outlier_df


# ============================================================
# CORRELATION ANALYSIS
# ============================================================

def create_correlation_heatmap(df):
    """
    Create a correlation heatmap for numerical variables.
    """

    numerical_df = (
        df.select_dtypes(
            include=np.number
        )
    )

    if numerical_df.shape[1] < 2:

        print(
            "\nNot enough numerical columns "
            "for correlation analysis."
        )

        return

    correlation = (
        numerical_df
        .corr()
    )

    print("\n" + "=" * 70)
    print("CORRELATION ANALYSIS")
    print("=" * 70)

    print(
        correlation
    )

    plt.figure(
        figsize=(14, 10)
    )

    sns.heatmap(
        correlation,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        linewidths=0.5
    )

    plt.title(
        "Financial Feature Correlation"
    )

    plt.tight_layout()

    output_file = (
        FIGURES_DIR /
        "correlation_heatmap.png"
    )

    plt.savefig(
        output_file,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"\nSaved:\n{output_file}"
    )


# ============================================================
# NUMERICAL DISTRIBUTIONS
# ============================================================

def create_numerical_distributions(df):
    """
    Create distribution plots for numerical variables.
    """

    numerical_columns = (
        df.select_dtypes(
            include=np.number
        ).columns
    )

    if len(numerical_columns) == 0:

        return

    # Limit the number of plots to avoid
    # extremely large figures.

    columns = list(
        numerical_columns[:12]
    )

    rows = int(
        np.ceil(
            len(columns) / 3
        )
    )

    fig, axes = plt.subplots(
        rows,
        3,
        figsize=(16, 5 * rows)
    )

    axes = np.array(
        axes
    ).reshape(-1)

    for index, column in enumerate(
        columns
    ):

        sns.histplot(
            df[column].dropna(),
            kde=True,
            ax=axes[index]
        )

        axes[index].set_title(
            f"Distribution of {column}"
        )

        axes[index].set_xlabel(
            column
        )

        axes[index].set_ylabel(
            "Frequency"
        )

    # Hide unused axes

    for index in range(
        len(columns),
        len(axes)
    ):

        axes[index].set_visible(
            False
        )

    plt.tight_layout()

    output_file = (
        FIGURES_DIR /
        "numerical_distributions.png"
    )

    plt.savefig(
        output_file,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"\nSaved:\n{output_file}"
    )


# ============================================================
# CATEGORICAL DISTRIBUTIONS
# ============================================================

def create_categorical_distributions(df):
    """
    Create plots for categorical variables.
    """

    categorical_columns = (
        df.select_dtypes(
            include=["object", "category"]
        ).columns
    )

    if len(categorical_columns) == 0:

        return

    columns = list(
        categorical_columns[:6]
    )

    rows = int(
        np.ceil(
            len(columns) / 2
        )
    )

    fig, axes = plt.subplots(
        rows,
        2,
        figsize=(16, 6 * rows)
    )

    axes = np.array(
        axes
    ).reshape(-1)

    for index, column in enumerate(
        columns
    ):

        value_counts = (
            df[column]
            .value_counts()
            .head(10)
        )

        sns.barplot(
            x=value_counts.values,
            y=value_counts.index,
            ax=axes[index]
        )

        axes[index].set_title(
            f"Top Categories - {column}"
        )

        axes[index].set_xlabel(
            "Count"
        )

        axes[index].set_ylabel(
            column
        )

    for index in range(
        len(columns),
        len(axes)
    ):

        axes[index].set_visible(
            False
        )

    plt.tight_layout()

    output_file = (
        FIGURES_DIR /
        "categorical_distributions.png"
    )

    plt.savefig(
        output_file,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"\nSaved:\n{output_file}"
    )


# ============================================================
# FINANCIAL FEATURE DETECTION
# ============================================================

def identify_financial_features(df):
    """
    Identify columns that appear to represent
    financial variables based on their names.
    """

    keywords = [
        "income",
        "salary",
        "expense",
        "spending",
        "saving",
        "savings",
        "rent",
        "loan",
        "insurance",
        "grocery",
        "groceries",
        "transport",
        "utility",
        "utilities",
        "health",
        "education",
        "entertainment",
        "eating",
        "miscellaneous",
        "investment",
        "debt",
        "emi",
    ]

    financial_columns = []

    for column in df.columns:

        column_lower = (
            column.lower()
        )

        if any(
            keyword in column_lower
            for keyword in keywords
        ):

            financial_columns.append(
                column
            )

    print("\n" + "=" * 70)
    print("POTENTIAL FINANCIAL FEATURES")
    print("=" * 70)

    if financial_columns:

        for column in financial_columns:
            print(
                f"  ✓ {column}"
            )

    else:

        print(
            "No financial features automatically detected."
        )

    return financial_columns


# ============================================================
# DATASET SUMMARY REPORT
# ============================================================

def create_summary_report(
    df,
    missing_df,
    outlier_df,
    financial_columns
):
    """
    Save a human-readable EDA summary.
    """

    output_file = (
        REPORTS_DIR /
        "dataset_summary.txt"
    )

    with open(
        output_file,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(
            "BUDGETPLANNER EDA REPORT\n"
        )

        file.write(
            "=" * 70 + "\n\n"
        )

        file.write(
            f"Rows: {df.shape[0]}\n"
        )

        file.write(
            f"Columns: {df.shape[1]}\n\n"
        )

        file.write(
            "COLUMNS\n"
        )

        file.write(
            "-" * 70 + "\n"
        )

        for column in df.columns:

            file.write(
                f"{column}: "
                f"{df[column].dtype}\n"
            )

        file.write(
            "\nMISSING VALUES\n"
        )

        file.write(
            "-" * 70 + "\n"
        )

        for column, row in (
            missing_df.iterrows()
        ):

            file.write(
                f"{column}: "
                f"{row['missing_count']} "
                f"({row['missing_percentage']:.2f}%)\n"
            )

        file.write(
            "\nPOTENTIAL FINANCIAL FEATURES\n"
        )

        file.write(
            "-" * 70 + "\n"
        )

        for column in financial_columns:

            file.write(
                f"{column}\n"
            )

        file.write(
            "\nOUTLIER SUMMARY\n"
        )

        file.write(
            "-" * 70 + "\n"
        )

        if outlier_df is not None:

            for _, row in (
                outlier_df.iterrows()
            ):

                file.write(
                    f"{row['column']}: "
                    f"{row['outlier_count']} "
                    f"({row['outlier_percentage']:.2f}%)\n"
                )

    print(
        f"\nEDA report saved:\n{output_file}"
    )


# ============================================================
# MAIN EDA PIPELINE
# ============================================================

def run_eda():

    print("\n")
    print("=" * 70)
    print("BUDGETPLANNER - EXPLORATORY DATA ANALYSIS")
    print("=" * 70)

    # --------------------------------------------------------
    # 1. Load data
    # --------------------------------------------------------

    df = load_dataset(
        DATA_FILE
    )

    # --------------------------------------------------------
    # 2. Overview
    # --------------------------------------------------------

    dataset_overview(
        df
    )

    # --------------------------------------------------------
    # 3. Missing values
    # --------------------------------------------------------

    missing_df = (
        analyze_missing_values(
            df
        )
    )

    # --------------------------------------------------------
    # 4. Duplicates
    # --------------------------------------------------------

    analyze_duplicates(
        df
    )

    # --------------------------------------------------------
    # 5. Numerical analysis
    # --------------------------------------------------------

    analyze_numerical_columns(
        df
    )

    # --------------------------------------------------------
    # 6. Categorical analysis
    # --------------------------------------------------------

    analyze_categorical_columns(
        df
    )

    # --------------------------------------------------------
    # 7. Outlier analysis
    # --------------------------------------------------------

    outlier_df = (
        analyze_outliers(
            df
        )
    )

    # --------------------------------------------------------
    # 8. Financial feature detection
    # --------------------------------------------------------

    financial_columns = (
        identify_financial_features(
            df
        )
    )

    # --------------------------------------------------------
    # 9. Correlation
    # --------------------------------------------------------

    create_correlation_heatmap(
        df
    )

    # --------------------------------------------------------
    # 10. Numerical distributions
    # --------------------------------------------------------

    create_numerical_distributions(
        df
    )

    # --------------------------------------------------------
    # 11. Categorical distributions
    # --------------------------------------------------------

    create_categorical_distributions(
        df
    )

    # --------------------------------------------------------
    # 12. Summary report
    # --------------------------------------------------------

    create_summary_report(
        df,
        missing_df,
        outlier_df,
        financial_columns
    )

    print("\n")
    print("=" * 70)
    print("EDA COMPLETED SUCCESSFULLY")
    print("=" * 70)

    print(
        f"\nReports saved in:\n{REPORTS_DIR}"
    )


# ============================================================
# PROGRAM ENTRY POINT
# ============================================================

if __name__ == "__main__":
    run_eda()