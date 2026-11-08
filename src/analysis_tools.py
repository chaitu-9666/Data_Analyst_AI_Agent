

import pandas as pd


def validate_dataframe(df: pd.DataFrame):
    """Validate that the input is a usable Pandas DataFrame."""

    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a Pandas DataFrame.")

    if df.empty:
        raise ValueError("The dataset is empty.")

    if len(df.columns) == 0:
        raise ValueError("The dataset contains no columns.")


def get_dataset_overview(df: pd.DataFrame) -> dict:
    """Return basic information about the dataset."""

    validate_dataframe(df)

    return {
        "total_rows": len(df),
        "total_columns": len(df.columns),
        "column_names": df.columns.tolist(),
        "duplicate_rows": int(df.duplicated().sum()),
        "missing_cells": int(df.isna().sum().sum()),
    }


def get_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Identify missing values in each column."""

    validate_dataframe(df)

    missing = df.isna().sum()

    result = pd.DataFrame({
        "column": missing.index,
        "missing_count": missing.values,
        "missing_percentage": (
            missing.values / len(df) * 100
        ).round(2),
    })

    return result[result["missing_count"] > 0].reset_index(drop=True)


def get_numeric_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Generate descriptive statistics for numerical columns."""

    validate_dataframe(df)

    numeric_df = df.select_dtypes(include="number")

    if numeric_df.empty:
        raise ValueError("No numerical columns found.")

    return numeric_df.describe().T


def get_top_records(
    df: pd.DataFrame,
    column: str,
    n: int = 10,
    ascending: bool = False
) -> pd.DataFrame:
    """Return the top or bottom records based on a column."""

    validate_dataframe(df)

    if column not in df.columns:
        raise ValueError(f"Column '{column}' does not exist.")

    if not pd.api.types.is_numeric_dtype(df[column]):
        raise TypeError(f"Column '{column}' must be numerical.")

    if not isinstance(n, int) or isinstance(n, bool) or n < 1:
        raise ValueError("n must be a positive integer.")

    return df.sort_values(
        by=column,
        ascending=ascending
    ).head(n)


def get_column_average(df: pd.DataFrame, column: str) -> float:
    """Calculate the average of a numerical column."""

    validate_dataframe(df)

    if column not in df.columns:
        raise ValueError(f"Column '{column}' does not exist.")

    if not pd.api.types.is_numeric_dtype(df[column]):
        raise TypeError(f"Column '{column}' must be numerical.")

    if df[column].count() == 0:
        raise ValueError(f"Column '{column}' has no valid numerical values.")

    return float(df[column].mean())


def get_correlation_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate correlations between numerical columns."""

    validate_dataframe(df)

    numeric_df = df.select_dtypes(include="number")

    if numeric_df.shape[1] < 2:
        raise ValueError(
            "At least two numerical columns are required."
        )

    return numeric_df.corr()
