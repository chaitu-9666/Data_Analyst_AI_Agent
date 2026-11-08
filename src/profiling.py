
import re
import pandas as pd


# ==========================================================
# 1. DATAFRAME VALIDATION
# ==========================================================

def validate_dataframe(df: pd.DataFrame):
    """Check whether the input is a usable DataFrame."""

    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a Pandas DataFrame.")

    if df.empty:
        raise ValueError("Dataset is empty.")

    if len(df.columns) == 0:
        raise ValueError("Dataset has no columns.")


# ==========================================================
# 2. AUTOMATIC TIME DETECTION
# ==========================================================

def detect_time_columns(df: pd.DataFrame) -> list:
    """Identify columns containing time-only values."""

    time_columns = []

    time_pattern = (
        r"^\d{1,2}:\d{2}"
        r"(?::\d{2}(?:\.\d+)?)?"
        r"(?:\s?[APap][Mm])?$"
    )

    for column in df.columns:

        if not any(
            word in column.lower()
            for word in ["time", "clock"]
        ):
            continue

        series = df[column].dropna().astype(str).str.strip()

        if series.empty:
            continue

        valid_ratio = series.str.match(
            time_pattern
        ).mean()

        if valid_ratio >= 0.80:
            time_columns.append(column)

    return time_columns


# ==========================================================
# 3. AUTOMATIC DATETIME DETECTION
# ==========================================================

def detect_datetime_columns(df: pd.DataFrame) -> list:
    """
    Identify date columns using column names
    and successful datetime parsing.
    """

    datetime_columns = []

    time_columns = detect_time_columns(df)

    date_keywords = [
        "date",
        "datetime",
        "timestamp"
    ]

    for column in df.columns:

        if column in time_columns:
            continue

        if pd.api.types.is_numeric_dtype(df[column]):
            continue

        column_name = column.lower()

        name_suggests_datetime = any(
            keyword in column_name
            for keyword in date_keywords
        )

        if not name_suggests_datetime:
            continue

        series = df[column].dropna()

        if series.empty:
            continue

        try:
            parsed = pd.to_datetime(
                series,
                errors="coerce",
                format="mixed"
            )

        except (TypeError, ValueError):
            parsed = pd.to_datetime(
                series,
                errors="coerce"
            )

        valid_ratio = parsed.notna().mean()

        if valid_ratio >= 0.80:
            datetime_columns.append(column)

    return datetime_columns


# ==========================================================
# 4. OVERALL DATASET PROFILE
# ==========================================================

def profile_dataset(df: pd.DataFrame) -> dict:
    """Generate an overall dataset profile."""

    validate_dataframe(df)

    missing_cells = int(df.isna().sum().sum())

    total_cells = int(
        df.shape[0] * df.shape[1]
    )

    numeric_columns = df.select_dtypes(
        include="number"
    ).columns.tolist()

    time_columns = detect_time_columns(df)

    datetime_columns = detect_datetime_columns(df)

    categorical_columns = [
        col for col in df.select_dtypes(
            include=["object", "category", "string"]
        ).columns.tolist()
        if col not in datetime_columns
        and col not in time_columns
    ]

    return {
        "total_rows": int(df.shape[0]),
        "total_columns": int(df.shape[1]),
        "total_cells": total_cells,
        "missing_cells": missing_cells,
        "missing_percentage": round(
            missing_cells / total_cells * 100, 2
        ),
        "duplicate_rows": int(df.duplicated().sum()),
        "numeric_columns": numeric_columns,
        "categorical_columns": categorical_columns,
        "datetime_columns": datetime_columns,
        "time_columns": time_columns,
    }


# ==========================================================
# 5. ADVANCED COLUMN PROFILING
# ==========================================================

def get_column_profile(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate detailed statistics for every column.
    """

    validate_dataframe(df)

    datetime_columns = detect_datetime_columns(df)
    time_columns = detect_time_columns(df)

    records = []

    for column in df.columns:

        series = df[column]

        record = {
            "column": column,
            "data_type": str(series.dtype),
            "non_null_count": int(series.count()),
            "missing_count": int(series.isna().sum()),
            "missing_percentage": round(
                series.isna().mean() * 100, 2
            ),
            "unique_values": int(series.nunique()),
            "sample_values": ", ".join(
                series.dropna().astype(str).unique()[:3]
            ),
            "minimum": None,
            "maximum": None,
            "mean": None,
            "median": None,
            "std_dev": None,
            "most_frequent": None,
            "most_frequent_count": None,
        }

        # Date columns
        if column in datetime_columns:

            parsed = pd.to_datetime(
                series,
                errors="coerce",
                format="mixed"
            ).dropna()

            record["data_type"] = "datetime"

            if not parsed.empty:
                record["minimum"] = str(parsed.min())
                record["maximum"] = str(parsed.max())

        # Time columns
        elif column in time_columns:

            record["data_type"] = "time"

        # Numerical columns
        elif pd.api.types.is_numeric_dtype(series):

            valid = series.dropna()

            if not valid.empty:

                record["minimum"] = float(valid.min())

                record["maximum"] = float(valid.max())

                record["mean"] = round(
                    float(valid.mean()), 2
                )

                record["median"] = round(
                    float(valid.median()), 2
                )

                record["std_dev"] = round(
                    float(valid.std()), 2
                ) if len(valid) > 1 else 0.0

        # Categorical columns
        else:

            modes = series.mode(dropna=True)

            if not modes.empty:

                record["most_frequent"] = str(
                    modes.iloc[0]
                )

                record["most_frequent_count"] = int(
                    (series == modes.iloc[0]).sum()
                )

        records.append(record)

    return pd.DataFrame(records)


# ==========================================================
# 6. OUTLIER DETECTION USING IQR
# ==========================================================

def detect_outliers_iqr(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detect potential numerical outliers using IQR.

    Identifier and year columns are excluded because
    they are not generally suitable for this analysis.
    """

    validate_dataframe(df)

    numeric_columns = df.select_dtypes(
        include="number"
    ).columns.tolist()

    results = []

    for column in numeric_columns:

        column_name = column.lower()

        # Exclude identifiers and year fields
        if (
            column_name.endswith("_id")
            or column_name == "id"
            or "identifier" in column_name
            or column_name.endswith("_year")
        ):
            continue

        series = df[column].dropna()

        if len(series) < 4:
            continue

        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)

        iqr = q3 - q1

        if iqr == 0:
            continue

        lower_bound = q1 - (1.5 * iqr)
        upper_bound = q3 + (1.5 * iqr)

        outliers = series[
            (series < lower_bound) |
            (series > upper_bound)
        ]

        results.append({
            "column": column,
            "q1": round(float(q1), 2),
            "q3": round(float(q3), 2),
            "iqr": round(float(iqr), 2),
            "lower_bound": round(float(lower_bound), 2),
            "upper_bound": round(float(upper_bound), 2),
            "outlier_count": int(len(outliers)),
            "outlier_percentage": round(
                len(outliers) / len(series) * 100, 2
            ),
        })

    return pd.DataFrame(results)
