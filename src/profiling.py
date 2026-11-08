
import re
import pandas as pd
import numpy as np


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

# ==========================================================
# 7. ADVANCED OUTLIER DETECTION
# ==========================================================

def detect_outliers_advanced(
    df: pd.DataFrame,
    z_threshold: float = 3.0,
    modified_z_threshold: float = 3.5
) -> pd.DataFrame:
    """
    Detect numerical outliers using three statistical methods:
    1. IQR
    2. Z-Score
    3. Modified Z-Score

    Identifier and year columns are excluded.
    """

    validate_dataframe(df)

    if z_threshold <= 0 or modified_z_threshold <= 0:
        raise ValueError("Outlier thresholds must be positive.")

    numeric_columns = df.select_dtypes(
        include="number"
    ).columns.tolist()

    results = []

    for column in numeric_columns:

        column_name = str(column).lower()

        # Exclude identifier and year columns
        if (
            column_name.endswith("_id")
            or column_name == "id"
            or "identifier" in column_name
            or column_name.endswith("_year")
        ):
            continue

        series = pd.to_numeric(
            df[column], errors="coerce"
        ).dropna()

        # Remove infinite values
        series = series[np.isfinite(series)]

        if len(series) < 4:
            continue

        def add_result(method, outliers, lower, upper):

            results.append({
                "column": column,
                "method": method,
                "total_values": int(len(series)),
                "outlier_count": int(outliers.sum()),
                "outlier_percentage": round(
                    outliers.sum() / len(series) * 100, 2
                ),
                "lower_bound": (
                    round(float(lower), 2)
                    if np.isfinite(lower) else None
                ),
                "upper_bound": (
                    round(float(upper), 2)
                    if np.isfinite(upper) else None
                )
            })

        # --------------------------------------------------
        # METHOD 1: IQR
        # --------------------------------------------------

        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1

        if iqr > 0:

            lower = q1 - (1.5 * iqr)
            upper = q3 + (1.5 * iqr)

            outliers = (
                (series < lower) | (series > upper)
            )

            add_result("IQR", outliers, lower, upper)

        # --------------------------------------------------
        # METHOD 2: Z-SCORE
        # --------------------------------------------------

        mean = series.mean()
        std = series.std(ddof=0)

        if std > 0:

            z_scores = (series - mean) / std

            outliers = z_scores.abs() > z_threshold

            lower = mean - (z_threshold * std)
            upper = mean + (z_threshold * std)

            add_result("Z-Score", outliers, lower, upper)

        # --------------------------------------------------
        # METHOD 3: MODIFIED Z-SCORE
        # --------------------------------------------------

        median = series.median()

        mad = (series - median).abs().median()

        if mad > 0:

            modified_z_scores = (
                0.6745 * (series - median) / mad
            )

            outliers = (
                modified_z_scores.abs() > modified_z_threshold
            )

            lower = (
                median -
                (modified_z_threshold * mad / 0.6745)
            )

            upper = (
                median +
                (modified_z_threshold * mad / 0.6745)
            )

            add_result(
                "Modified Z-Score",
                outliers,
                lower,
                upper
            )

    return pd.DataFrame(
        results,
        columns=[
            "column",
            "method",
            "total_values",
            "outlier_count",
            "outlier_percentage",
            "lower_bound",
            "upper_bound"
        ]
    )


# ==========================================================
# 8. DATA QUALITY SCORING
# ==========================================================

def calculate_data_quality_score(df: pd.DataFrame) -> dict:
    """
    Calculate overall dataset and individual column
    quality scores using:

    1. Completeness - 50%
    2. Uniqueness - 25%
    3. Outlier Health - 25%

    Unavailable components are excluded and remaining
    weights are proportionally adjusted.
    """

    validate_dataframe(df)

    total_rows, total_columns = df.shape
    total_cells = total_rows * total_columns

    # ------------------------------------------------------
    # 1. COMPLETENESS SCORE
    # ------------------------------------------------------

    missing_cells = int(df.isna().sum().sum())

    completeness_score = round(
        (1 - missing_cells / total_cells) * 100, 2
    )

    # ------------------------------------------------------
    # 2. UNIQUENESS SCORE
    # ------------------------------------------------------

    duplicate_rows = int(df.duplicated().sum())

    uniqueness_score = round(
        (1 - duplicate_rows / total_rows) * 100, 2
    )

    # ------------------------------------------------------
    # 3. OUTLIER HEALTH SCORE
    # ------------------------------------------------------

    outlier_results = detect_outliers_iqr(df)

    outlier_health_score = None
    outlier_columns_assessed = 0

    if not outlier_results.empty:

        assessed_values = 0
        total_outliers = 0

        for _, row in outlier_results.iterrows():

            column = row["column"]

            valid_count = int(df[column].notna().sum())

            assessed_values += valid_count
            total_outliers += int(row["outlier_count"])

        if assessed_values > 0:

            outlier_health_score = round(
                max(
                    0,
                    100 - (total_outliers / assessed_values * 100)
                ),
                2
            )

            outlier_columns_assessed = len(outlier_results)

    # ------------------------------------------------------
    # 4. OVERALL DATASET QUALITY SCORE
    # ------------------------------------------------------

    components = {
        "completeness": completeness_score,
        "uniqueness": uniqueness_score
    }

    weights = {
        "completeness": 50,
        "uniqueness": 25
    }

    if outlier_health_score is not None:

        components["outlier_health"] = outlier_health_score
        weights["outlier_health"] = 25

    total_weight = sum(weights.values())

    overall_score = round(
        sum(
            components[name] * weight
            for name, weight in weights.items()
        ) / total_weight,
        2
    )

    # ------------------------------------------------------
    # 5. QUALITY GRADE
    # ------------------------------------------------------

    def get_grade(score):

        if score >= 90:
            return "Excellent"

        elif score >= 75:
            return "Good"

        elif score >= 60:
            return "Fair"

        return "Poor"

    # ------------------------------------------------------
    # 6. INDIVIDUAL COLUMN QUALITY SCORES
    # ------------------------------------------------------

    column_records = []

    outlier_lookup = {}

    if not outlier_results.empty:

        outlier_lookup = {
            row["column"]: row
            for _, row in outlier_results.iterrows()
        }

    for column in df.columns:

        series = df[column]

        missing_percentage = round(
            series.isna().mean() * 100, 2
        )

        column_completeness = round(
            100 - missing_percentage, 2
        )

        column_outlier_score = None
        outlier_count = None
        outlier_percentage = None

        if column in outlier_lookup:

            outlier_row = outlier_lookup[column]

            outlier_count = int(
                outlier_row["outlier_count"]
            )

            outlier_percentage = float(
                outlier_row["outlier_percentage"]
            )

            column_outlier_score = round(
                100 - outlier_percentage, 2
            )

            column_score = round(
                column_completeness * 0.70 +
                column_outlier_score * 0.30,
                2
            )

        else:

            # No outlier assessment available:
            # score this column on completeness only.
            column_score = column_completeness

        column_records.append({

            "column": column,

            "data_type": str(series.dtype),

            "missing_count": int(series.isna().sum()),

            "missing_percentage": missing_percentage,

            "completeness_score": column_completeness,

            "outlier_count": outlier_count,

            "outlier_percentage": outlier_percentage,

            "outlier_score": column_outlier_score,

            "quality_score": column_score,

            "quality_grade": get_grade(column_score)

        })

    column_scores = pd.DataFrame(column_records)

    # ------------------------------------------------------
    # 7. FINAL QUALITY REPORT
    # ------------------------------------------------------

    return {

        "overall_score": overall_score,

        "overall_grade": get_grade(overall_score),

        "component_scores": components,

        "total_rows": int(total_rows),

        "total_columns": int(total_columns),

        "total_cells": int(total_cells),

        "missing_cells": missing_cells,

        "duplicate_rows": duplicate_rows,

        "outlier_columns_assessed": outlier_columns_assessed,

        "column_scores": column_scores

    }
