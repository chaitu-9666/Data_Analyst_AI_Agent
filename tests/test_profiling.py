
import pandas as pd

from src.profiling import (
    profile_dataset,
    get_column_profile,
    detect_outliers_iqr
)


def sample_data():

    return pd.DataFrame({
        "sales": [100, 200, 300, 400, 500, 10000],
        "quantity": [1, 2, 3, 4, 5, 100],
        "category": [
            "A", "B", "A", "B", "A", "C"
        ],
        "rating": [5, 4, None, 3, 5, 2]
    })


def test_profile_dataset():

    df = sample_data()

    result = profile_dataset(df)

    assert result["total_rows"] == 6
    assert result["total_columns"] == 4
    assert result["missing_cells"] == 1
    assert result["duplicate_rows"] == 0


def test_column_profile():

    df = sample_data()

    result = get_column_profile(df)

    rating = result[
        result["column"] == "rating"
    ].iloc[0]

    assert rating["missing_count"] == 1


def test_outlier_detection():

    df = sample_data()

    result = detect_outliers_iqr(df)

    sales_result = result[
        result["column"] == "sales"
    ].iloc[0]

    assert sales_result["outlier_count"] >= 1
