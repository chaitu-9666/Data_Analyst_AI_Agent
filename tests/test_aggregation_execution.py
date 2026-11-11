
import pandas as pd
import pytest

from src.analysis_tools import get_column_aggregation


@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "Sales_Amount": [100, 200, 300, 400],
        "Profit": [20, 40, 60, 80],
        "Region": ["North", "South", "North", "West"]
    })


def test_sum_aggregation(sample_df):
    result = get_column_aggregation(
        sample_df, "Sales_Amount", "sum"
    )
    assert result == 1000.0


def test_mean_aggregation(sample_df):
    result = get_column_aggregation(
        sample_df, "Profit", "mean"
    )
    assert result == 50.0


def test_median_aggregation(sample_df):
    result = get_column_aggregation(
        sample_df, "Profit", "median"
    )
    assert result == 50.0


def test_count_aggregation(sample_df):
    result = get_column_aggregation(
        sample_df, "Region", "count"
    )
    assert result == 4


def test_min_aggregation(sample_df):
    result = get_column_aggregation(
        sample_df, "Sales_Amount", "min"
    )
    assert result == 100.0


def test_max_aggregation(sample_df):
    result = get_column_aggregation(
        sample_df, "Sales_Amount", "max"
    )
    assert result == 400.0


def test_count_excludes_missing_values():
    df = pd.DataFrame({"Sales": [100, None, 300]})

    result = get_column_aggregation(
        df, "Sales", "count"
    )

    assert result == 2


def test_invalid_column_raises_error(sample_df):
    with pytest.raises(ValueError):
        get_column_aggregation(
            sample_df, "Invalid_Column", "sum"
        )


def test_invalid_aggregation_raises_error(sample_df):
    with pytest.raises(ValueError):
        get_column_aggregation(
            sample_df, "Sales_Amount", "multiply"
        )


def test_non_numeric_sum_raises_error(sample_df):
    with pytest.raises(TypeError):
        get_column_aggregation(
            sample_df, "Region", "sum"
        )


def test_all_missing_numeric_column_raises_error():
    df = pd.DataFrame({"Sales": [None, None, None]})

    with pytest.raises(ValueError):
        get_column_aggregation(
            df, "Sales", "sum"
        )
