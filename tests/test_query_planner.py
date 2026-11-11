
import pandas as pd
import pytest

from src.query_planner import build_query_plan


@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "Sales_Amount": [100, 200, 300, 400],
        "Profit": [20, 40, 60, 80],
        "Region": ["North", "South", "North", "South"],
        "Product_Name": ["A", "B", "C", "D"],
        "Order_Date": pd.to_datetime([
            "2025-01-01",
            "2025-02-01",
            "2025-03-01",
            "2025-04-01",
        ]),
    })


def test_dataset_overview_plan(sample_df):
    plan = build_query_plan(
        "Show dataset overview",
        sample_df
    )

    assert plan["intent"] == "DATASET_OVERVIEW"
    assert plan["tool"]["tool_name"] == "get_dataset_overview"
    assert plan["visualization"] == "kpi"


def test_missing_values_plan(sample_df):
    plan = build_query_plan(
        "Show missing values",
        sample_df
    )

    assert plan["intent"] == "MISSING_VALUES"
    assert plan["tool"]["tool_name"] == "get_missing_values"


def test_correlation_plan(sample_df):
    plan = build_query_plan(
        "Show correlation between Sales_Amount and Profit",
        sample_df
    )

    assert plan["intent"] == "CORRELATION"
    assert plan["visualization"] == "heatmap"
    assert plan["tool"]["tool_name"] == "get_correlation_matrix"


def test_average_plan(sample_df):
    plan = build_query_plan(
        "What is the average Profit?",
        sample_df
    )

    assert plan["aggregation"] == "mean"
    assert plan["tool"]["tool_name"] == "get_column_average"
    assert plan["tool"]["status"] == "selected"


def test_top_records_plan(sample_df):
    plan = build_query_plan(
        "Show top 3 Sales_Amount",
        sample_df
    )

    assert plan["tool"]["tool_name"] == "get_top_records"
    assert plan["tool"]["status"] == "selected"
    assert plan["grouping"]["limit"] == 3


def test_top_products_needs_numeric_measure(sample_df):
    plan = build_query_plan(
        "Show top 3 products",
        sample_df
    )

    assert plan["grouping"]["group_by"] == "Product_Name"
    assert plan["tool"]["status"] == "needs_parameters"
    assert "column" in plan["tool"]["required_parameters"]


def test_empty_query_raises_error(sample_df):
    with pytest.raises(ValueError):
        build_query_plan("", sample_df)
