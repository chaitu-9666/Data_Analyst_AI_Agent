
import pandas as pd
import pytest

from src.execution_engine import execute_plan


@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "Sales": [100, 200, 300, 400],
        "Profit": [20, 40, 60, 80],
        "Region": ["North", "South", "North", "West"]
    })


def make_plan(
    tool_name,
    entities=None,
    aggregation=None,
    status="selected"
):
    return {
        "entities": entities or {},
        "aggregation": aggregation,
        "grouping": {},
        "tool": {
            "tool_name": tool_name,
            "status": status,
            "required_parameters": [],
        },
    }


def test_execute_dataset_overview(sample_df):
    plan = make_plan("get_dataset_overview")

    result = execute_plan(sample_df, plan)

    assert result["status"] == "success"
    assert result["result"]["total_rows"] == 4


def test_execute_missing_values(sample_df):
    plan = make_plan("get_missing_values")

    result = execute_plan(sample_df, plan)

    assert result["status"] == "success"
    assert result["result"].empty


def test_execute_numeric_summary(sample_df):
    plan = make_plan("get_numeric_summary")

    result = execute_plan(sample_df, plan)

    assert result["status"] == "success"
    assert "Sales" in result["result"].index


def test_execute_correlation(sample_df):
    plan = make_plan("get_correlation_matrix")

    result = execute_plan(sample_df, plan)

    assert result["status"] == "success"
    assert result["result"].shape == (2, 2)


def test_execute_top_records(sample_df):
    plan = make_plan(
        "get_top_records",
        entities={
            "matched_columns": ["Sales"],
            "limit": 2,
        }
    )

    result = execute_plan(sample_df, plan)

    assert result["status"] == "success"
    assert len(result["result"]) == 2
    assert result["result"]["Sales"].tolist() == [400, 300]


def test_execute_average(sample_df):
    plan = make_plan(
        "get_column_average",
        entities={"matched_columns": ["Sales"]}
    )

    result = execute_plan(sample_df, plan)

    assert result["result"] == 250.0


def test_execute_sum_aggregation(sample_df):
    plan = make_plan(
        "get_column_aggregation",
        entities={"matched_columns": ["Sales"]},
        aggregation="sum"
    )

    result = execute_plan(sample_df, plan)

    assert result["result"] == 1000.0


def test_execute_count_aggregation(sample_df):
    plan = make_plan(
        "get_column_aggregation",
        entities={"matched_columns": ["Region"]},
        aggregation="count"
    )

    result = execute_plan(sample_df, plan)

    assert result["result"] == 4


def test_reject_unapproved_tool(sample_df):
    plan = make_plan("delete_all_files")

    with pytest.raises(ValueError):
        execute_plan(sample_df, plan)


def test_reject_tool_not_selected(sample_df):
    plan = make_plan(
        "get_dataset_overview",
        status="needs_parameters"
    )

    with pytest.raises(ValueError):
        execute_plan(sample_df, plan)


def test_reject_non_dictionary_plan(sample_df):
    with pytest.raises(TypeError):
        execute_plan(sample_df, "invalid plan")


def test_reject_missing_column(sample_df):
    plan = make_plan(
        "get_column_average",
        entities={"matched_columns": ["Unknown"]}
    )

    with pytest.raises(ValueError):
        execute_plan(sample_df, plan)


def test_reject_invalid_aggregation(sample_df):
    plan = make_plan(
        "get_column_aggregation",
        entities={"matched_columns": ["Sales"]},
        aggregation="multiply"
    )

    with pytest.raises(ValueError):
        execute_plan(sample_df, plan)

from src.execution_engine import execute_plan_safely


def test_safe_execution_returns_success(sample_df):
    plan = make_plan("get_dataset_overview")

    result = execute_plan_safely(sample_df, plan)

    assert result["status"] == "success"
    assert result["result"]["total_rows"] == 4


def test_safe_execution_handles_invalid_tool(sample_df):
    plan = make_plan("delete_all_files")

    result = execute_plan_safely(sample_df, plan)

    assert result["status"] == "error"
    assert result["error_type"] == "ValueError"
    assert result["result"] is None


def test_safe_execution_handles_missing_column(sample_df):
    plan = make_plan(
        "get_column_average",
        entities={"matched_columns": ["Unknown"]}
    )

    result = execute_plan_safely(sample_df, plan)

    assert result["status"] == "error"
    assert "column" in result["message"].lower()


def test_safe_execution_handles_invalid_plan(sample_df):
    result = execute_plan_safely(sample_df, "invalid")

    assert result["status"] == "error"
    assert result["error_type"] == "TypeError"


def test_safe_execution_handles_empty_dataframe():
    df = pd.DataFrame()

    plan = make_plan("get_dataset_overview")

    result = execute_plan_safely(df, plan)

    assert result["status"] == "error"
    assert result["result"] is None
