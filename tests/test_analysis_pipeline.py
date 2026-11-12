
import pandas as pd

from src.analysis_pipeline import run_analysis


def sample_df():
    return pd.DataFrame({
        "Sales": [100, 200, 300, 400],
        "Profit": [20, 40, 60, 80],
        "Region": ["North", "South", "North", "West"]
    })


def test_pipeline_executes_dataset_overview():
    df = sample_df()

    response = run_analysis(
        "show dataset overview",
        df
    )

    assert response["status"] == "success"
    assert response["plan"] is not None
    assert response["execution"]["result"]["total_rows"] == 4


def test_pipeline_executes_sum_aggregation():
    df = sample_df()

    response = run_analysis(
        "what is the total Sales",
        df
    )

    assert response["status"] == "success"
    assert response["execution"]["result"] == 1000.0


def test_pipeline_handles_unsupported_query():
    df = sample_df()

    response = run_analysis(
        "hello there",
        df
    )

    assert response["status"] == "unsupported"
    assert response["execution"] is None


def test_pipeline_handles_empty_query():
    df = sample_df()

    response = run_analysis("", df)

    assert response["status"] == "error"
    assert response["plan"] is None
    assert response["execution"] is None
