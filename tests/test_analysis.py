
import pytest
import pandas as pd

from src.analysis_tools import analyze_question


@pytest.fixture
def sales_data():
    return pd.DataFrame(
        {
            "Category": ["Books", "Games", "Books", "Games"],
            "Sales": [10, 30, 20, 40],
            "Salary": [100, 200, 300, 400],
            "Bonus": [1, 2, 3, 4],
        }
    )


def test_question_reports_dataset_dimensions(sales_data):
    result = analyze_question(
        sales_data, "How many rows and columns are in my dataset?"
    )

    assert result.operation == "dimensions"
    assert result.answer == "The dataset has 4 rows and 4 columns."


def test_question_lists_only_columns_with_missing_values():
    df = pd.DataFrame({"Name": ["A", None], "Sales": [None, None]})

    result = analyze_question(df, "Which columns have missing values?")

    assert result.operation == "missing_values"
    assert result.data["Column"].tolist() == ["Name", "Sales"]
    assert result.data["Missing Values"].tolist() == [1, 2]


def test_question_returns_top_records(sales_data):
    result = analyze_question(sales_data, "Show me the top 2 records by sales.")

    assert result.operation == "top_records"
    assert result.data["Sales"].tolist() == [40, 30]


def test_question_calculates_numeric_average(sales_data):
    result = analyze_question(sales_data, "What is the average salary?")

    assert result.operation == "average"
    assert result.answer == "The average Salary is 250."


def test_average_matches_a_term_within_a_multiword_column_name():
    df = pd.DataFrame({"Annual Salary": [100, 200, 300]})

    result = analyze_question(df, "What is the average salary?")

    assert result.operation == "average"
    assert result.answer == "The average Annual Salary is 200."


def test_question_finds_strongest_correlation(sales_data):
    result = analyze_question(sales_data, "Which columns have the strongest correlation?")

    assert result.operation == "strongest_correlation"
    assert "Salary and Bonus" in result.answer
    assert result.data.loc[0, "Correlation"] == 1


def test_question_creates_aggregated_chart_data(sales_data):
    result = analyze_question(
        sales_data, "Create a chart showing sales by category."
    )

    assert result.operation == "grouped_chart"
    assert result.chart
    assert result.data.to_dict("records") == [
        {"Category": "Books", "Sales": 30},
        {"Category": "Games", "Sales": 70},
    ]


def test_unsupported_question_does_not_execute_code(sales_data):
    result = analyze_question(sales_data, "__import__('os').system('echo unsafe')")

    assert result.operation == "unsupported"
    assert result.data is None


def test_average_of_non_numeric_column_returns_guidance(sales_data):
    result = analyze_question(sales_data, "What is the average category?")

    assert result.operation == "average"
    assert result.data is None
    assert "not a numeric column" in result.answer
