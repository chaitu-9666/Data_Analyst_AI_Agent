
from src.tool_selector import select_tool, ALLOWED_TOOLS


def test_dataset_overview_tool():
    result = select_tool("DATASET_OVERVIEW")

    assert result["tool_name"] == "get_dataset_overview"
    assert result["status"] == "selected"


def test_missing_values_tool():
    result = select_tool("MISSING_VALUES")

    assert result["tool_name"] == "get_missing_values"


def test_correlation_tool():
    result = select_tool("CORRELATION")

    assert result["tool_name"] == "get_correlation_matrix"


def test_average_tool():
    result = select_tool(
        "AGGREGATION",
        {"aggregation": "mean", "column": "Profit"}
    )

    assert result["tool_name"] == "get_column_average"
    assert result["status"] == "selected"


def test_average_tool_needs_column():
    result = select_tool(
        "AGGREGATION",
        {"aggregation": "mean"}
    )

    assert result["status"] == "needs_parameters"
    assert "column" in result["required_parameters"]


def test_top_records_tool():
    result = select_tool(
        "TOP_N",
        {"column": "Sales", "n": 10}
    )

    assert result["tool_name"] == "get_top_records"
    assert result["status"] == "selected"


def test_top_records_needs_parameters():
    result = select_tool("TOP_N", {})

    assert result["status"] == "needs_parameters"
    assert "column" in result["required_parameters"]
    assert "n" in result["required_parameters"]


def test_unsupported_intent():
    result = select_tool("UNKNOWN_INTENT")

    assert result["tool_name"] is None
    assert result["status"] == "unsupported"


def test_only_allowed_tools_are_selected():
    for intent in (
        "DATASET_OVERVIEW",
        "MISSING_VALUES",
        "CORRELATION",
        "TOP_N",
        "AGGREGATION",
    ):
        result = select_tool(intent, {
            "aggregation": "mean",
            "column": "Sales",
            "n": 5,
        })

        if result["tool_name"] is not None:
            assert result["tool_name"] in ALLOWED_TOOLS
