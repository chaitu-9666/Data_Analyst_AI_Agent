
"""Select an approved analysis tool for a user query."""

ALLOWED_TOOLS = {
    "get_dataset_overview",
    "get_missing_values",
    "get_numeric_summary",
    "get_top_records",
    "get_column_average",
    "get_column_aggregation",
    "get_correlation_matrix",
}

TOOL_MAPPING = {
    "DATASET_OVERVIEW": "get_dataset_overview",
    "MISSING_VALUES": "get_missing_values",
    "CORRELATION": "get_correlation_matrix",
    "TOP_N": "get_top_records",
}


def select_tool(intent, entities=None):
    entities = entities or {}

    tool_name = TOOL_MAPPING.get(intent)

    # Select an approved tool for aggregation requests.
    if intent == "AGGREGATION":
        aggregation = entities.get("aggregation")

        if aggregation == "mean":
            tool_name = "get_column_average"

        elif aggregation in {
            "sum",
            "median",
            "count",
            "min",
            "max",
        }:
            tool_name = "get_column_aggregation"

    if tool_name is None:
        return {
            "tool_name": None,
            "status": "unsupported",
            "required_parameters": [],
        }

    if tool_name not in ALLOWED_TOOLS:
        return {
            "tool_name": None,
            "status": "blocked",
            "required_parameters": [],
        }

    required_parameters = []

    if tool_name in (
        "get_top_records",
        "get_column_average",
        "get_column_aggregation",
    ):
        required_parameters.append("column")

    if tool_name == "get_top_records":
        required_parameters.append("n")

    if tool_name == "get_column_aggregation":
        required_parameters.append("aggregation")

    missing_parameters = [
        parameter
        for parameter in required_parameters
        if entities.get(parameter) is None
    ]

    return {
        "tool_name": tool_name,
        "status": (
            "needs_parameters"
            if missing_parameters
            else "selected"
        ),
        "required_parameters": missing_parameters,
    }
