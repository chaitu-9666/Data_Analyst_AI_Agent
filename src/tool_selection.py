
"""Select an approved analysis tool for a user query."""

ALLOWED_TOOLS = {
    "get_dataset_overview",
    "get_missing_values",
    "get_numeric_summary",
    "get_top_records",
    "get_column_average",
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

    # Only a mean request can use the existing average function.
    if intent == "AGGREGATION":
        if entities.get("aggregation") == "mean":
            tool_name = "get_column_average"

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

    if tool_name in ("get_top_records", "get_column_average"):
        required_parameters.append("column")

    if tool_name == "get_top_records":
        required_parameters.append("n")

    missing_parameters = [
        parameter
        for parameter in required_parameters
        if entities.get(parameter) is None
    ]

    return {
        "tool_name": tool_name,
        "status": "needs_parameters" if missing_parameters else "selected",
        "required_parameters": missing_parameters,
    }
