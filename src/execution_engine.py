
"""Safe execution engine for approved DataFrame analysis tools."""

import pandas as pd

from src.analysis_tools import (
    validate_dataframe,
    get_dataset_overview,
    get_missing_values,
    get_numeric_summary,
    get_top_records,
    get_column_average,
    get_column_aggregation,
    get_correlation_matrix,
)


ALLOWED_TOOLS = {
    "get_dataset_overview",
    "get_missing_values",
    "get_numeric_summary",
    "get_top_records",
    "get_column_average",
    "get_column_aggregation",
    "get_correlation_matrix",
}


def _resolve_column(df, entities, require_numeric=True):
    """Resolve one unambiguous column from extracted entities."""

    matched_columns = entities.get("matched_columns", [])

    # Use an explicitly resolved column if it is valid.
    explicit_column = entities.get("column")

    if explicit_column in df.columns:
        candidates = [explicit_column]
    else:
        candidates = [
            column
            for column in matched_columns
            if column in df.columns
        ]

    if require_numeric:
        candidates = [
            column
            for column in candidates
            if pd.api.types.is_numeric_dtype(df[column])
        ]

    # Remove duplicates while preserving order.
    candidates = list(dict.fromkeys(candidates))

    if not candidates:
        raise ValueError(
            "No suitable column was identified for this operation."
        )

    if len(candidates) > 1:
        raise ValueError(
            "Multiple columns were identified. Please specify one column."
        )

    return candidates[0]


def execute_plan(df, plan):
    """
    Execute a validated query plan using approved analysis functions.

    This function does not execute arbitrary Python code.
    """

    validate_dataframe(df)

    if not isinstance(plan, dict):
        raise TypeError("Plan must be a dictionary.")

    tool_plan = plan.get("tool")

    if not isinstance(tool_plan, dict):
        raise ValueError("Plan does not contain a valid tool selection.")

    if tool_plan.get("status") != "selected":
        raise ValueError(
            "The requested analysis tool is not ready for execution."
        )

    tool_name = tool_plan.get("tool_name")

    if tool_name not in ALLOWED_TOOLS:
        raise ValueError(
            f"Tool '{tool_name}' is not approved for execution."
        )

    entities = plan.get("entities", {})
    if not isinstance(entities, dict):
        raise TypeError("Plan entities must be a dictionary.")

    grouping = plan.get("grouping", {})
    if not isinstance(grouping, dict):
        grouping = {}

    # Execute only explicitly approved functions.
    if tool_name == "get_dataset_overview":
        result = get_dataset_overview(df)

    elif tool_name == "get_missing_values":
        result = get_missing_values(df)

    elif tool_name == "get_numeric_summary":
        result = get_numeric_summary(df)

    elif tool_name == "get_correlation_matrix":
        result = get_correlation_matrix(df)

    elif tool_name == "get_top_records":
        column = _resolve_column(
            df,
            entities,
            require_numeric=True
        )

        n = entities.get("limit")
        if n is None:
            n = grouping.get("limit")

        if not isinstance(n, int) or isinstance(n, bool) or n < 1:
            raise ValueError(
                "A positive integer limit is required for top-record analysis."
            )

        sort = grouping.get("sort") or entities.get("sort")
        ascending = sort == "ascending"

        result = get_top_records(
            df,
            column,
            n=n,
            ascending=ascending
        )

    elif tool_name == "get_column_average":
        column = _resolve_column(
            df,
            entities,
            require_numeric=True
        )

        result = get_column_average(df, column)

    elif tool_name == "get_column_aggregation":
        aggregation = plan.get("aggregation")

        if aggregation not in {
            "sum",
            "median",
            "count",
            "min",
            "max",
        }:
            raise ValueError(
                "A supported aggregation is required."
            )

        column = _resolve_column(
            df,
            entities,
            require_numeric=(aggregation != "count")
        )

        result = get_column_aggregation(
            df,
            column,
            aggregation
        )

    else:
        # Defensive fallback: never execute an unknown tool.
        raise ValueError("Unsupported analysis tool.")

    return {
        "status": "success",
        "tool_name": tool_name,
        "result": result,
    }

def execute_plan_safely(df, plan):
    """
    Execute a plan and return a structured response instead of
    allowing expected or unexpected execution errors to crash the app.
    """

    tool_name = None

    if isinstance(plan, dict):
        tool_plan = plan.get("tool")

        if isinstance(tool_plan, dict):
            tool_name = tool_plan.get("tool_name")

    try:
        result = execute_plan(df, plan)

        return result

    except (TypeError, ValueError) as error:
        return {
            "status": "error",
            "tool_name": tool_name,
            "error_type": type(error).__name__,
            "message": str(error),
            "result": None,
        }

    except Exception:
        # Do not expose internal implementation details to the user.
        return {
            "status": "error",
            "tool_name": tool_name,
            "error_type": "ExecutionError",
            "message": (
                "An unexpected error occurred while executing "
                "the analysis. Please check the dataset and try again."
            ),
            "result": None,
        }
