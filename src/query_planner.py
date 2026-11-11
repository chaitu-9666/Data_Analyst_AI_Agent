
"""Combine NLU and planning modules into one query planner."""

from src.nlu.intent_detection import detect_intent
from src.nlu.entity_extraction import extract_entities

from src.query_planning import select_aggregation
from src.grouping_planner import build_grouping_plan
from src.visualization_planner import select_visualization
from src.tool_selector import select_tool


def build_query_plan(query, df):
    """
    Convert a natural-language question into a structured plan.

    This function plans the analysis but does not execute tools.
    """

    if not isinstance(query, str) or not query.strip():
        raise ValueError("Query must be a non-empty string.")

    if df is None or not hasattr(df, "columns"):
        raise TypeError("A valid Pandas DataFrame is required.")

    # 1. Detect intent.
    intent_result = detect_intent(query)

    if isinstance(intent_result, dict):
        intent = intent_result.get("intent")
    else:
        intent = intent_result

    if not intent:
        intent = "UNKNOWN"

    # 2. Extract entities.
    entities = extract_entities(query, df)

    # 3. Select aggregation.
    aggregation = select_aggregation(
        query,
        intent,
        entities
    )

    # 4. Prepare planning entities.
    planning_entities = dict(entities)
    planning_entities["aggregation"] = aggregation

    # Map extracted limit to the tool's expected parameter.
    planning_entities["n"] = entities.get("limit")

    # Identify explicitly mentioned numeric columns.
    numeric_columns = [
        column
        for column in entities.get("matched_columns", [])
        if column in df.columns
        and df[column].dtype.kind in "iufc"
    ]

    if len(numeric_columns) == 1:
        planning_entities["column"] = numeric_columns[0]
    else:
        planning_entities["column"] = None

    # 5. Build grouping, sorting and ranking plan.
    grouping_plan = build_grouping_plan(
        query,
        intent,
        planning_entities,
        df
    )

    # 6. Select visualization.
    visualization = select_visualization(
        query,
        intent,
        entities,
        df
    )

    # 7. Select an approved analysis tool.
    tool_plan = select_tool(
        intent,
        planning_entities
    )

    # 8. Return complete plan.
    return {
        "query": query,
        "intent": intent,
        "entities": entities,
        "aggregation": aggregation,
        "grouping": grouping_plan,
        "visualization": visualization,
        "tool": tool_plan,
    }
