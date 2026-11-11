
import re


def select_group_by(intent, entities, df):
    """Choose the categorical column used for grouping."""

    matched_columns = entities.get("matched_columns", [])

    if intent not in ("COMPARISON", "TOP_N", "AGGREGATION"):
        return None

    for column in matched_columns:
        if column not in df.columns:
            continue

        if (
            df[column].dtype.kind not in "biufc"
            or df[column].nunique(dropna=True) <= 20
        ):
            return column

    return None


def select_sort(query, entities):
    """Determine sorting direction."""

    query = str(query).lower()

    ascending_patterns = [
        r"\bbottom\b",
        r"\blowest\b",
        r"\bsmallest\b",
        r"\bworst\b",
        r"\bascending\b",
        r"\bincreasing\b",
    ]

    descending_patterns = [
        r"\btop\b",
        r"\bhighest\b",
        r"\blargest\b",
        r"\bbest\b",
        r"\bdescending\b",
        r"\bdecreasing\b",
    ]

    if any(re.search(p, query) for p in ascending_patterns):
        return "ascending"

    if any(re.search(p, query) for p in descending_patterns):
        return "descending"

    return None


def select_limit(query, entities):
    """Extract the requested top-N or bottom-N limit."""

    limit = entities.get("limit")

    if isinstance(limit, int) and limit > 0:
        return limit

    match = re.search(
        r"\b(?:top|bottom)\s+(\d+)\b",
        str(query).lower()
    )

    return int(match.group(1)) if match else None


def build_grouping_plan(query, intent, entities, df):
    """Build a structured grouping and ranking plan."""

    group_by = select_group_by(intent, entities, df)
    sort = select_sort(query, entities)
    limit = select_limit(query, entities)

    aggregation = entities.get("aggregation")

    matched_columns = entities.get("matched_columns", [])

    # Check whether the query explicitly mentions a numeric measure.
    numeric_measure = any(
        column in df.columns
        and df[column].dtype.kind in "iufc"
        for column in matched_columns
    )

    # Only infer SUM when a numeric measure is explicitly identified.
    if (
        group_by is not None
        and sort is not None
        and numeric_measure
        and aggregation in (None, "max", "min")
    ):
        aggregation = "sum"

    return {
        "group_by": group_by,
        "sort": sort,
        "limit": limit,
        "aggregation": aggregation,
    }
