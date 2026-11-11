
import re


AGGREGATION_PATTERNS = {
    "sum": [
        r"\btotal\b",
        r"\bsum\b",
        r"\boverall\b",
        r"\btotalled\b",
        r"\bhow many units\b",
        r"\bhow many items\b",
        r"\bunits sold\b",
        r"\bitems sold\b",
        r"\bsales by\b",
        r"\brevenue by\b",
        r"\bcosts by\b",
    ],

    "mean": [
        r"\baverage\b",
        r"\bmean\b",
        r"\bavg\b",
    ],

    "median": [
        r"\bmedian\b",
    ],

    "count": [
        r"\bcount\b",
        r"\bnumber of customers\b",
        r"\bhow many customers\b",
        r"\bnumber of orders\b",
        r"\bhow many orders\b",
        r"\bnumber of records\b",
    ],

    "max": [
        r"\bmaximum\b",
        r"\bhighest\b",
        r"\blargest\b",
        r"\bmaximum value\b",
    ],

    "min": [
        r"\bminimum\b",
        r"\blowest\b",
        r"\bsmallest\b",
        r"\bminimum value\b",
    ],

    "distribution": [
        r"\bdistribution\b",
        r"\bspread of\b",
        r"\bhistogram\b",
        r"\bfrequency distribution\b",
    ],
}


def select_aggregation(query, intent, entities):
    """
    Determine the analytical aggregation
    from the user question and extracted entities.
    """

    query = str(query).lower().strip()

    # Distribution intent takes priority.
    if intent == "DISTRIBUTION":
        return "distribution"

    # Check explicitly detected aggregation.
    detected = entities.get("aggregation")

    if detected in AGGREGATION_PATTERNS:
        return detected

    # Check aggregation keywords.
    for operation, patterns in AGGREGATION_PATTERNS.items():

        for pattern in patterns:

            if re.search(pattern, query):
                return operation

    # Infer aggregation for category comparisons.
    if intent == "COMPARISON":

        matched_columns = entities.get(
            "matched_columns", []
        )

        numeric_columns = entities.get(
            "numeric_columns", []
        )

        if numeric_columns:
            return "sum"

        # No numeric column: don't guess.
        return None

    return None
