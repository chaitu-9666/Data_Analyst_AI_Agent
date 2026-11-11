
import re


VISUALIZATION_PATTERNS = {
    "histogram": [
        r"\bhistogram\b",
        r"\bdistribution\b",
        r"\bspread of\b",
        r"\bfrequency distribution\b",
    ],
    "line": [
        r"\btrend\b",
        r"\bover time\b",
        r"\bmonthly\b",
        r"\bweekly\b",
        r"\byearly\b",
        r"\bper month\b",
        r"\bper year\b",
        r"\btime series\b",
    ],
    "heatmap": [
        r"\bcorrelation\b",
        r"\bcorrelated\b",
        r"\brelationship between\b",
    ],
    "pie": [
        r"\bshare\b",
        r"\bpercentage\b",
        r"\bproportion\b",
        r"\bbreakdown percentage\b",
    ],
    "bar": [
        r"\bcompare\b",
        r"\bcomparison\b",
        r"\bby region\b",
        r"\bby category\b",
        r"\bby product\b",
        r"\bby customer\b",
        r"\btop\b",
        r"\bbottom\b",
        r"\bhighest\b",
        r"\blowest\b",
    ],
}


def select_visualization(query, intent, entities, df):
    """
    Select an appropriate visualization based on
    the user's query, detected intent and dataset.
    """

    query = str(query).lower().strip()

    # Correlation analysis
    if intent == "CORRELATION":
        return "heatmap"

    # Distribution analysis
    if intent == "DISTRIBUTION":
        return "histogram"

    # Trend analysis
    if intent == "TREND":
        return "line"

    # Explicit visualization keywords
    for chart_type, patterns in VISUALIZATION_PATTERNS.items():
        if any(re.search(pattern, query) for pattern in patterns):
            return chart_type

    # Comparison and ranking
    if intent in ("COMPARISON", "TOP_N"):
        return "bar"

    # Dataset overview
    if intent == "DATASET_OVERVIEW":
        return "kpi"

    # Missing values
    if intent == "MISSING_VALUES":
        return "bar"

    # Outlier analysis
    if intent == "OUTLIERS":
        return "box"

    # Default visualization
    numeric_columns = df.select_dtypes(include="number").columns

    if len(numeric_columns) > 0:
        return "histogram"

    return "table"
