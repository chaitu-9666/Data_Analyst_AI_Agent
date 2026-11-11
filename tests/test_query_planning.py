
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pytest

from src.query_planning import select_aggregation


@pytest.mark.parametrize(
    "question, intent, entities, expected",
    [
        (
            "Calculate total revenue",
            "AGGREGATION",
            {"aggregation": "sum"},
            "sum"
        ),
        (
            "What is average profit?",
            "AGGREGATION",
            {"aggregation": "mean"},
            "mean"
        ),
        (
            "Show sales by region",
            "COMPARISON",
            {
                "aggregation": None,
                "numeric_columns": ["Sales_Amount"]
            },
            "sum"
        ),
        (
            "Which products have highest revenue?",
            "TOP_N",
            {"aggregation": None},
            "max"
        ),
        (
            "Show customer age distribution",
            "DISTRIBUTION",
            {"aggregation": None},
            "distribution"
        ),
        (
            "How many units were sold?",
            "AGGREGATION",
            {"aggregation": None},
            "sum"
        ),
        (
            "How many customers are there?",
            "AGGREGATION",
            {"aggregation": None},
            "count"
        ),
    ]
)
def test_select_aggregation(
    question,
    intent,
    entities,
    expected
):

    result = select_aggregation(
        question,
        intent,
        entities
    )

    assert result == expected
