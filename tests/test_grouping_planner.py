
import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.grouping_planner import build_grouping_plan


@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "Region": ["North", "South", "North", "West"],
        "Product_Name": ["A", "B", "C", "A"],
        "Sales_Amount": [100, 200, 150, 300],
        "Customer_Age": [25, 30, 35, 40],
    })


@pytest.mark.parametrize(
    "query,intent,entities,expected_group,expected_sort,expected_limit,expected_aggregation",
    [
        (
            "Show sales by region",
            "COMPARISON",
            {
                "matched_columns": ["Region", "Sales_Amount"],
                "aggregation": "sum",
                "limit": None
            },
            "Region", None, None, "sum",
        ),
        (
            "Which products have highest revenue?",
            "TOP_N",
            {
                "matched_columns": ["Product_Name", "Sales_Amount"],
                "aggregation": "max",
                "limit": None
            },
            "Product_Name", "descending", None, "sum",
        ),
        (
            "Show top 5 products",
            "TOP_N",
            {
                "matched_columns": ["Product_Name"],
                "aggregation": None,
                "limit": 5
            },
            "Product_Name", "descending", 5, None,
        ),
        (
            "Show bottom 3 regions",
            "TOP_N",
            {
                "matched_columns": ["Region"],
                "aggregation": None,
                "limit": 3
            },
            "Region", "ascending", 3, None,
        ),
    ],
)
def test_build_grouping_plan(
    sample_df,
    query,
    intent,
    entities,
    expected_group,
    expected_sort,
    expected_limit,
    expected_aggregation
):
    plan = build_grouping_plan(query, intent, entities, sample_df)

    assert plan["group_by"] == expected_group
    assert plan["sort"] == expected_sort
    assert plan["limit"] == expected_limit
    assert plan["aggregation"] == expected_aggregation
