
import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.visualization_planner import select_visualization


@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "Region": ["North", "South", "West", "East"],
        "Product_Name": ["A", "B", "C", "D"],
        "Sales_Amount": [100, 200, 150, 300],
        "Profit": [20, 40, 30, 60],
        "Order_Date": pd.date_range(
            start="2025-01-01",
            periods=4,
            freq="ME"
        ),
    })


@pytest.mark.parametrize(
    "query,intent,expected",
    [
        ("Show sales by region", "COMPARISON", "bar"),
        ("Show sales trend over time", "TREND", "line"),
        ("Show customer age distribution", "DISTRIBUTION", "histogram"),
        ("Show revenue percentage share", "AGGREGATION", "pie"),
        ("Show correlation between sales and profit", "CORRELATION", "heatmap"),
        ("Show top 5 products", "TOP_N", "bar"),
        ("Give me a dataset overview", "DATASET_OVERVIEW", "kpi"),
        ("Find missing values", "MISSING_VALUES", "bar"),
        ("Find outliers in sales", "OUTLIERS", "box"),
    ],
)
def test_select_visualization(sample_df, query, intent, expected):
    entities = {
        "matched_columns": [],
        "aggregation": None,
        "limit": None,
    }

    result = select_visualization(
        query,
        intent,
        entities,
        sample_df
    )

    assert result == expected
