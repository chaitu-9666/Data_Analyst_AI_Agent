import pandas as pd
import pytest

from src.nlu.preprocessing import preprocess_query
from src.nlu.intent_detection import detect_intent
from src.nlu.entity_extraction import extract_entities
from src.nlu.dataset_context import build_dataset_context
from src.nlu.query_understanding import understand_query


# ==========================================================
# 1. SAMPLE DATASET
# ==========================================================

@pytest.fixture
def sample_df():

    return pd.DataFrame({
        "Product_Name": ["A", "B", "C", "A"],
        "Revenue": [100, 200, 300, 150],
        "Profit": [20, 40, 60, 30],
        "Region": ["North", "South", "North", "West"]
    })


# ==========================================================
# 2. PREPROCESSING TESTS
# ==========================================================

def test_preprocessing_strips_and_normalizes_whitespace():

    result = preprocess_query(
        "  TOTAL   SALES  "
    )

    assert result == "total sales"


def test_preprocessing_rejects_non_string_input():

    with pytest.raises(TypeError):

        preprocess_query(None)


# ==========================================================
# 3. INTENT DETECTION TESTS
# ==========================================================

def test_intent_detection_identifies_top_n():

    result = detect_intent(
        "Show top 10 products"
    )

    assert result["intent"] == "TOP_N"


def test_intent_detection_returns_unknown():

    result = detect_intent(
        "hello there"
    )

    assert result["intent"] == "UNKNOWN"


# ==========================================================
# 4. ENTITY EXTRACTION TESTS
# ==========================================================

def test_entity_extraction_finds_columns_and_limit(sample_df):

    result = extract_entities(
        "Show top 5 products by Revenue",
        sample_df
    )

    assert result["limit"] == 5

    assert "Revenue" in result["matched_columns"]


def test_entity_extraction_detects_sum(sample_df):

    result = extract_entities(
        "Calculate total Revenue",
        sample_df
    )

    assert result["aggregation"] == "sum"

    assert "Revenue" in result["matched_columns"]


def test_entity_extraction_detects_bottom_sort(sample_df):

    result = extract_entities(
        "Show bottom 3 Revenue values",
        sample_df
    )

    assert result["limit"] == 3

    assert result["sort"] == "ascending"


# ==========================================================
# 5. DATASET CONTEXT TESTS
# ==========================================================

def test_dataset_context_reports_schema(sample_df):

    context = build_dataset_context(sample_df)

    assert context["row_count"] == 4

    assert context["column_count"] == 4

    assert "Revenue" in context["numeric_columns"]

    assert "Region" in context["categorical_columns"]

    assert (
        context["column_details"]["Revenue"]["missing_values"]
        == 0
    )


# ==========================================================
# 6. COMPLETE NLU TESTS
# ==========================================================

def test_query_understanding_returns_expected_structure(sample_df):

    result = understand_query(
        "Calculate total Revenue",
        sample_df
    )

    assert result["intent"] == "AGGREGATION"

    assert result["entities"]["aggregation"] == "sum"

    assert "Revenue" in result["entities"]["matched_columns"]

    assert 0 <= result["confidence"] <= 1

    assert isinstance(
        result["needs_clarification"],
        bool
    )


def test_query_understanding_unknown_intent_requests_clarification(sample_df):

    result = understand_query(
        "hello there",
        sample_df
    )

    assert result["intent"] == "UNKNOWN"

    assert result["needs_clarification"] is True