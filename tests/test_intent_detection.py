
import sys
from pathlib import Path

# Add project root to Python import path
PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pytest

from src.nlu.intent_detection import detect_intent


@pytest.mark.parametrize(
    "question, expected_intent",
    [
        ("Calculate total revenue", "AGGREGATION"),
        ("What is average profit?", "AGGREGATION"),
        ("Show sales by region", "COMPARISON"),
        ("Which products generate highest sales?", "TOP_N"),
        ("Show customer age distribution", "DISTRIBUTION"),
        ("Show monthly sales trends", "TREND"),
        ("Find missing values", "MISSING_VALUES"),
        ("Compare sales versus profit", "COMPARISON"),
        ("Show the dataset overview", "DATASET_OVERVIEW"),
        ("Find unusual values", "OUTLIERS"),
        ("Show records above 100", "FILTER"),
        ("Hello there", "UNKNOWN"),
    ]
)
def test_intent_detection(question, expected_intent):
    result = detect_intent(question)
    assert result["intent"] == expected_intent


def test_confidence_is_valid():
    result = detect_intent("Calculate total revenue")
    assert 0 <= result["confidence"] <= 1


def test_unknown_intent():
    result = detect_intent("Hello there")

    assert result["intent"] == "UNKNOWN"
    assert result["confidence"] == 0.0
