
import pandas as pd

from src.profiling import (
    profile_dataset,
    get_column_profile,
    detect_outliers_iqr,
    detect_outliers_advanced,
    calculate_data_quality_score,
    generate_data_quality_insights
)


def sample_data():

    return pd.DataFrame({
        "sales": [100, 200, 300, 400, 500, 10000],
        "quantity": [1, 2, 3, 4, 5, 100],
        "category": [
            "A", "B", "A", "B", "A", "C"
        ],
        "rating": [5, 4, None, 3, 5, 2]
    })


def test_profile_dataset():

    df = sample_data()

    result = profile_dataset(df)

    assert result["total_rows"] == 6
    assert result["total_columns"] == 4
    assert result["missing_cells"] == 1
    assert result["duplicate_rows"] == 0


def test_column_profile():

    df = sample_data()

    result = get_column_profile(df)

    rating = result[
        result["column"] == "rating"
    ].iloc[0]

    assert rating["missing_count"] == 1


def test_outlier_detection():

    df = sample_data()

    result = detect_outliers_iqr(df)

    sales_result = result[
        result["column"] == "sales"
    ].iloc[0]

    assert sales_result["outlier_count"] >= 1

def test_advanced_outlier_detection():

    df = sample_data()

    result = detect_outliers_advanced(
        df,
        z_threshold=2.0
    )

    # Check that all three methods are available
    assert "IQR" in result["method"].values
    assert "Z-Score" in result["method"].values
    assert "Modified Z-Score" in result["method"].values

    # Check that extreme sales value is detected
    sales_iqr = result[
        (result["column"] == "sales") &
        (result["method"] == "IQR")
    ]

    assert not sales_iqr.empty
    assert sales_iqr.iloc[0]["outlier_count"] >= 1


def test_advanced_outlier_detection_excludes_ids():

    df = pd.DataFrame({
        "customer_id": [1, 2, 3, 4, 10000],
        "sales": [100, 200, 300, 400, 10000]
    })

    result = detect_outliers_advanced(df)

    assert "customer_id" not in result["column"].values


def test_advanced_outlier_detection_invalid_threshold():

    df = sample_data()

    try:
        detect_outliers_advanced(df, z_threshold=0)

        assert False, "Expected ValueError"

    except ValueError:
        assert True


def test_data_quality_score():

    df = sample_data()

    result = calculate_data_quality_score(df)

    assert 0 <= result["overall_score"] <= 100

    assert result["overall_grade"] in [
        "Excellent", "Good", "Fair", "Poor"
    ]

    assert "completeness" in result["component_scores"]

    assert "uniqueness" in result["component_scores"]

    assert "outlier_health" in result["component_scores"]


def test_column_quality_scores():

    df = sample_data()

    result = calculate_data_quality_score(df)

    column_scores = result["column_scores"]

    assert len(column_scores) == len(df.columns)

    sales = column_scores[
        column_scores["column"] == "sales"
    ].iloc[0]

    assert sales["quality_score"] >= 0

    assert sales["quality_score"] <= 100



def test_data_quality_detects_missing_and_duplicates():

    df = pd.DataFrame({
        "name": ["A", "B", "B", "D", "E"],
        "city": ["London", "Paris", "Paris", "London", None]
    })

    result = calculate_data_quality_score(df)

    assert result["missing_cells"] == 1

    assert result["duplicate_rows"] == 1

    assert result["overall_score"] < 100

def test_data_quality_without_numeric_columns():

    df = pd.DataFrame({
        "name": ["A", "B", "C"],
        "city": ["London", "Paris", "Rome"]
    })

    result = calculate_data_quality_score(df)

    assert "outlier_health" not in result["component_scores"]

    assert result["overall_score"] == 100



# ==========================================================
# STEP 6: AUTOMATED DATA QUALITY INSIGHTS TESTS
# ==========================================================

def test_generate_data_quality_insights():

    df = sample_data()

    result = generate_data_quality_insights(df)

    assert 0 <= result["overall_score"] <= 100

    assert result["overall_grade"] in [
        "Excellent", "Good", "Fair", "Poor"
    ]

    assert isinstance(result["summary"], str)

    assert result["total_insights"] == len(result["insights"])

    assert len(result["insights"]) > 0

    assert len(result["recommendations"]) > 0


def test_insights_detect_missing_values():

    df = pd.DataFrame({
        "name": ["A", "B", "C"],
        "age": [20, None, 30]
    })

    result = generate_data_quality_insights(df)

    categories = [
        item["category"] for item in result["insights"]
    ]

    assert "Missing Values" in categories


def test_insights_detect_duplicate_rows():

    df = pd.DataFrame({
        "name": ["A", "B", "B", "D"],
        "city": ["London", "Paris", "Paris", "Rome"]
    })

    result = generate_data_quality_insights(df)

    categories = [
        item["category"] for item in result["insights"]
    ]

    assert "Duplicate Records" in categories


def test_insights_detect_outliers():

    df = sample_data()

    result = generate_data_quality_insights(df)

    categories = [
        item["category"] for item in result["insights"]
    ]

    assert "Potential Outliers" in categories


def test_insights_for_clean_dataset():

    df = pd.DataFrame({
        "name": ["A", "B", "C"],
        "city": ["London", "Paris", "Rome"]
    })

    result = generate_data_quality_insights(df)

    assert result["total_insights"] == 0

    assert len(result["strengths"]) > 0

    assert result["overall_grade"] == "Excellent"


def test_insights_invalid_threshold():

    df = sample_data()

    try:
        generate_data_quality_insights(
            df,
            low_quality_threshold=150
        )

        assert False, "Expected ValueError"

    except ValueError:
        assert True
