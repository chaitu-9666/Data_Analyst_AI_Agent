
import pandas as pd
import pytest

from src.data_validation import validate_dataset


# ---------------------------------------------------------
# 1. SAMPLE DATASET FOR TESTING
# ---------------------------------------------------------

@pytest.fixture
def sample_sales_data():

    return pd.DataFrame({
        "Transaction_ID": [
            "T1001",
            "T1002",
            "T1003",
            "T1004",
        ],
        "Order_ID": [
            "O100",
            "O100",
            "O101",
            "O102",
        ],
        "Customer_ID": [
            "C001",
            "C002",
            "C003",
            "C004",
        ],
        "Customer_Age": [
            25,
            42,
            35,
            50,
        ],
        "Order_Date": [
            "2024-01-10",
            "2024-02-15",
            "2024-03-20",
            "2024-04-25",
        ],
        "Order_Time": [
            "10:30:00",
            "14:37:21",
            "09:15:00",
            "18:45:30",
        ],
        "Quantity": [
            2,
            1,
            3,
            2,
        ],
        "Unit_Price": [
            100.00,
            200.00,
            150.00,
            300.00,
        ],
        "Discount_Percentage": [
            10,
            5,
            0,
            20,
        ],
        "Sales_Amount": [
            180.00,
            190.00,
            450.00,
            480.00,
        ],
        "Cost_Amount": [
            120.00,
            150.00,
            300.00,
            350.00,
        ],
        "Profit": [
            60.00,
            40.00,
            150.00,
            130.00,
        ],
        "Return_Flag": [
            "No",
            "Yes",
            "No",
            "No",
        ],
        "Return_Reason": [
            None,
            "Damaged",
            None,
            None,
        ],
        "Customer_Rating": [
            5,
            4,
            3,
            5,
        ],
        "Delivery_Days": [
            2,
            3,
            1,
            4,
        ],
        "Order_Year": [
            2024,
            2024,
            2024,
            2024,
        ],
    })


# ---------------------------------------------------------
# 2. CLEAN DATA SHOULD PASS ALL CHECKS
# ---------------------------------------------------------

def test_clean_dataset_has_perfect_score(
    sample_sales_data
):

    result = validate_dataset(
        sample_sales_data
    )

    assert result["overall_score"] == 100.0

    assert result["issues"].empty

    assert (
        result["dimensions"]["score"]
        .dropna()
        .eq(100.0)
        .all()
    )


# ---------------------------------------------------------
# 3. DUPLICATE TRANSACTION ID TEST
# ---------------------------------------------------------

def test_duplicate_transaction_ids_are_detected(
    sample_sales_data
):

    sample_sales_data.loc[1, "Transaction_ID"] = "T1001"

    result = validate_dataset(
        sample_sales_data
    )

    duplicate_issues = result["issues"][
        result["issues"]["rule_id"]
        == "UNIQUE_TRANSACTION_ID"
    ]

    assert not duplicate_issues.empty

    assert (
        duplicate_issues.iloc[0]["affected_rows"]
        == 2
    )


# ---------------------------------------------------------
# 4. ORDER ID CAN REPEAT
# ---------------------------------------------------------

def test_repeated_order_id_is_not_flagged(
    sample_sales_data
):

    result = validate_dataset(
        sample_sales_data
    )

    assert (
        "ORDER_ID" not in
        result["issues"]["rule_id"].to_string()
    )


# ---------------------------------------------------------
# 5. DISCOUNT VALIDATION TEST
# ---------------------------------------------------------

def test_invalid_discount_is_detected(
    sample_sales_data
):

    sample_sales_data.loc[
        0,
        "Discount_Percentage"
    ] = 110

    result = validate_dataset(
        sample_sales_data
    )

    discount_issues = result["issues"][
        result["issues"]["rule_id"]
        == "DISCOUNT_RANGE"
    ]

    assert not discount_issues.empty

    assert (
        discount_issues.iloc[0]["affected_rows"]
        == 1
    )


# ---------------------------------------------------------
# 6. CUSTOMER RATING TEST
# ---------------------------------------------------------

def test_invalid_customer_rating_is_detected(
    sample_sales_data
):

    sample_sales_data.loc[
        2,
        "Customer_Rating"
    ] = 6

    result = validate_dataset(
        sample_sales_data
    )

    rating_issues = result["issues"][
        result["issues"]["rule_id"]
        == "CUSTOMER_RATING_RANGE"
    ]

    assert not rating_issues.empty


# ---------------------------------------------------------
# 7. RETURN REASON BUSINESS RULE TEST
# ---------------------------------------------------------

def test_missing_return_reason_is_detected(
    sample_sales_data
):

    sample_sales_data.loc[
        1,
        "Return_Reason"
    ] = None

    result = validate_dataset(
        sample_sales_data
    )

    return_issues = result["issues"][
        result["issues"]["rule_id"]
        == "RETURN_REASON_REQUIRED"
    ]

    assert not return_issues.empty

    assert (
        return_issues.iloc[0]["affected_rows"]
        == 1
    )


# ---------------------------------------------------------
# 8. RETURN REASON FOR NON-RETURN TEST
# ---------------------------------------------------------

def test_reason_for_non_return_is_detected(
    sample_sales_data
):

    sample_sales_data.loc[
        0,
        "Return_Reason"
    ] = "Damaged"

    result = validate_dataset(
        sample_sales_data
    )

    return_issues = result["issues"][
        result["issues"]["rule_id"]
        == "RETURN_REASON_WITHOUT_RETURN"
    ]

    assert not return_issues.empty


# ---------------------------------------------------------
# 9. ORDER YEAR CONSISTENCY TEST
# ---------------------------------------------------------

def test_order_year_mismatch_is_detected(
    sample_sales_data
):

    sample_sales_data.loc[
        0,
        "Order_Year"
    ] = 2023

    result = validate_dataset(
        sample_sales_data
    )

    year_issues = result["issues"][
        result["issues"]["rule_id"]
        == "ORDER_YEAR_MATCH"
    ]

    assert not year_issues.empty


# ---------------------------------------------------------
# 10. NEGATIVE QUANTITY TEST
# ---------------------------------------------------------

def test_negative_quantity_is_flagged(
    sample_sales_data
):

    sample_sales_data.loc[
        0,
        "Quantity"
    ] = -2

    result = validate_dataset(
        sample_sales_data
    )

    quantity_issues = result["issues"][
        result["issues"]["rule_id"]
        == "POSITIVE_QUANTITY"
    ]

    assert not quantity_issues.empty


# ---------------------------------------------------------
# 11. ORIGINAL DATA MUST NOT BE MODIFIED
# ---------------------------------------------------------

def test_validation_does_not_modify_original_data(
    sample_sales_data
):

    original_data = sample_sales_data.copy(
        deep=True
    )

    validate_dataset(
        sample_sales_data
    )

    pd.testing.assert_frame_equal(
        sample_sales_data,
        original_data
    )


# ---------------------------------------------------------
# 12. MISSING REQUIRED COLUMN TEST
# ---------------------------------------------------------

def test_missing_required_column_is_reported(
    sample_sales_data
):

    sample_sales_data = sample_sales_data.drop(
        columns=["Customer_ID"]
    )

    result = validate_dataset(
        sample_sales_data
    )

    missing_column_issues = result["issues"][
        result["issues"]["rule_id"]
        == "MISSING_COLUMN_CUSTOMER_ID"
    ]

    assert not missing_column_issues.empty

    assert (
        missing_column_issues.iloc[0]["severity"]
        == "Critical"
    )


# ---------------------------------------------------------
# 13. EMPTY DATASET TEST
# ---------------------------------------------------------

def test_empty_dataset_raises_error():

    empty_df = pd.DataFrame()

    with pytest.raises(
        ValueError,
        match="empty DataFrame"
    ):

        validate_dataset(empty_df)


# ---------------------------------------------------------
# 14. INVALID INPUT TYPE TEST
# ---------------------------------------------------------

def test_invalid_input_type_raises_error():

    with pytest.raises(
        TypeError,
        match="pandas DataFrame"
    ):

        validate_dataset(
            ["not", "a", "dataframe"]
        )


# ---------------------------------------------------------
# 15. QUALITY DIMENSIONS TEST
# ---------------------------------------------------------

def test_all_quality_dimensions_are_returned(
    sample_sales_data
):

    result = validate_dataset(
        sample_sales_data
    )

    expected_dimensions = {
        "Completeness",
        "Validity",
        "Uniqueness",
        "Consistency",
    }

    actual_dimensions = set(
        result["dimensions"]["dimension"]
    )

    assert actual_dimensions == expected_dimensions


# ---------------------------------------------------------
# 16. OVERALL SCORE IS WITHIN VALID RANGE
# ---------------------------------------------------------

def test_overall_score_is_between_zero_and_100(
    sample_sales_data
):

    sample_sales_data.loc[
        0,
        "Discount_Percentage"
    ] = 150

    result = validate_dataset(
        sample_sales_data
    )

    assert 0 <= result["overall_score"] <= 100
