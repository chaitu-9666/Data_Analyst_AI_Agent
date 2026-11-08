
import re
import pandas as pd


# ---------------------------------------------------------
# 1. CONFIGURATION
# ---------------------------------------------------------

REQUIRED_COLUMNS = [
    "Transaction_ID",
    "Customer_ID",
    "Order_Date",
    "Quantity",
    "Unit_Price",
    "Sales_Amount",
]

# These are review thresholds, not automatic corrections.
MIN_CUSTOMER_AGE = 0
MAX_CUSTOMER_AGE = 120

VALID_RETURN_FLAGS = {"yes", "no"}

QUALITY_DIMENSIONS = [
    "Completeness",
    "Validity",
    "Uniqueness",
    "Consistency",
]


# ---------------------------------------------------------
# 2. HELPER FUNCTIONS
# ---------------------------------------------------------

def _normalise_text(series):
    """Convert values to comparable lowercase text."""
    return series.astype("string").str.strip().str.lower()


def _parse_datetime(series):
    """Safely parse date values."""
    return pd.to_datetime(
        series,
        errors="coerce",
        format="mixed"
    )


def _valid_time_mask(series):
    """
    Check common 24-hour and 12-hour time formats.
    Examples:
    14:37
    14:37:21
    02:37:21 PM
    """
    values = series.astype("string").str.strip()

    pattern = (
        r"^(?:"
        r"(?:[01]?\d|2[0-3]):[0-5]\d"
        r"(?::[0-5]\d(?:\.\d+)?)?"
        r"|"
        r"(?:0?[1-9]|1[0-2]):[0-5]\d"
        r"(?::[0-5]\d(?:\.\d+)?)?"
        r"\s?[APap][Mm]"
        r")$"
    )

    return values.str.match(pattern, na=False)


# ---------------------------------------------------------
# 3. MAIN VALIDATION FUNCTION
# ---------------------------------------------------------

def validate_dataset(df):
    """
    Run data quality validation checks.

    Returns:
        dict containing:
            issues: DataFrame of detected issues
            dimensions: DataFrame of dimension scores
            overall_score: Overall quality score
            summary: Summary metrics

    The original DataFrame is never modified.
    """

    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")

    if df.empty:
        raise ValueError("Cannot validate an empty DataFrame.")

    total_rows = len(df)

    issues = []
    checks = []

    # -----------------------------------------------------
    # Internal function to record every validation check
    # -----------------------------------------------------

    def add_check(
        rule_id,
        column,
        issue,
        severity,
        affected_mask,
        description,
        recommendation,
        dimension,
    ):
        affected_mask = pd.Series(
            affected_mask,
            index=df.index
        ).fillna(False).astype(bool)

        affected_count = int(affected_mask.sum())

        checks.append({
            "rule_id": rule_id,
            "dimension": dimension,
            "checked_rows": total_rows,
            "affected_rows": affected_count,
        })

        if affected_count > 0:
            affected_positions = [
                int(position + 1)
                for position, is_affected
                in enumerate(affected_mask.tolist())
                if is_affected
            ]

            issues.append({
                "rule_id": rule_id,
                "column": column,
                "issue": issue,
                "severity": severity,
                "affected_rows": affected_count,
                "description": description,
                "recommendation": recommendation,
                "dimension": dimension,
                "sample_row_numbers": affected_positions[:10],
            })

    # -----------------------------------------------------
    # 4. REQUIRED COLUMN CHECKS
    # -----------------------------------------------------

    for column in REQUIRED_COLUMNS:

        if column not in df.columns:
            add_check(
                rule_id=f"MISSING_COLUMN_{column.upper()}",
                column=column,
                issue="Required column is missing",
                severity="Critical",
                affected_mask=pd.Series(
                    True,
                    index=df.index
                ),
                description=(
                    f"The required column '{column}' "
                    "does not exist in the dataset."
                ),
                recommendation=(
                    "Check the source file and column names."
                ),
                dimension="Completeness",
            )

        else:
            missing_mask = df[column].isna()

            add_check(
                rule_id=f"MISSING_VALUES_{column.upper()}",
                column=column,
                issue="Missing required values",
                severity="High",
                affected_mask=missing_mask,
                description=(
                    f"{column} contains missing values."
                ),
                recommendation=(
                    "Investigate missing records and determine "
                    "whether the values can be recovered."
                ),
                dimension="Completeness",
            )

    # -----------------------------------------------------
    # 5. TRANSACTION ID UNIQUENESS
    # -----------------------------------------------------

    if "Transaction_ID" in df.columns:

        transaction_ids = df["Transaction_ID"].astype("string").str.strip()

        duplicate_mask = (
            transaction_ids.notna()
            & transaction_ids.ne("")
            & transaction_ids.duplicated(keep=False)
        )

        add_check(
            rule_id="UNIQUE_TRANSACTION_ID",
            column="Transaction_ID",
            issue="Duplicate transaction identifiers",
            severity="High",
            affected_mask=duplicate_mask,
            description=(
                "Transaction_ID is defined as a unique "
                "identifier, but repeated values were found."
            ),
            recommendation=(
                "Investigate repeated transaction IDs. "
                "Do not automatically delete these records."
            ),
            dimension="Uniqueness",
        )

    # -----------------------------------------------------
    # 6. CUSTOMER AGE VALIDATION
    # -----------------------------------------------------

    if "Customer_Age" in df.columns:

        age = pd.to_numeric(
            df["Customer_Age"],
            errors="coerce"
        )

        invalid_age_mask = (
            age.notna()
            & (
                (age < MIN_CUSTOMER_AGE)
                | (age > MAX_CUSTOMER_AGE)
            )
        )

        add_check(
            rule_id="CUSTOMER_AGE_RANGE",
            column="Customer_Age",
            issue="Unusual customer age",
            severity="Medium",
            affected_mask=invalid_age_mask,
            description=(
                "Customer age falls outside the configured "
                f"review range of {MIN_CUSTOMER_AGE} to "
                f"{MAX_CUSTOMER_AGE}."
            ),
            recommendation=(
                "Review the original customer information. "
                "The age threshold is configurable."
            ),
            dimension="Validity",
        )

    # -----------------------------------------------------
    # 7. QUANTITY VALIDATION
    # -----------------------------------------------------

    if "Quantity" in df.columns:

        quantity = pd.to_numeric(
            df["Quantity"],
            errors="coerce"
        )

        invalid_quantity_mask = (
            quantity.notna() & (quantity <= 0)
        )

        add_check(
            rule_id="POSITIVE_QUANTITY",
            column="Quantity",
            issue="Zero or negative quantity",
            severity="Medium",
            affected_mask=invalid_quantity_mask,
            description=(
                "Quantity is zero or negative. "
                "This may represent a return or correction."
            ),
            recommendation=(
                "Check order status and return information "
                "before deciding whether the value is invalid."
            ),
            dimension="Validity",
        )

    # -----------------------------------------------------
    # 8. UNIT PRICE VALIDATION
    # -----------------------------------------------------

    if "Unit_Price" in df.columns:

        price = pd.to_numeric(
            df["Unit_Price"],
            errors="coerce"
        )

        invalid_price_mask = (
            price.notna() & (price < 0)
        )

        add_check(
            rule_id="NON_NEGATIVE_UNIT_PRICE",
            column="Unit_Price",
            issue="Negative unit price",
            severity="High",
            affected_mask=invalid_price_mask,
            description=(
                "Unit_Price contains negative values."
            ),
            recommendation=(
                "Review the transaction pricing information."
            ),
            dimension="Validity",
        )

    # -----------------------------------------------------
    # 9. DISCOUNT VALIDATION
    # -----------------------------------------------------

    if "Discount_Percentage" in df.columns:

        discount = pd.to_numeric(
            df["Discount_Percentage"],
            errors="coerce"
        )

        invalid_discount_mask = (
            discount.notna()
            & (
                (discount < 0)
                | (discount > 100)
            )
        )

        add_check(
            rule_id="DISCOUNT_RANGE",
            column="Discount_Percentage",
            issue="Discount outside 0–100%",
            severity="High",
            affected_mask=invalid_discount_mask,
            description=(
                "Discount percentage should be between "
                "0 and 100."
            ),
            recommendation=(
                "Verify the discount value against the "
                "original promotion or transaction."
            ),
            dimension="Validity",
        )

    # -----------------------------------------------------
    # 10. SALES AND COST VALIDATION
    # -----------------------------------------------------

    for column in ["Sales_Amount", "Cost_Amount"]:

        if column in df.columns:

            amount = pd.to_numeric(
                df[column],
                errors="coerce"
            )

            negative_mask = (
                amount.notna() & (amount < 0)
            )

            add_check(
                rule_id=f"NON_NEGATIVE_{column.upper()}",
                column=column,
                issue=f"Negative {column}",
                severity="Medium",
                affected_mask=negative_mask,
                description=(
                    f"{column} contains negative values. "
                    "These require transaction-level investigation."
                ),
                recommendation=(
                    "Check whether the value represents a "
                    "refund, reversal, correction, or data error."
                ),
                dimension="Validity",
            )

    # -----------------------------------------------------
    # 11. DELIVERY DAYS VALIDATION
    # -----------------------------------------------------

    if "Delivery_Days" in df.columns:

        delivery_days = pd.to_numeric(
            df["Delivery_Days"],
            errors="coerce"
        )

        negative_delivery_mask = (
            delivery_days.notna()
            & (delivery_days < 0)
        )

        add_check(
            rule_id="NON_NEGATIVE_DELIVERY_DAYS",
            column="Delivery_Days",
            issue="Negative delivery duration",
            severity="Medium",
            affected_mask=negative_delivery_mask,
            description=(
                "Delivery_Days cannot normally be negative."
            ),
            recommendation=(
                "Check the delivery record and source data."
            ),
            dimension="Validity",
        )

    # -----------------------------------------------------
    # 12. CUSTOMER RATING VALIDATION
    # -----------------------------------------------------

    if "Customer_Rating" in df.columns:

        rating = pd.to_numeric(
            df["Customer_Rating"],
            errors="coerce"
        )

        invalid_rating_mask = (
            rating.notna()
            & (
                (rating < 1)
                | (rating > 5)
            )
        )

        add_check(
            rule_id="CUSTOMER_RATING_RANGE",
            column="Customer_Rating",
            issue="Rating outside 1–5",
            severity="Medium",
            affected_mask=invalid_rating_mask,
            description=(
                "Customer ratings are generally expected "
                "to be between 1 and 5."
            ),
            recommendation=(
                "Check the original customer rating."
            ),
            dimension="Validity",
        )

    # -----------------------------------------------------
    # 13. RETURN FLAG VALIDATION
    # -----------------------------------------------------

    if "Return_Flag" in df.columns:

        return_flag = _normalise_text(
            df["Return_Flag"]
        )

        invalid_return_flag_mask = (
            return_flag.notna()
            & ~return_flag.isin(VALID_RETURN_FLAGS)
        )

        add_check(
            rule_id="VALID_RETURN_FLAG",
            column="Return_Flag",
            issue="Unexpected return flag",
            severity="High",
            affected_mask=invalid_return_flag_mask,
            description=(
                "Return_Flag should contain Yes or No."
            ),
            recommendation=(
                "Check the source values and standardise "
                "the return flag where appropriate."
            ),
            dimension="Validity",
        )

    # -----------------------------------------------------
    # 14. RETURN REASON BUSINESS RULE
    # -----------------------------------------------------

    if (
        "Return_Flag" in df.columns
        and "Return_Reason" in df.columns
    ):

        return_flag = _normalise_text(
            df["Return_Flag"]
        )

        return_reason = (
            df["Return_Reason"]
            .astype("string")
            .str.strip()
        )

        missing_reason = (
            return_reason.isna()
            | return_reason.eq("")
        )

        has_reason = ~missing_reason

        missing_reason_for_return = (
            return_flag.eq("yes")
            & missing_reason
        )

        reason_for_non_return = (
            return_flag.eq("no")
            & has_reason
        )

        add_check(
            rule_id="RETURN_REASON_REQUIRED",
            column="Return_Reason",
            issue="Missing return reason",
            severity="Medium",
            affected_mask=missing_reason_for_return,
            description=(
                "A transaction marked as returned has "
                "no recorded return reason."
            ),
            recommendation=(
                "Check the return record and recover the "
                "reason if it is available."
            ),
            dimension="Consistency",
        )

        add_check(
            rule_id="RETURN_REASON_WITHOUT_RETURN",
            column="Return_Reason",
            issue="Return reason recorded for non-return",
            severity="Low",
            affected_mask=reason_for_non_return,
            description=(
                "A return reason exists even though "
                "Return_Flag is No."
            ),
            recommendation=(
                "Check whether the return flag or reason "
                "was recorded incorrectly."
            ),
            dimension="Consistency",
        )

    # -----------------------------------------------------
    # 15. DATE VALIDATION
    # -----------------------------------------------------

    parsed_dates = None

    if "Order_Date" in df.columns:

        parsed_dates = _parse_datetime(
            df["Order_Date"]
        )

        invalid_date_mask = (
            df["Order_Date"].notna()
            & parsed_dates.isna()
        )

        add_check(
            rule_id="VALID_ORDER_DATE",
            column="Order_Date",
            issue="Invalid order date",
            severity="High",
            affected_mask=invalid_date_mask,
            description=(
                "Some non-empty Order_Date values "
                "could not be parsed as valid dates."
            ),
            recommendation=(
                "Review the original date format and values."
            ),
            dimension="Validity",
        )

    # -----------------------------------------------------
    # 16. TIME VALIDATION
    # -----------------------------------------------------

    if "Order_Time" in df.columns:

        time_valid = _valid_time_mask(
            df["Order_Time"]
        )

        invalid_time_mask = (
            df["Order_Time"].notna()
            & ~time_valid
        )

        add_check(
            rule_id="VALID_ORDER_TIME",
            column="Order_Time",
            issue="Invalid order time",
            severity="Medium",
            affected_mask=invalid_time_mask,
            description=(
                "Some non-empty Order_Time values do not "
                "match the supported time formats."
            ),
            recommendation=(
                "Review the time values and standardise "
                "their format."
            ),
            dimension="Validity",
        )

    # -----------------------------------------------------
    # 17. ORDER YEAR CONSISTENCY
    # -----------------------------------------------------

    if (
        "Order_Year" in df.columns
        and parsed_dates is not None
    ):

        order_year = pd.to_numeric(
            df["Order_Year"],
            errors="coerce"
        )

        valid_comparison = (
            parsed_dates.notna()
            & order_year.notna()
        )

        year_mismatch_mask = (
            valid_comparison
            & (
                order_year
                != parsed_dates.dt.year
            )
        )

        add_check(
            rule_id="ORDER_YEAR_MATCH",
            column="Order_Year",
            issue="Order year does not match order date",
            severity="Medium",
            affected_mask=year_mismatch_mask,
            description=(
                "Order_Year should match the year "
                "derived from Order_Date."
            ),
            recommendation=(
                "Review the year and date values. "
                "Order_Year can be recalculated from Order_Date."
            ),
            dimension="Consistency",
        )

    # -----------------------------------------------------
    # 18. BUILD ISSUES DATAFRAME
    # -----------------------------------------------------

    issue_columns = [
        "rule_id",
        "column",
        "issue",
        "severity",
        "affected_rows",
        "description",
        "recommendation",
        "dimension",
        "sample_row_numbers",
    ]

    issues_df = pd.DataFrame(
        issues,
        columns=issue_columns
    )

    if not issues_df.empty:

        severity_order = {
            "Critical": 0,
            "High": 1,
            "Medium": 2,
            "Low": 3,
        }

        issues_df["_severity_order"] = (
            issues_df["severity"].map(severity_order)
        )

        issues_df = (
            issues_df
            .sort_values(
                ["_severity_order", "affected_rows"],
                ascending=[True, False]
            )
            .drop(columns="_severity_order")
            .reset_index(drop=True)
        )

    # -----------------------------------------------------
    # 19. CALCULATE DIMENSION SCORES
    # -----------------------------------------------------

    checks_df = pd.DataFrame(checks)

    dimension_results = []

    for dimension in QUALITY_DIMENSIONS:

        dimension_checks = checks_df[
            checks_df["dimension"] == dimension
        ]

        if dimension_checks.empty:

            score = None
            applicable_checks = 0

        else:

            pass_rates = (
                1
                - (
                    dimension_checks["affected_rows"]
                    / dimension_checks["checked_rows"]
                )
            )

            score = round(
                float(pass_rates.mean() * 100),
                2
            )

            applicable_checks = len(
                dimension_checks
            )

        dimension_results.append({
            "dimension": dimension,
            "score": score,
            "applicable_checks": applicable_checks,
        })

    dimensions_df = pd.DataFrame(
        dimension_results
    )

    available_scores = dimensions_df[
        "score"
    ].dropna()

    if available_scores.empty:
        overall_score = None
    else:
        overall_score = round(
            float(available_scores.mean()),
            2
        )

    # -----------------------------------------------------
    # 20. SUMMARY
    # -----------------------------------------------------

    if issues_df.empty:
        severity_counts = {}
    else:
        severity_counts = (
            issues_df["severity"]
            .value_counts()
            .to_dict()
        )

    summary = {
        "total_rows": total_rows,
        "total_columns": len(df.columns),
        "total_rules_checked": len(checks_df),
        "total_issues": len(issues_df),
        "total_affected_rows_across_issues": (
            int(issues_df["affected_rows"].sum())
            if not issues_df.empty else 0
        ),
        "overall_score": overall_score,
        "severity_counts": severity_counts,
    }

    return {
        "issues": issues_df,
        "dimensions": dimensions_df,
        "overall_score": overall_score,
        "summary": summary,
    }
