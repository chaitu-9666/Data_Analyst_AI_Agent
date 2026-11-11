
import re
import pandas as pd


COLUMN_ALIASES = {
    "Sales_Amount": [
        "revenue",
        "sales",
        "total sales",
        "sales amount",
        "total revenue",
        "turnover"
    ],
    "Cost_Amount": [
        "cost",
        "cost amount",
        "total cost",
        "expenses"
    ],
    "Profit": [
        "profit",
        "earnings",
        "net profit"
    ],
    "Quantity": [
        "quantity",
        "units",
        "items sold"
    ],
    "Customer_Age": [
        "customer age",
        "age"
    ],
    "Order_Date": [
        "order date",
        "date",
        "sales date"
    ],
    "Product_Name": [
        "product",
        "products",
        "product name",
        "products name",
        "item",
        "items"
    ],
    "Product_Category": [
        "category",
        "product category",
        "categories"
    ],
    "Region": [
        "region",
        "sales region",
        "regions"
    ],
    "Customer_Name": [
        "customer",
        "customers",
        "customer name"
    ]
}


def normalize_text(text):
    text = str(text).lower().strip()
    text = re.sub(r"[_\s]+", " ", text)
    return re.sub(r"\s+", " ", text)


def extract_entities(query: str, df) -> dict:

    normalized_query = normalize_text(query)
    matched_columns = []

    # Match actual dataset column names.
    for column in df.columns:
        normalized_column = normalize_text(column)

        pattern = (
            r"(?<!\w)"
            + re.escape(normalized_column)
            + r"(?!\w)"
        )

        if re.search(pattern, normalized_query):
            matched_columns.append(column)

    # Match business-language aliases.
    for column, aliases in COLUMN_ALIASES.items():

        if column not in df.columns:
            continue

        for alias in aliases:
            normalized_alias = normalize_text(alias)

            pattern = (
                r"(?<!\w)"
                + re.escape(normalized_alias)
                + r"(?!\w)"
            )

            if re.search(pattern, normalized_query):
                if column not in matched_columns:
                    matched_columns.append(column)
                break

    # Detect aggregation.
    aggregation_patterns = {
        "sum": ["total", "sum", "overall"],
        "mean": ["average", "mean", "avg"],
        "median": ["median"],
        "count": ["count", "number of"],
        "min": ["minimum", "smallest", "lowest"],
        "max": ["maximum", "largest", "highest"]
    }

    aggregation = None

    for operation, keywords in aggregation_patterns.items():
        if any(
            re.search(
                r"\b" + re.escape(word) + r"\b",
                normalized_query
            )
            for word in keywords
        ):
            aggregation = operation
            break

    # Detect top/bottom limit.
    limit_match = re.search(
        r"\b(?:top|bottom)\s+(\d+)\b",
        normalized_query
    )

    limit = int(limit_match.group(1)) if limit_match else None

    # Detect sorting.
    if re.search(
        r"\b(bottom|lowest|smallest|worst|ascending)\b",
        normalized_query
    ):
        sort = "ascending"
    else:
        sort = "descending"

    return {
        "matched_columns": matched_columns,
        "aggregation": aggregation,
        "limit": limit,
        "sort": sort
    }
