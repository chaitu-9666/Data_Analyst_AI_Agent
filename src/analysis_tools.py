
"""Safe, deterministic analysis operations for natural-language questions."""

from dataclasses import dataclass
import re

import pandas as pd


@dataclass(frozen=True)
class AnalysisResult:
    """A textual answer and optional tabular or chart-ready result."""

    operation: str
    answer: str
    data: pd.DataFrame | None = None
    chart: bool = False


def analyze_question(df: pd.DataFrame, question: str) -> AnalysisResult:
    """Answer a supported question using a fixed set of Pandas operations.

    This intentionally does not interpret or execute Python code from the
    question. Unsupported or underspecified questions return guidance instead.
    """
    question = question.strip()
    if not question:
        return _unsupported_result()

    text = question.casefold()

    if re.search(r"\b(correlation|correlate|correlated)\b", text):
        return _strongest_correlation(df)

    if re.search(r"\b(chart|plot|graph|visuali[sz]e)\b", text) and re.search(
        r"\bby\b", text
    ):
        return _grouped_chart(df, text)

    if re.search(r"\b(top|highest|largest)\b", text):
        return _top_records(df, text)

    if re.search(r"\b(missing|null|na|nan)\b", text):
        return _missing_values(df)

    if re.search(r"\b(average|avg|mean)\b", text):
        return _average(df, text)

    if re.search(r"\b(rows?|records?|observations?|columns?|fields?|shape)\b", text):
        return _dimensions(df, text)

    return _unsupported_result()


def _dimensions(df: pd.DataFrame, question: str) -> AnalysisResult:
    row_count, column_count = df.shape
    asks_rows = bool(re.search(r"\b(rows?|records?|observations?|shape)\b", question))
    asks_columns = bool(re.search(r"\b(columns?|fields?|shape)\b", question))

    if asks_rows and asks_columns:
        answer = f"The dataset has {row_count} rows and {column_count} columns."
    elif asks_rows:
        answer = f"The dataset has {row_count} rows."
    elif asks_columns:
        answer = f"The dataset has {column_count} columns."
    else:
        answer = f"The dataset has {row_count} rows and {column_count} columns."

    return AnalysisResult("dimensions", answer)


def _missing_values(df: pd.DataFrame) -> AnalysisResult:
    counts = df.isna().sum()
    missing = pd.DataFrame(
        {
            "Column": counts.index[counts.gt(0)],
            "Missing Values": counts[counts.gt(0)].values,
            "Missing Percentage": (
                df.isna().mean()[counts.gt(0)].values * 100
            ).round(2),
        }
    )
    if missing.empty:
        answer = "No columns have missing values."
    else:
        answer = f"{len(missing)} column(s) have missing values."
    return AnalysisResult("missing_values", answer, missing)


def _top_records(df: pd.DataFrame, question: str) -> AnalysisResult:
    count_match = re.search(r"\btop\s+(\d+)\b", question)
    count = int(count_match.group(1)) if count_match else 10
    column = _column_after(df, question, r"\b(?:by|on|in)\b")
    if column is None:
        return _need_column("top_records", "Name the column to rank the records by.")

    records = df.sort_values(column, ascending=False, na_position="last").head(count)
    return AnalysisResult(
        "top_records",
        f"Showing the top {len(records)} record(s) by {column}.",
        records,
    )


def _average(df: pd.DataFrame, question: str) -> AnalysisResult:
    match = re.search(r"\b(?:average|avg|mean)\b", question)
    target = question[match.end():] if match else ""
    target = re.split(r"\b(?:by|for|in)\b", target, maxsplit=1)[0]
    column = _mentioned_column(df, target)
    if column is None:
        return _need_column("average", "Name a numeric column to average.")
    if not pd.api.types.is_numeric_dtype(df[column]):
        return _need_column("average", f"{column} is not a numeric column.")
    value = df[column].mean()
    if pd.isna(value):
        return _need_column("average", f"{column} has no numeric values to average.")
    return AnalysisResult("average", f"The average {column} is {value:g}.")


def _strongest_correlation(df: pd.DataFrame) -> AnalysisResult:
    numeric = df.select_dtypes(include="number")
    if len(numeric.columns) < 2:
        return _need_column(
            "strongest_correlation",
            "At least two numeric columns are required to calculate correlations.",
        )

    correlations = numeric.corr()
    upper = correlations.where(
        pd.DataFrame(
            [[i < j for j in range(len(correlations.columns))]
             for i in range(len(correlations.columns))],
            index=correlations.index,
            columns=correlations.columns,
        )
    ).stack()
    if upper.empty:
        return _need_column(
            "strongest_correlation",
            "There are no valid correlations between numeric columns.",
        )

    first, second = upper.abs().idxmax()
    value = upper.loc[(first, second)]
    pair = pd.DataFrame(
        [{"Column 1": first, "Column 2": second, "Correlation": value}]
    )
    return AnalysisResult(
        "strongest_correlation",
        f"The strongest correlation is between {first} and {second} "
        f"({value:.3f}).",
        pair,
    )


def _grouped_chart(df: pd.DataFrame, question: str) -> AnalysisResult:
    by_match = re.search(r"\bby\s+(.+)", question)
    prefix = question[:by_match.start()] if by_match else question
    value_column = _mentioned_column(df, prefix)
    group_column = _mentioned_column(df, by_match.group(1)) if by_match else None

    if value_column is None or group_column is None:
        return _need_column(
            "grouped_chart",
            "Name a numeric value column and a category column, for example "
            "'chart sales by category'.",
        )
    if not pd.api.types.is_numeric_dtype(df[value_column]):
        return _need_column(
            "grouped_chart", f"{value_column} must be numeric to create this chart."
        )

    grouped = (
        df.groupby(group_column, dropna=False)[value_column]
        .sum()
        .reset_index()
    )
    return AnalysisResult(
        "grouped_chart",
        f"Showing total {value_column} by {group_column}.",
        grouped,
        chart=True,
    )


def _column_after(
    df: pd.DataFrame, text: str, marker_pattern: str
) -> str | None:
    match = re.search(marker_pattern, text)
    if match is None:
        return None
    return _mentioned_column(df, text[match.end():])


def _mentioned_column(df: pd.DataFrame, text: str) -> str | None:
    normalized_text = _normalize(text)
    columns = sorted((str(column) for column in df.columns), key=len, reverse=True)

    for column in columns:
        normalized_column = _normalize(column)
        if not normalized_column:
            continue
        if re.search(
            rf"(?<!\w){re.escape(normalized_column)}(?!\w)",
            normalized_text,
        ):
            return column

    # Accept common singular/plural phrasing such as "category" for "Categories".
    text_words = {_singularize(word) for word in normalized_text.split()}
    matches = [
        column
        for column in columns
        if _normalize(column)
        and set(_singularize(word) for word in _normalize(column).split())
        <= text_words
    ]
    if len(matches) == 1:
        return matches[0]
    if matches:
        return None

    stop_words = {
        "a", "an", "and", "by", "for", "in", "of", "on", "the", "to",
        "average", "avg", "mean", "what", "is", "are", "show", "me",
        "create", "chart", "plot", "graph", "showing",
    }
    query_words = text_words - stop_words
    partial_matches = [
        column
        for column in columns
        if query_words
        & {
            _singularize(word)
            for word in _normalize(column).split()
        }
    ]
    return partial_matches[0] if len(partial_matches) == 1 else None


def _normalize(text: str) -> str:
    return " ".join(re.findall(r"\w+", text.casefold()))


def _singularize(word: str) -> str:
    if len(word) > 3 and word.endswith("ies"):
        return word[:-3] + "y"
    if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    return word


def _need_column(operation: str, message: str) -> AnalysisResult:
    return AnalysisResult(operation, message)


def _unsupported_result() -> AnalysisResult:
    return AnalysisResult(
        "unsupported",
        "I can't answer that yet. Try asking about dataset size, missing values, "
        "top records, a numeric average, correlations, or a chart by category.",
    )
