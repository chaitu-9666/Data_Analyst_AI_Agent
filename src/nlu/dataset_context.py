
import pandas as pd


def build_dataset_context(df: pd.DataFrame) -> dict:

    numeric_columns = df.select_dtypes(
        include="number"
    ).columns.tolist()

    categorical_columns = df.select_dtypes(
        include=["object", "category", "string"]
    ).columns.tolist()

    datetime_columns = []

    for column in df.columns:

        if pd.api.types.is_datetime64_any_dtype(df[column]):
            datetime_columns.append(column)

    return {
        "columns": df.columns.tolist(),

        "numeric_columns": numeric_columns,

        "categorical_columns": categorical_columns,

        "datetime_columns": datetime_columns,

        "row_count": len(df),

        "column_count": len(df.columns),

        "column_details": {
            str(column): {
                "dtype": str(df[column].dtype),
                "unique_values": int(df[column].nunique()),
                "missing_values": int(df[column].isna().sum()),
                "sample_values": (
                    df[column]
                    .dropna()
                    .astype(str)
                    .unique()[:5]
                    .tolist()
                )
            }
            for column in df.columns
        }
    }
