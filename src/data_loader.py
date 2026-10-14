
import pandas as pd
from io import BytesIO


def load_csv(file) -> pd.DataFrame:
    """
    Load and validate a CSV file.

    Args:
        file: An uploaded CSV file or file-like object.

    Returns:
        A pandas DataFrame containing the dataset.

    Raises:
        ValueError: If the file is empty or invalid.
    """
    if file is None:
        raise ValueError("No file was provided.")

    try:
        # Reset the file pointer when supported.
        if hasattr(file, "seek"):
            file.seek(0)

        df = pd.read_csv(file)

    except pd.errors.EmptyDataError as exc:
        raise ValueError("The uploaded CSV file is empty.") from exc
    except UnicodeDecodeError as exc:
        raise ValueError(
            "Unable to read the file encoding. Please upload a UTF-8 CSV."
        ) from exc
    except pd.errors.ParserError as exc:
        raise ValueError(
            "The CSV format is invalid. Please check the file."
        ) from exc

    if df.empty:
        raise ValueError("The CSV contains no data rows.")

    if len(df.columns) == 0:
        raise ValueError("The CSV contains no columns.")

    if df.columns.astype(str).str.strip().eq("").any():
        raise ValueError("The CSV contains a blank column name.")

    return df

