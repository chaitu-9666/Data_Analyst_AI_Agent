
import re
import unicodedata


def preprocess_query(query: str) -> str:
    """
    Normalize a user's analytical question.
    """

    if not isinstance(query, str):
        raise TypeError("Query must be a string.")

    query = unicodedata.normalize("NFKC", query)
    query = query.strip().lower()

    query = re.sub(r"\s+", " ", query)

    return query
