
from src.nlu.preprocessing import preprocess_query
from src.nlu.intent_detection import detect_intent
from src.nlu.entity_extraction import extract_entities
from src.nlu.dataset_context import build_dataset_context


def understand_query(query, df):

    # 1. Normalize query
    clean_query = preprocess_query(query)

    # 2. Understand intent
    intent_result = detect_intent(clean_query)

    # 3. Extract entities
    entities = extract_entities(
        clean_query,
        df
    )

    # 4. Understand dataset
    dataset_context = build_dataset_context(df)

    # 5. Basic validation
    matched_columns = entities["matched_columns"]

    valid_columns = [
        column for column in matched_columns
        if column in dataset_context["columns"]
    ]

    entities["matched_columns"] = valid_columns

    # 6. Initial confidence assessment
    confidence = 0.0

    if intent_result["intent"] != "UNKNOWN":
        confidence += 0.5

    if valid_columns:
        confidence += 0.3

    if entities["aggregation"] is not None:
        confidence += 0.1

    if entities["limit"] is not None:
        confidence += 0.1

    confidence = min(confidence, 1.0)

    # 7. Structured NLU result
    return {
        "original_query": query,

        "normalized_query": clean_query,

        "intent": intent_result["intent"],

        "entities": entities,

        "dataset_context": dataset_context,

        "confidence": round(confidence, 2),

        "needs_clarification": (
            intent_result["intent"] == "UNKNOWN"
        )
    }
