
import re


INTENT_PATTERNS = {
    "DATASET_OVERVIEW": [
        r"\boverview\b",
        r"\bsummarize dataset\b",
        r"\bdescribe dataset\b",
        r"\bhow big is\b",
        r"\bdescribe the data\b",
    ],

    "MISSING_VALUES": [
        r"\bmissing\b",
        r"\bnull values\b",
        r"\bempty values\b",
        r"\bincomplete\b",
    ],

    "OUTLIERS": [
        r"\boutliers?\b",
        r"\bunusual values\b",
        r"\banomalies\b",
        r"\babnormal values\b",
    ],

    "CORRELATION": [
        r"\bcorrelation\b",
        r"\brelationship between\b",
        r"\bcorrelated\b",
    ],

    "DISTRIBUTION": [
        r"\bdistribution\b",
        r"\bspread of\b",
        r"\bhistogram\b",
    ],

    "TREND": [
        r"\btrend\b",
        r"\bover time\b",
        r"\bmonthly\b",
        r"\bweekly\b",
        r"\byearly\b",
        r"\bper month\b",
    ],

    "FILTER": [
        r"\bwhere\b",
        r"\bfilter\b",
        r"\bgreater than\b",
        r"\bless than\b",
        r"\babove\b",
        r"\bbelow\b",
    ],

    "TOP_N": [
        r"\btop\b",
        r"\bbottom\b",
        r"\bhighest\b",
        r"\blowest\b",
        r"\bbest\b",
        r"\bworst\b",
    ],

    "COMPARISON": [
        r"\bcompare\b",
        r"\bcomparison\b",
        r"\bversus\b",
        r"\bvs\b",
        r"\bby region\b",
        r"\bby category\b",
        r"\bby product\b",
        r"\bby customer\b",
        r"\bgrouped by\b",
    ],

    "AGGREGATION": [
        r"\btotal\b",
        r"\bsum\b",
        r"\baverage\b",
        r"\bmean\b",
        r"\bmedian\b",
        r"\bcount\b",
    ],
}


# Higher-priority intents resolve ambiguous matches.
INTENT_PRIORITY = [
    "MISSING_VALUES",
    "OUTLIERS",
    "CORRELATION",
    "DISTRIBUTION",
    "TREND",
    "FILTER",
    "TOP_N",
    "COMPARISON",
    "AGGREGATION",
    "DATASET_OVERVIEW",
]


def detect_intent(query: str) -> dict:
    """
    Identify the primary analytical intent.

    Returns the detected intent, pattern scores,
    matched patterns and classification confidence.
    """

    query = str(query).lower().strip()

    scores = {}
    matched_patterns = {}

    for intent, patterns in INTENT_PATTERNS.items():

        matches = [
            pattern
            for pattern in patterns
            if re.search(pattern, query)
        ]

        scores[intent] = len(matches)
        matched_patterns[intent] = matches

    matched_intents = [
        intent
        for intent, score in scores.items()
        if score > 0
    ]

    if not matched_intents:
        return {
            "intent": "UNKNOWN",
            "scores": scores,
            "matched_patterns": {},
            "confidence": 0.0,
        }

    # Resolve ties and overlapping intents.
    best_intent = max(
        matched_intents,
        key=lambda intent: (
            scores[intent],
            -INTENT_PRIORITY.index(intent)
        )
    )

    total_matches = sum(scores.values())
    winning_score = scores[best_intent]

    confidence = min(
        0.95,
        0.65 + (0.10 * (winning_score - 1))
    )

    if len(matched_intents) > 1:
        confidence -= 0.05

    return {
        "intent": best_intent,
        "scores": scores,
        "matched_patterns": matched_patterns[best_intent],
        "confidence": round(max(confidence, 0.0), 2),
    }
