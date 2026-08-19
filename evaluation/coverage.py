import re


STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "has",
    "have",
    "how",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "this",
    "to",
    "was",
    "were",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "with",
}


# ---------------------------------------------------------
# Semantic concept groups
# ---------------------------------------------------------
#
# These allow the coverage guard to understand that
# different words can express the same concept.
#
# Example:
#
#     "What caused the increase?"
#
# can be answered by:
#
#     "Government incentives contributed..."
#
# ---------------------------------------------------------

CONCEPT_GROUPS = {
    "caused": {
        "cause",
        "caused",
        "causes",
        "causing",
        "contributed",
        "contribute",
        "contributes",
        "contributing",
        "contributor",
        "contributors",
        "factor",
        "factors",
        "reason",
        "reasons",
        "led",
        "lead",
        "leads",
        "due",
        "because",
        "resulted",
        "result",
        "resulting",
    },

    "increase": {
        "increase",
        "increased",
        "increases",
        "increasing",
        "growth",
        "grew",
        "grow",
        "expanded",
        "expansion",
        "rise",
        "rose",
        "rising",
    },

    "production": {
        "production",
        "productions",
        "produce",
        "produced",
        "producing",
        "generation",
        "generated",
        "generating",
    },
}


def normalize_word(
    word: str
) -> str:
    """
    Normalize a word for concept matching.
    """

    word = word.lower()

    word = re.sub(
        r"[^a-z0-9%]",
        "",
        word
    )

    return word


def extract_query_concepts(
    query: str
) -> list[str]:
    """
    Extract meaningful concepts from a query.

    Important query words are preserved while common
    stopwords are removed.
    """

    concepts = []

    for word in query.split():

        normalized = normalize_word(
            word
        )

        if not normalized:
            continue

        if normalized in STOPWORDS:
            continue

        if len(normalized) <= 2:
            continue

        if normalized not in concepts:

            concepts.append(
                normalized
            )

    return concepts


def get_concept_group(
    concept: str
) -> set[str]:
    """
    Return the semantic group associated with a concept.

    If no explicit group exists, the concept itself is used.
    """

    normalized = normalize_word(
        concept
    )

    for group in CONCEPT_GROUPS.values():

        if normalized in group:

            return group

    return {
        normalized
    }


def concept_present(
    concept: str,
    context: str
) -> bool:
    """
    Determine whether a query concept is represented
    in the supplied context.

    Matching supports semantic concept groups.

    Example:

        query concept:
            caused

        context:
            government incentives contributed...

        result:
            True
    """

    context_words = {
        normalize_word(word)
        for word in context.split()
    }

    concept_group = get_concept_group(
        concept
    )

    return any(
        word in context_words
        for word in concept_group
    )


def calculate_coverage(
    query: str,
    context: str
) -> dict:
    """
    Calculate query concept coverage.

    Semantic aliases are accepted instead of requiring
    exact word matches.
    """

    concepts = extract_query_concepts(
        query
    )

    if not concepts:

        return {
            "coverage": 1.0,
            "concepts": [],
            "matched_concepts": [],
            "missing_concepts": [],
            "passed": True
        }

    matched = []

    missing = []

    for concept in concepts:

        if concept_present(
            concept,
            context
        ):

            matched.append(
                concept
            )

        else:

            missing.append(
                concept
            )

    coverage = (
        len(matched)
        /
        len(concepts)
    )

    return {
        "coverage": coverage,
        "concepts": concepts,
        "matched_concepts": matched,
        "missing_concepts": missing,
        "passed": coverage >= 0.80
    }


def check_coverage(
    query: str,
    context: str,
    threshold: float = 0.80
) -> dict:
    """
    Check whether the context satisfies the
    required query coverage threshold.
    """

    result = calculate_coverage(
        query=query,
        context=context
    )

    result["threshold"] = threshold

    result["passed"] = (
        result["coverage"]
        >=
        threshold
    )

    return result