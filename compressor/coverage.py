import re


# Common words that generally do not represent
# useful query concepts.
STOPWORDS = {
    "what",
    "when",
    "where",
    "why",
    "how",
    "which",
    "who",
    "whom",
    "whose",
    "the",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "a",
    "an",
    "of",
    "to",
    "in",
    "on",
    "for",
    "and",
    "or",
    "but",
    "with",
    "from",
    "by",
    "about",
    "into",
    "during",
    "between",
    "through",
    "after",
    "before",
    "above",
    "below",
    "this",
    "that",
    "these",
    "those",
    "it",
    "its",
    "they",
    "their",
    "them",
    "can",
    "could",
    "would",
    "should",
    "do",
    "does",
    "did",
}


def extract_query_concepts(
    query: str
) -> set[str]:
    """
    Extract important concepts from a user query.

    This first implementation uses lightweight lexical
    extraction rather than another ML model.

    Parameters
    ----------
    query : str
        User query.

    Returns
    -------
    set[str]
        Important query terms.
    """

    if not query or not query.strip():
        return set()

    words = re.findall(
        r"\b[a-zA-Z0-9]+\b",
        query.lower()
    )

    concepts = set()

    for word in words:

        if (
            len(word) > 2
            and word not in STOPWORDS
        ):
            concepts.add(word)

    return concepts


def calculate_coverage(
    query: str,
    compressed_context: str
) -> float:
    """
    Calculate how much of the query's important concepts
    remain represented in the compressed context.

    Formula:

        Coverage =
            covered concepts / total concepts

    Parameters
    ----------
    query : str
        Original user query.

    compressed_context : str
        Context after compression.

    Returns
    -------
    float
        Coverage between 0 and 1.
    """

    concepts = extract_query_concepts(
        query
    )

    if not concepts:
        return 1.0

    context_lower = (
        compressed_context.lower()
    )

    covered = 0

    for concept in concepts:

        if concept in context_lower:
            covered += 1

    return (
        covered
        /
        len(concepts)
    )


def get_missing_concepts(
    query: str,
    compressed_context: str
) -> list[str]:
    """
    Return query concepts that are missing
    from the compressed context.

    Parameters
    ----------
    query : str
        Original user query.

    compressed_context : str
        Compressed context.

    Returns
    -------
    list[str]
        Missing query concepts.
    """

    concepts = extract_query_concepts(
        query
    )

    if not concepts:
        return []

    context_lower = (
        compressed_context.lower()
    )

    missing = [
        concept
        for concept in concepts
        if concept not in context_lower
    ]

    return sorted(
        missing
    )


def coverage_passed(
    coverage: float,
    threshold: float = 0.80
) -> bool:
    """
    Determine whether the coverage guard passes.

    Parameters
    ----------
    coverage : float
        Calculated coverage between 0 and 1.

    threshold : float
        Minimum acceptable coverage.

    Returns
    -------
    bool
        True if coverage passes.
    """

    return coverage >= threshold