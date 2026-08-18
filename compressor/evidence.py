import re

import spacy


# Load the lightweight spaCy English model.
nlp = spacy.load("en_core_web_sm")


# Technical terms that can indicate useful information
# in technical / RAG contexts.
TECHNICAL_TERMS = {
    "algorithm",
    "accuracy",
    "api",
    "architecture",
    "classification",
    "database",
    "dataset",
    "embedding",
    "evaluation",
    "f1",
    "latency",
    "llm",
    "machine learning",
    "model",
    "precision",
    "recall",
    "retrieval",
    "transformer",
    "training",
    "validation",
    "vector",
}


def evidence_bonus(sentence: str) -> float:
    """
    Calculate an evidence bonus for a sentence.

    Evidence signals include:

    - Numeric values
    - Percentages
    - Dates / years
    - Named entities
    - Citations
    - Technical terms

    Parameters
    ----------
    sentence : str
        Sentence to evaluate.

    Returns
    -------
    float
        Evidence score between 0 and 1.
    """

    if not sentence or not sentence.strip():
        return 0.0

    score = 0.0

    # ---------------------------------------------------------
    # 1. Numeric evidence
    # ---------------------------------------------------------

    # Detect numbers such as:
    # 42
    # 42.5
    # 1,000
    # 1,000.50
    if re.search(
        r"\b\d+(?:,\d{3})*(?:\.\d+)?\b",
        sentence
    ):
        score += 0.15

    # ---------------------------------------------------------
    # 2. Percentage evidence
    # ---------------------------------------------------------

    # Detect:
    # 42%
    # 42.5%
    if re.search(
        r"\b\d+(?:\.\d+)?\s*%",
        sentence
    ):
        score += 0.15

    # ---------------------------------------------------------
    # 3. Date / year evidence
    # ---------------------------------------------------------

    # Detect common four-digit years.
    if re.search(
        r"\b(?:19|20)\d{2}\b",
        sentence
    ):
        score += 0.10

    # Detect common date formats such as:
    # 18/08/2026
    # 18-08-2026
    # 2026-08-18
    if re.search(
        r"\b\d{1,4}[-/]\d{1,2}[-/]\d{1,4}\b",
        sentence
    ):
        score += 0.10

    # ---------------------------------------------------------
    # 4. Citation evidence
    # ---------------------------------------------------------

    # Detect simple citation formats such as:
    # [1]
    # [23]
    # [4, 5]
    if re.search(
        r"\[\s*\d+(?:\s*,\s*\d+)*\s*\]",
        sentence
    ):
        score += 0.15

    # Detect author-year style citations:
    # (Smith, 2024)
    if re.search(
        r"\([A-Z][A-Za-z-]+,\s*(?:19|20)\d{2}\)",
        sentence
    ):
        score += 0.15

    # ---------------------------------------------------------
    # 5. Named entity evidence
    # ---------------------------------------------------------

    doc = nlp(sentence)

    if doc.ents:
        score += 0.15

    # ---------------------------------------------------------
    # 6. Technical term evidence
    # ---------------------------------------------------------

    sentence_lower = sentence.lower()

    technical_matches = 0

    for term in TECHNICAL_TERMS:

        if term in sentence_lower:
            technical_matches += 1

    # Give a small bonus for technical terminology.
    # Cap this contribution so a sentence isn't rewarded
    # excessively simply because it contains many terms.
    score += min(
        technical_matches * 0.05,
        0.20
    )

    # ---------------------------------------------------------
    # Final normalization / cap
    # ---------------------------------------------------------

    return min(score, 1.0)