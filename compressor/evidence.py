import re

nlp = None

def _get_nlp():
    global nlp
    if nlp is not None:
        return nlp
    try:
        import spacy
        try:
            nlp = spacy.load("en_core_web_sm")
        except Exception:
            import spacy.cli
            spacy.cli.download("en_core_web_sm")
            nlp = spacy.load("en_core_web_sm")
    except Exception as e:
        print(f"Warning: spaCy model could not be loaded in evidence: {e}")
        nlp = False
    return nlp


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


# Phrases that indicate a causal or explanatory claim.
CAUSAL_PHRASES = {
    "caused",
    "cause",
    "causes",
    "contributed",
    "contributed to",
    "contributed significantly",
    "led to",
    "lead to",
    "resulted in",
    "result in",
    "drove",
    "driven by",
    "due to",
    "because of",
    "as a result",
    "responsible for",
    "major contributor",
    "major contributors",
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
    - Causal / explanatory language
    """

    if not sentence or not sentence.strip():
        return 0.0

    score = 0.0

    # ---------------------------------------------------------
    # 1. Numeric evidence
    # ---------------------------------------------------------

    if re.search(
        r"\b\d+(?:,\d{3})*(?:\.\d+)?\b",
        sentence
    ):
        score += 0.15

    # ---------------------------------------------------------
    # 2. Percentage evidence
    # ---------------------------------------------------------

    if re.search(
        r"\b\d+(?:\.\d+)?\s*%",
        sentence
    ):
        score += 0.15

    # ---------------------------------------------------------
    # 3. Date / year evidence
    # ---------------------------------------------------------

    if re.search(
        r"\b(?:19|20)\d{2}\b",
        sentence
    ):
        score += 0.10

    if re.search(
        r"\b\d{1,4}[-/]\d{1,2}[-/]\d{1,4}\b",
        sentence
    ):
        score += 0.10

    # ---------------------------------------------------------
    # 4. Citation evidence
    # ---------------------------------------------------------

    if re.search(
        r"\[\s*\d+(?:\s*,\s*\d+)*\s*\]",
        sentence
    ):
        score += 0.15

    if re.search(
        r"\([A-Z][A-Za-z-]+,\s*(?:19|20)\d{2}\)",
        sentence
    ):
        score += 0.15

    # ---------------------------------------------------------
    # 5. Named entity evidence
    # ---------------------------------------------------------

    spacy_nlp = _get_nlp()
    if spacy_nlp:
        try:
            doc = spacy_nlp(sentence)
            if doc.ents:
                score += 0.15
        except Exception:
            pass
    elif re.search(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', sentence):
        score += 0.10

    # ---------------------------------------------------------
    # 6. Technical term evidence
    # ---------------------------------------------------------

    sentence_lower = sentence.lower()

    technical_matches = 0

    for term in TECHNICAL_TERMS:

        if term in sentence_lower:
            technical_matches += 1

    score += min(
        technical_matches * 0.05,
        0.20
    )

    # ---------------------------------------------------------
    # 7. Causal / explanatory evidence
    # ---------------------------------------------------------

    causal_matches = 0

    for phrase in CAUSAL_PHRASES:

        if phrase in sentence_lower:
            causal_matches += 1

    # Causal language is valuable because it explains
    # WHY something happened, not just WHAT happened.
    score += min(
        causal_matches * 0.15,
        0.30
    )

    # ---------------------------------------------------------
    # Final normalization
    # ---------------------------------------------------------

    return min(
        score,
        1.0
    )