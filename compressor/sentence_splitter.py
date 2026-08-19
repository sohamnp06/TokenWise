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
        print(f"Warning: spaCy model could not be loaded: {e}. Using regex splitter fallback.")
        nlp = False
    return nlp


def split_sentences(text: str) -> list[str]:
    """
    Split retrieved context into individual sentences.

    Parameters
    ----------
    text : str
        Retrieved context provided by the RAG system.

    Returns
    -------
    list[str]
        A list containing cleaned sentences.
    """
    if not text or not text.strip():
        return []

    spacy_nlp = _get_nlp()
    if spacy_nlp:
        doc = spacy_nlp(text)
        sentences = []
        for sentence in doc.sents:
            cleaned = sentence.text.strip()
            if cleaned:
                sentences.append(cleaned)
        if sentences:
            return sentences

    # Fallback to regex sentence splitting if spaCy is not available
    raw_sents = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in raw_sents if s.strip()]