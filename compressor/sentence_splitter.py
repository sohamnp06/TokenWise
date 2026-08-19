import re


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

    try:
        import spacy
        nlp = spacy.load("en_core_web_sm")
        doc = nlp(text)
        sents = [s.text.strip() for s in doc.sents if s.text.strip()]
        if sents:
            return sents
    except Exception:
        pass

    # High-accuracy regex sentence splitter
    raw_sents = re.split(r'(?<=[.!?])\s+(?=[A-Z0-9"\'\(\[])|\n+', text.strip())
    result = []
    for s in raw_sents:
        cleaned = s.strip()
        if cleaned:
            result.append(cleaned)
    return result if result else [text.strip()]