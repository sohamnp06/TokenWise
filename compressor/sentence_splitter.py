import spacy


# Load the English spaCy pipeline.
# This model was installed using:
# python -m spacy download en_core_web_sm
nlp = spacy.load("en_core_web_sm")


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

    doc = nlp(text)

    sentences = []

    for sentence in doc.sents:

        cleaned_sentence = sentence.text.strip()

        if cleaned_sentence:
            sentences.append(cleaned_sentence)

    return sentences