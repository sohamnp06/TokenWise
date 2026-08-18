from compressor.sentence_splitter import split_sentences


class DocumentChunker:
    """
    Split documents into manageable chunks for retrieval.

    The chunker operates at the sentence level so that
    retrieved chunks remain compatible with TokenWise's
    sentence-level compression pipeline.
    """

    def __init__(
        self,
        chunk_size: int = 3,
        overlap: int = 1
    ):
        """
        Initialize the document chunker.

        Parameters
        ----------
        chunk_size : int
            Number of sentences in each chunk.

        overlap : int
            Number of sentences shared between consecutive
            chunks.
        """

        if chunk_size <= 0:
            raise ValueError(
                "chunk_size must be greater than zero."
            )

        if overlap < 0:
            raise ValueError(
                "overlap cannot be negative."
            )

        if overlap >= chunk_size:
            raise ValueError(
                "overlap must be smaller than chunk_size."
            )

        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk_document(
        self,
        text: str
    ) -> list[str]:
        """
        Split a document into overlapping sentence chunks.

        Parameters
        ----------
        text : str
            Full document text.

        Returns
        -------
        list[str]
            List of text chunks.
        """

        if not text or not text.strip():
            return []

        sentences = split_sentences(
            text
        )

        if not sentences:
            return []

        chunks = []

        step = (
            self.chunk_size
            -
            self.overlap
        )

        for start in range(
            0,
            len(sentences),
            step
        ):

            chunk_sentences = sentences[
                start:
                start + self.chunk_size
            ]

            if not chunk_sentences:
                break

            chunk = " ".join(
                chunk_sentences
            )

            chunks.append(
                chunk
            )

            # Stop once the final chunk has
            # reached the end of the document.
            if (
                start
                +
                self.chunk_size
                >=
                len(sentences)
            ):
                break

        return chunks

    def chunk_documents(
        self,
        documents: list[str]
    ) -> list[str]:
        """
        Chunk multiple documents.

        Parameters
        ----------
        documents : list[str]
            Full documents.

        Returns
        -------
        list[str]
            All generated chunks.
        """

        if not documents:
            return []

        all_chunks = []

        for document in documents:

            chunks = self.chunk_document(
                document
            )

            all_chunks.extend(
                chunks
            )

        return all_chunks