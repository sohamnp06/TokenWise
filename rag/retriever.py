from sentence_transformers import SentenceTransformer
import faiss
import numpy as np


class FAISSRetriever:
    """
    Lightweight dense retriever using Sentence Transformers
    and FAISS.

    Pipeline:

        Documents
            ↓
        Embeddings
            ↓
        FAISS Index
            ↓
        Query Embedding
            ↓
        Top-K Similar Chunks
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2"
    ):
        """
        Initialize the embedding model.

        Parameters
        ----------
        model_name : str
            Sentence Transformer model used for embeddings.
        """

        self.model_name = model_name

        print(
            f"Loading retrieval model: "
            f"{self.model_name}"
        )

        self.model = SentenceTransformer(
            self.model_name
        )

        # all-MiniLM-L6-v2 produces 384-dimensional
        # sentence embeddings.
        self.embedding_dimension = (
            self.model.get_sentence_embedding_dimension()
        )

        # FAISS inner-product index.
        #
        # Because embeddings are normalized before insertion,
        # inner product is equivalent to cosine similarity.
        self.index = faiss.IndexFlatIP(
            self.embedding_dimension
        )

        # Store original chunks separately because FAISS
        # stores vectors, not the actual text.
        self.documents = []

        print(
            "Retrieval model loaded successfully."
        )

        print(
            f"Embedding dimension: "
            f"{self.embedding_dimension}"
        )

    def _embed(
        self,
        texts: list[str]
    ) -> np.ndarray:
        """
        Convert text into normalized embeddings.

        Parameters
        ----------
        texts : list[str]
            Texts to embed.

        Returns
        -------
        np.ndarray
            Normalized float32 embeddings.
        """

        if not texts:
            return np.empty(
                (0, self.embedding_dimension),
                dtype=np.float32
            )

        embeddings = self.model.encode(
            texts,
            normalize_embeddings=True
        )

        return np.asarray(
            embeddings,
            dtype=np.float32
        )

    def add_documents(
        self,
        documents: list[str]
    ) -> None:
        """
        Add document chunks to the FAISS index.

        Parameters
        ----------
        documents : list[str]
            Document chunks to index.
        """

        if not documents:
            raise ValueError(
                "Documents cannot be empty."
            )

        cleaned_documents = [
            document.strip()
            for document in documents
            if document
            and document.strip()
        ]

        if not cleaned_documents:
            raise ValueError(
                "No valid documents were provided."
            )

        embeddings = self._embed(
            cleaned_documents
        )

        self.index.add(
            embeddings
        )

        self.documents.extend(
            cleaned_documents
        )

    def search(
        self,
        query: str,
        top_k: int = 5
    ) -> list[dict]:
        """
        Retrieve the most relevant document chunks.

        Parameters
        ----------
        query : str
            User query.

        top_k : int
            Number of chunks to retrieve.

        Returns
        -------
        list[dict]
            Retrieved chunks containing:

                text
                score
                rank
        """

        if not query or not query.strip():
            raise ValueError(
                "Query cannot be empty."
            )

        if self.index.ntotal == 0:
            raise RuntimeError(
                "No documents have been indexed."
            )

        if top_k <= 0:
            raise ValueError(
                "top_k must be greater than zero."
            )

        # We cannot retrieve more documents than
        # are actually present.
        top_k = min(
            top_k,
            self.index.ntotal
        )

        query_embedding = self._embed(
            [query]
        )

        scores, indices = self.index.search(
            query_embedding,
            top_k
        )

        results = []

        for rank, (
            score,
            index
        ) in enumerate(
            zip(
                scores[0],
                indices[0]
            ),
            start=1
        ):

            # FAISS may return -1 for an unavailable result.
            if index < 0:
                continue

            results.append({
                "text": self.documents[index],
                "score": float(score),
                "rank": rank
            })

        return results

    def clear(self) -> None:
        """
        Clear the FAISS index and stored documents.
        """

        self.index.reset()

        self.documents = []

    @property
    def document_count(self) -> int:
        """
        Return the number of indexed documents.
        """

        return len(
            self.documents
        )