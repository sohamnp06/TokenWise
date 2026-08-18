import numpy as np
from sentence_transformers import CrossEncoder


class RelevanceScorer:
    """
    Scores the relevance of retrieved sentences against a user query
    using a Cross-Encoder model.

    The Cross-Encoder receives:
        [query, sentence]

    and produces a relevance score for each pair.
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    ):
        """
        Initialize the Cross-Encoder model.

        Parameters
        ----------
        model_name : str
            Hugging Face / Sentence Transformers Cross-Encoder model.
        """

        self.model_name = model_name

        print(
            f"Loading Cross-Encoder model: {self.model_name}"
        )

        self.model = CrossEncoder(
            self.model_name
        )

        print("Cross-Encoder model loaded successfully.")

    def score(
        self,
        query: str,
        sentences: list[str]
    ) -> list[float]:
        """
        Calculate normalized relevance scores.

        Parameters
        ----------
        query : str
            User's search/query text.

        sentences : list[str]
            Sentences extracted from retrieved context.

        Returns
        -------
        list[float]
            Normalized relevance scores between 0 and 1.
        """

        if not query or not query.strip():
            raise ValueError(
                "Query cannot be empty."
            )

        if not sentences:
            return []

        # Create query-sentence pairs.
        pairs = [
            [query, sentence]
            for sentence in sentences
        ]

        # Generate raw Cross-Encoder scores.
        raw_scores = np.asarray(
            self.model.predict(pairs),
            dtype=float
        )

        # Normalize scores to the range [0, 1].
        normalized_scores = self._normalize(
            raw_scores
        )

        return normalized_scores.tolist()

    @staticmethod
    def _normalize(
        scores: np.ndarray
    ) -> np.ndarray:
        """
        Normalize raw scores using min-max normalization.

        Formula:

            normalized = (x - min) / (max - min)

        If every sentence receives the same score,
        every sentence receives a neutral relevance score of 1.0.
        """

        if len(scores) == 0:
            return scores

        min_score = scores.min()
        max_score = scores.max()

        # Avoid division by zero when all scores are identical.
        if max_score == min_score:
            return np.ones_like(scores)

        normalized = (
            (scores - min_score)
            /
            (max_score - min_score)
        )

        return normalized