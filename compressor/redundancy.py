import numpy as np

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


class RedundancyDetector:
    """
    Detect semantic redundancy between sentences.

    Sentence embeddings are generated using a lightweight
    Sentence Transformer model, and cosine similarity is used
    to determine how semantically similar sentences are.
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2"
    ):
        """
        Initialize the sentence embedding model.

        Parameters
        ----------
        model_name : str
            Sentence Transformer model used for semantic similarity.
        """

        self.model_name = model_name

        print(
            f"Loading redundancy model: {self.model_name}"
        )

        self.model = SentenceTransformer(
            self.model_name
        )

        print(
            "Redundancy model loaded successfully."
        )

    def encode(
        self,
        sentences: list[str]
    ) -> np.ndarray:
        """
        Convert sentences into normalized embeddings.

        Parameters
        ----------
        sentences : list[str]
            Sentences to encode.

        Returns
        -------
        np.ndarray
            Normalized sentence embeddings.
        """

        if not sentences:
            return np.empty((0, 0))

        embeddings = self.model.encode(
            sentences,
            normalize_embeddings=True
        )

        return np.asarray(
            embeddings
        )

    def similarity_matrix(
        self,
        sentences: list[str]
    ) -> np.ndarray:
        """
        Calculate pairwise semantic similarity.

        Parameters
        ----------
        sentences : list[str]
            Sentences to compare.

        Returns
        -------
        np.ndarray
            Pairwise cosine similarity matrix.
        """

        if not sentences:
            return np.empty((0, 0))

        embeddings = self.encode(
            sentences
        )

        return cosine_similarity(
            embeddings
        )

    def redundancy_penalties(
        self,
        sentences: list[str]
    ) -> list[float]:
        """
        Calculate a redundancy penalty for every sentence.

        A sentence receives a higher penalty when it is
        semantically similar to an earlier sentence.

        For sentence i:

            penalty(i) =
                max(similarity(i, previous sentences))

        The first sentence receives a penalty of 0 because
        there are no previous sentences to compare against.

        Parameters
        ----------
        sentences : list[str]
            Sentences to evaluate.

        Returns
        -------
        list[float]
            Redundancy penalties between 0 and 1.
        """

        if not sentences:
            return []

        similarity = self.similarity_matrix(
            sentences
        )

        penalties = []

        for i in range(
            len(sentences)
        ):

            # First sentence has nothing before it,
            # so it cannot be redundant with a previous sentence.
            if i == 0:
                penalties.append(0.0)
                continue

            previous_similarities = similarity[
                i,
                :i
            ]

            max_similarity = float(
                np.max(
                    previous_similarities
                )
            )

            # Cosine similarity can theoretically produce
            # tiny floating-point values outside [0, 1].
            max_similarity = max(
                0.0,
                min(
                    1.0,
                    max_similarity
                )
            )

            penalties.append(
                max_similarity
            )

        return penalties

    def find_redundant_pairs(
        self,
        sentences: list[str],
        threshold: float = 0.80
    ) -> list[dict]:
        """
        Find pairs of highly similar sentences.

        Parameters
        ----------
        sentences : list[str]
            Sentences to compare.

        threshold : float
            Similarity threshold above which two sentences
            are considered potentially redundant.

        Returns
        -------
        list[dict]
            List of redundant sentence pairs.
        """

        if not sentences:
            return []

        similarity = self.similarity_matrix(
            sentences
        )

        redundant_pairs = []

        for i in range(
            len(sentences)
        ):

            for j in range(
                i + 1,
                len(sentences)
            ):

                score = float(
                    similarity[i][j]
                )

                if score >= threshold:

                    redundant_pairs.append({
                        "sentence_a": sentences[i],
                        "sentence_b": sentences[j],
                        "similarity": score
                    })

        return redundant_pairs