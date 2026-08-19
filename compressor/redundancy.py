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
        """

        if not sentences:
            return np.empty(
                (0, 0)
            )

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
        """

        if not sentences:
            return np.empty(
                (0, 0)
            )

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
        Calculate redundancy penalty for every sentence.

        Each sentence is compared against previous sentences.

        The penalty is the maximum semantic similarity
        with any previous sentence.
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

            if i == 0:

                penalties.append(
                    0.0
                )

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

    def score(
        self,
        sentences: list[str]
    ) -> list[float]:
        """
        Return redundancy scores for sentences.

        This is the public scoring interface used by
        the TokenWise compression pipeline.

        Higher score means the sentence is more redundant.
        """

        return self.redundancy_penalties(
            sentences
        )

    def find_redundant_pairs(
        self,
        sentences: list[str],
        threshold: float = 0.80
    ) -> list[dict]:
        """
        Find pairs of highly similar sentences.
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

                    redundant_pairs.append(
                        {
                            "sentence_a": sentences[i],
                            "sentence_b": sentences[j],
                            "similarity": score
                        }
                    )

        return redundant_pairs