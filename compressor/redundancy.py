import numpy as np

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


class RedundancyDetector:

    def __init__(
        self,
        model_name="all-MiniLM-L6-v2"
    ):

        self.model = SentenceTransformer(model_name)

    def similarity_matrix(self, sentences):

        embeddings = self.model.encode(
            sentences,
            normalize_embeddings=True
        )

        return cosine_similarity(embeddings)

    def redundancy_penalties(self, sentences):

        similarity = self.similarity_matrix(sentences)

        penalties = []

        for i in range(len(sentences)):

            previous_similarities = [
                similarity[i][j]
                for j in range(i)
            ]

            if not previous_similarities:
                penalties.append(0.0)

            else:
                penalties.append(
                    max(previous_similarities)
                )

        return penalties