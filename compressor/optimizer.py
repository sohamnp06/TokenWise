class TokenOptimizer:

    def __init__(
        self,
        evidence_weight=0.25,
        redundancy_weight=0.25
    ):

        self.evidence_weight = evidence_weight
        self.redundancy_weight = redundancy_weight

    def calculate_scores(
        self,
        relevance_scores,
        evidence_scores,
        redundancy_scores
    ):

        final_scores = []

        for r, e, d in zip(
            relevance_scores,
            evidence_scores,
            redundancy_scores
        ):

            score = (
                r
                +
                self.evidence_weight * e
                -
                self.redundancy_weight * d
            )

            final_scores.append(
                max(score, 0)
            )

        return final_scores