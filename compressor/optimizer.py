from evaluation.metrics import count_tokens


class TokenOptimizer:
    """
    Optimize retrieved sentences according to their
    information value per token.

    TokenWise first calculates a final sentence score:

        Score =
            Relevance
            + Evidence Weight * Evidence
            - Redundancy Weight * Redundancy

    It then calculates:

        Token Value =
            Score / Token Cost

    Finally, it selects the highest-value sentences
    while respecting a fixed token budget.
    """

    def __init__(
        self,
        evidence_weight: float = 0.25,
        redundancy_weight: float = 0.25
    ):
        """
        Initialize the optimizer.

        Parameters
        ----------
        evidence_weight : float
            Weight applied to the evidence bonus.

        redundancy_weight : float
            Weight applied to the redundancy penalty.
        """

        self.evidence_weight = (
            evidence_weight
        )

        self.redundancy_weight = (
            redundancy_weight
        )

    def calculate_scores(
        self,
        relevance_scores: list[float],
        evidence_scores: list[float],
        redundancy_scores: list[float]
    ) -> list[float]:
        """
        Calculate the final score for every sentence.

        Formula:

            S(s) =
                R(s)
                + λE(s)
                - μD(s)

        Parameters
        ----------
        relevance_scores : list[float]
            Cross-Encoder relevance scores.

        evidence_scores : list[float]
            Evidence bonuses.

        redundancy_scores : list[float]
            Redundancy penalties.

        Returns
        -------
        list[float]
            Final sentence scores.
        """

        if not (
            len(relevance_scores)
            ==
            len(evidence_scores)
            ==
            len(redundancy_scores)
        ):
            raise ValueError(
                "All score lists must have the same length."
            )

        final_scores = []

        for (
            relevance,
            evidence,
            redundancy
        ) in zip(
            relevance_scores,
            evidence_scores,
            redundancy_scores
        ):

            score = (
                relevance
                +
                (
                    self.evidence_weight
                    *
                    evidence
                )
                -
                (
                    self.redundancy_weight
                    *
                    redundancy
                )
            )

            # Prevent negative sentence scores.
            score = max(
                0.0,
                score
            )

            final_scores.append(
                score
            )

        return final_scores

    def calculate_token_values(
        self,
        sentences: list[str],
        scores: list[float]
    ) -> list[dict]:
        """
        Calculate token cost and token value for each sentence.

        Formula:

            Token Value =
                Sentence Score / Token Cost

        Parameters
        ----------
        sentences : list[str]
            Candidate sentences.

        scores : list[float]
            Final sentence scores.

        Returns
        -------
        list[dict]
            Candidate information including:

            sentence
            score
            token_cost
            token_value
        """

        if len(sentences) != len(scores):
            raise ValueError(
                "Sentences and scores must have the same length."
            )

        candidates = []

        for sentence, score in zip(
            sentences,
            scores
        ):

            token_cost = count_tokens(
                sentence
            )

            # Prevent division by zero.
            token_cost = max(
                1,
                token_cost
            )

            token_value = (
                score
                /
                token_cost
            )

            candidates.append({
                "sentence": sentence,
                "score": score,
                "token_cost": token_cost,
                "token_value": token_value
            })

        return candidates

    def select(
        self,
        candidates: list[dict],
        token_budget: int
    ) -> list[dict]:
        """
        Select high-value sentences under a token budget.

        The current implementation uses a greedy
        value-per-token strategy:

            1. Sort candidates by token value.
            2. Select the highest-value candidate.
            3. Continue until the token budget is exhausted.

        Parameters
        ----------
        candidates : list[dict]
            Candidate sentences produced by
            calculate_token_values().

        token_budget : int
            Maximum number of tokens allowed.

        Returns
        -------
        list[dict]
            Selected candidates.
        """

        if token_budget <= 0:
            return []

        # Sort by information value per token.
        ranked_candidates = sorted(
            candidates,
            key=lambda item: item["token_value"],
            reverse=True
        )

        selected = []

        total_tokens = 0

        for candidate in ranked_candidates:

            token_cost = candidate[
                "token_cost"
            ]

            # Do not exceed the token budget.
            if (
                total_tokens
                +
                token_cost
                <=
                token_budget
            ):

                selected.append(
                    candidate
                )

                total_tokens += (
                    token_cost
                )

        return selected