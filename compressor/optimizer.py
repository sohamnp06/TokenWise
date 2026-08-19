from evaluation.metrics import count_tokens


class TokenOptimizer:
    """
    Optimize sentences according to information value per token.

    Final score:

        Score =
            Relevance
            + Evidence Weight * Evidence
            - Redundancy Weight * Redundancy

    Selection considers:

        - Relevance
        - Evidence
        - Redundancy
        - Token efficiency
        - Already selected content

    The optimizer protects useful evidence while aggressively
    removing highly redundant sentences.
    """

    def __init__(
        self,
        evidence_weight: float = 0.45,
        redundancy_weight: float = 0.25,
        evidence_priority: float = 0.15,
        redundancy_threshold: float = 0.80
    ):
        self.evidence_weight = evidence_weight
        self.redundancy_weight = redundancy_weight
        self.evidence_priority = evidence_priority
        self.redundancy_threshold = redundancy_threshold

    # =========================================================
    # SCORE CALCULATION
    # =========================================================

    def calculate_scores(
        self,
        relevance_scores: list[float],
        evidence_scores: list[float],
        redundancy_scores: list[float]
    ) -> list[float]:

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
                self.evidence_weight * evidence
                -
                self.redundancy_weight * redundancy
            )

            # Protect strong evidence.
            if evidence >= 0.50:

                evidence_floor = (
                    evidence * 0.30
                )

                score = max(
                    score,
                    evidence_floor
                )

            score = max(
                0.0,
                score
            )

            final_scores.append(
                score
            )

        return final_scores

    # =========================================================
    # TOKEN VALUE
    # =========================================================

    def calculate_token_values(
        self,
        sentences: list[str],
        scores: list[float],
        evidence_scores: list[float] | None = None
    ) -> list[dict]:

        if len(sentences) != len(scores):
            raise ValueError(
                "Sentences and scores must have the same length."
            )

        if evidence_scores is None:

            evidence_scores = [
                0.0
                for _ in sentences
            ]

        if len(evidence_scores) != len(sentences):
            raise ValueError(
                "Evidence scores must have the same length "
                "as sentences."
            )

        candidates = []

        for (
            sentence,
            score,
            evidence
        ) in zip(
            sentences,
            scores,
            evidence_scores
        ):

            token_cost = max(
                1,
                count_tokens(sentence)
            )

            token_value = (
                score
                /
                token_cost
            )

            candidates.append(
                {
                    "sentence": sentence,
                    "score": score,
                    "evidence": evidence,
                    "token_cost": token_cost,
                    "token_value": token_value
                }
            )

        return candidates

    # =========================================================
    # SELECTION
    # =========================================================

    def select(
        self,
        candidates: list[dict],
        token_budget: int
    ) -> list[dict]:
        """
        Select high-value sentences under a token budget.

        Selection strategy:

        Phase 1:
            Preserve important evidence.

        Phase 2:
            Select high-value sentences by token efficiency.

        Redundancy guard:
            Candidates with extremely high redundancy are skipped
            when they do not provide sufficiently unique evidence.

        This prevents cases such as:

            "Solar generation increased by 42%..."

        and

            "Solar energy generation experienced a 42% increase..."

        from both consuming the context budget.
        """

        if token_budget <= 0:
            return []

        remaining = list(candidates)

        selected = []

        total_tokens = 0

        # -----------------------------------------------------
        # Phase 1: Evidence-aware candidates
        # -----------------------------------------------------

        evidence_candidates = [
            candidate
            for candidate in remaining
            if candidate.get(
                "evidence",
                0.0
            ) >= 0.25
        ]

        evidence_candidates.sort(
            key=lambda candidate: (
                candidate.get(
                    "evidence",
                    0.0
                )
                *
                (
                    1.0
                    -
                    candidate.get(
                        "redundancy",
                        0.0
                    )
                )
            ),
            reverse=True
        )

        for candidate in evidence_candidates:

            token_cost = candidate[
                "token_cost"
            ]

            redundancy = candidate.get(
                "redundancy",
                0.0
            )

            evidence = candidate.get(
                "evidence",
                0.0
            )

            # -------------------------------------------------
            # Skip extremely redundant evidence.
            #
            # Exception:
            # Very strong evidence can still survive.
            # -------------------------------------------------

            if (
                redundancy
                >=
                self.redundancy_threshold
                and
                evidence
                <
                0.65
            ):
                remaining.remove(
                    candidate
                )
                continue

            if (
                total_tokens
                +
                token_cost
                <=
                token_budget
            ):
                selected_candidate = dict(candidate)
                ev_mult = 1.0 + (self.evidence_priority * evidence)
                red_mult = 1.0 - (self.redundancy_weight * redundancy)
                selected_candidate["selection_value"] = candidate.get("token_value", 0.0) * ev_mult * red_mult

                selected.append(
                    selected_candidate
                )

                total_tokens += (
                    token_cost
                )

                remaining.remove(
                    candidate
                )

        # -----------------------------------------------------
        # Phase 2: General candidate selection
        # -----------------------------------------------------

        ranked_candidates = []

        for candidate in remaining:

            token_value = candidate[
                "token_value"
            ]

            evidence = candidate.get(
                "evidence",
                0.0
            )

            redundancy = candidate.get(
                "redundancy",
                0.0
            )

            # -------------------------------------------------
            # Evidence multiplier
            # -------------------------------------------------

            evidence_multiplier = (
                1.0
                +
                (
                    self.evidence_priority
                    *
                    evidence
                )
            )

            # -------------------------------------------------
            # Redundancy multiplier
            # -------------------------------------------------

            redundancy_multiplier = (
                1.0
                -
                (
                    self.redundancy_weight
                    *
                    redundancy
                )
            )

            selection_value = (
                token_value
                *
                evidence_multiplier
                *
                redundancy_multiplier
            )

            ranked_candidates.append(
                (
                    selection_value,
                    candidate
                )
            )

        ranked_candidates.sort(
            key=lambda item: item[0],
            reverse=True
        )

        # -----------------------------------------------------
        # Select remaining candidates
        # -----------------------------------------------------

        for (
            selection_value,
            candidate
        ) in ranked_candidates:

            token_cost = candidate[
                "token_cost"
            ]

            redundancy = candidate.get(
                "redundancy",
                0.0
            )

            evidence = candidate.get(
                "evidence",
                0.0
            )

            # Skip extremely redundant candidates unless
            # they contain very strong evidence.

            if (
                redundancy
                >=
                self.redundancy_threshold
                and
                evidence
                <
                0.65
            ):
                continue

            if (
                total_tokens
                +
                token_cost
                <=
                token_budget
            ):

                selected_candidate = dict(
                    candidate
                )

                selected_candidate[
                    "selection_value"
                ] = selection_value

                selected.append(
                    selected_candidate
                )

                total_tokens += (
                    token_cost
                )

        return selected