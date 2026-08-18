from compressor.sentence_splitter import (
    split_sentences
)

from compressor.relevance import (
    RelevanceScorer
)

from compressor.evidence import (
    evidence_bonus
)

from compressor.redundancy import (
    RedundancyDetector
)

from compressor.optimizer import (
    TokenOptimizer
)

from compressor.coverage import (
    calculate_coverage,
    get_missing_concepts,
    coverage_passed
)

from evaluation.metrics import (
    count_tokens,
    compression_ratio,
    tokens_saved
)


class TokenDiet:
    """
    Main TokenWise context compression pipeline.

    Pipeline:

        Query + Retrieved Context
                    ↓
             Sentence Splitting
                    ↓
             Relevance Scoring
                    ↓
              Evidence Bonus
                    ↓
            Redundancy Detection
                    ↓
             Final Sentence Score
                    ↓
              Token Value
                    ↓
          Budget-Constrained Selection
                    ↓
              Coverage Guard
                    ↓
             Compressed Context
    """

    def __init__(
        self,
        evidence_weight: float = 0.25,
        redundancy_weight: float = 0.25,
        coverage_threshold: float = 0.80,
        relevance_model: str = (
            "cross-encoder/ms-marco-MiniLM-L-6-v2"
        ),
        redundancy_model: str = (
            "all-MiniLM-L6-v2"
        )
    ):
        """
        Initialize all TokenWise components.

        Parameters
        ----------
        evidence_weight : float
            Weight applied to evidence bonus.

        redundancy_weight : float
            Weight applied to redundancy penalty.

        coverage_threshold : float
            Minimum acceptable query coverage.

        relevance_model : str
            Cross-Encoder model used for relevance.

        redundancy_model : str
            Sentence Transformer model used for redundancy.
        """

        self.coverage_threshold = (
            coverage_threshold
        )

        # -----------------------------------------------------
        # Relevance scorer
        # -----------------------------------------------------

        self.relevance_scorer = (
            RelevanceScorer(
                model_name=relevance_model
            )
        )

        # -----------------------------------------------------
        # Redundancy detector
        # -----------------------------------------------------

        self.redundancy_detector = (
            RedundancyDetector(
                model_name=redundancy_model
            )
        )

        # -----------------------------------------------------
        # Token optimizer
        # -----------------------------------------------------

        self.optimizer = (
            TokenOptimizer(
                evidence_weight=evidence_weight,
                redundancy_weight=redundancy_weight
            )
        )

    def compress(
        self,
        query: str,
        context: str,
        token_budget: int = 800
    ) -> dict:
        """
        Compress retrieved context according to a token budget.

        Parameters
        ----------
        query : str
            User query.

        context : str
            Retrieved context from the RAG system.

        token_budget : int
            Maximum number of tokens allowed for the
            compressed context.

        Returns
        -------
        dict
            Complete compression result containing:

            - compressed_context
            - kept
            - removed
            - coverage
            - missing_concepts
            - original_tokens
            - compressed_tokens
            - tokens_saved
            - compression_ratio
            - total_sentences
            - kept_sentences
            - removed_sentences
            - coverage_guard_passed
            - coverage_guard_triggered
        """

        # -----------------------------------------------------
        # Input validation
        # -----------------------------------------------------

        if not query or not query.strip():
            raise ValueError(
                "Query cannot be empty."
            )

        if not context or not context.strip():
            raise ValueError(
                "Context cannot be empty."
            )

        if token_budget <= 0:
            raise ValueError(
                "Token budget must be greater than zero."
            )

        # -----------------------------------------------------
        # 1. Sentence splitting
        # -----------------------------------------------------

        sentences = split_sentences(
            context
        )

        if not sentences:

            return {
                "compressed_context": "",
                "kept": [],
                "removed": [],
                "coverage": 0.0,
                "missing_concepts": [],
                "original_tokens": count_tokens(
                    context
                ),
                "compressed_tokens": 0,
                "tokens_saved": count_tokens(
                    context
                ),
                "compression_ratio": 100.0,
                "total_sentences": 0,
                "kept_sentences": 0,
                "removed_sentences": 0,
                "coverage_guard_passed": False,
                "coverage_guard_triggered": False
            }

        # -----------------------------------------------------
        # 2. Relevance scoring
        # -----------------------------------------------------

        relevance_scores = (
            self.relevance_scorer.score(
                query=query,
                sentences=sentences
            )
        )

        # -----------------------------------------------------
        # 3. Evidence scoring
        # -----------------------------------------------------

        evidence_scores = [
            evidence_bonus(sentence)
            for sentence in sentences
        ]

        # -----------------------------------------------------
        # 4. Redundancy scoring
        # -----------------------------------------------------

        redundancy_scores = (
            self.redundancy_detector
            .redundancy_penalties(
                sentences
            )
        )

        # -----------------------------------------------------
        # 5. Final sentence score
        # -----------------------------------------------------

        final_scores = (
            self.optimizer.calculate_scores(
                relevance_scores=relevance_scores,
                evidence_scores=evidence_scores,
                redundancy_scores=redundancy_scores
            )
        )

        # -----------------------------------------------------
        # 6. Token value calculation
        # -----------------------------------------------------

        candidates = (
            self.optimizer.calculate_token_values(
                sentences=sentences,
                scores=final_scores
            )
        )

        # Add scoring information to every candidate.
        for index, candidate in enumerate(
            candidates
        ):

            candidate["index"] = index

            candidate["relevance"] = (
                relevance_scores[index]
            )

            candidate["evidence"] = (
                evidence_scores[index]
            )

            candidate["redundancy"] = (
                redundancy_scores[index]
            )

            candidate["decision"] = (
                "PENDING"
            )

        # -----------------------------------------------------
        # 7. Budget-constrained selection
        # -----------------------------------------------------

        selected = self.optimizer.select(
            candidates=candidates,
            token_budget=token_budget
        )

        selected_indices = {
            candidate["index"]
            for candidate in selected
        }

        # Mark selected sentences.
        for candidate in candidates:

            if candidate["index"] in selected_indices:
                candidate["decision"] = "KEEP"

            else:
                candidate["decision"] = "REMOVE"

        # -----------------------------------------------------
        # 8. Build initial compressed context
        # -----------------------------------------------------

        # Keep the original document order rather than
        # token-value ranking order.
        #
        # This is important because the LLM should receive
        # coherent context instead of randomly reordered
        # sentences.

        selected_by_index = sorted(
            selected,
            key=lambda item: item["index"]
        )

        compressed_context = " ".join(
            candidate["sentence"]
            for candidate in selected_by_index
        )

        # -----------------------------------------------------
        # 9. Coverage Guard
        # -----------------------------------------------------

        coverage = calculate_coverage(
            query=query,
            compressed_context=compressed_context
        )

        guard_triggered = False

        # -----------------------------------------------------
        # 10. Restore best excluded sentence if coverage fails
        # -----------------------------------------------------

        if not coverage_passed(
            coverage,
            threshold=self.coverage_threshold
        ):

            guard_triggered = True

            excluded = [
                candidate
                for candidate in candidates
                if candidate["index"]
                not in selected_indices
            ]

            # Prioritize excluded sentences by final score.
            excluded.sort(
                key=lambda item: item["score"],
                reverse=True
            )

            # Restore sentences one at a time until
            # coverage reaches the required threshold.
            #
            # We also respect the token budget whenever possible.
            for candidate in excluded:

                candidate_cost = (
                    candidate["token_cost"]
                )

                current_tokens = count_tokens(
                    compressed_context
                )

                if (
                    current_tokens
                    +
                    candidate_cost
                    >
                    token_budget
                ):
                    continue

                selected.append(
                    candidate
                )

                selected_indices.add(
                    candidate["index"]
                )

                candidate["decision"] = (
                    "RESTORED"
                )

                selected_by_index = sorted(
                    selected,
                    key=lambda item: item["index"]
                )

                compressed_context = " ".join(
                    item["sentence"]
                    for item in selected_by_index
                )

                coverage = calculate_coverage(
                    query=query,
                    compressed_context=compressed_context
                )

                if coverage_passed(
                    coverage,
                    threshold=self.coverage_threshold
                ):
                    break

        # -----------------------------------------------------
        # 11. Final coverage check
        # -----------------------------------------------------

        coverage_passed_final = (
            coverage_passed(
                coverage,
                threshold=self.coverage_threshold
            )
        )

        # -----------------------------------------------------
        # 12. Final kept / removed lists
        # -----------------------------------------------------

        kept = [
            candidate
            for candidate in candidates
            if candidate["index"]
            in selected_indices
        ]

        removed = [
            candidate
            for candidate in candidates
            if candidate["index"]
            not in selected_indices
        ]

        # -----------------------------------------------------
        # 13. Missing query concepts
        # -----------------------------------------------------

        missing_concepts = (
            get_missing_concepts(
                query=query,
                compressed_context=compressed_context
            )
        )

        # -----------------------------------------------------
        # 14. Token metrics
        # -----------------------------------------------------

        original_tokens = count_tokens(
            context
        )

        compressed_tokens = count_tokens(
            compressed_context
        )

        saved_tokens = tokens_saved(
            original_tokens=original_tokens,
            compressed_tokens=compressed_tokens
        )

        saved_ratio = compression_ratio(
            original_tokens=original_tokens,
            compressed_tokens=compressed_tokens
        )

        # -----------------------------------------------------
        # 15. Final result
        # -----------------------------------------------------

        return {
            "compressed_context":
                compressed_context,

            "kept":
                kept,

            "removed":
                removed,

            "coverage":
                coverage,

            "missing_concepts":
                missing_concepts,

            "original_tokens":
                original_tokens,

            "compressed_tokens":
                compressed_tokens,

            "tokens_saved":
                saved_tokens,

            "compression_ratio":
                saved_ratio,

            "total_sentences":
                len(sentences),

            "kept_sentences":
                len(kept),

            "removed_sentences":
                len(removed),

            "coverage_guard_passed":
                coverage_passed_final,

            "coverage_guard_triggered":
                guard_triggered
        }