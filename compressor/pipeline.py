from compressor.sentence_splitter import split_sentences
from compressor.relevance import RelevanceScorer
from compressor.evidence import evidence_bonus
from compressor.redundancy import RedundancyDetector
from compressor.optimizer import TokenOptimizer
from evaluation.metrics import count_tokens
from evaluation.coverage import (
    check_coverage,
    concept_present
)


class TokenDiet:
    """
    TokenWise context compression pipeline.

    Pipeline:

        Sentence splitting
            ↓
        Relevance scoring
            ↓
        Evidence scoring
            ↓
        Redundancy detection
            ↓
        Token optimization
            ↓
        Semantic coverage guard
            ↓
        Compressed context
    """

    def __init__(
        self,
        evidence_weight: float = 0.45,
        redundancy_weight: float = 0.25,
        coverage_threshold: float = 0.80
    ):

        self.coverage_threshold = (
            coverage_threshold
        )

        self.relevance_scorer = (
            RelevanceScorer()
        )

        self.redundancy_detector = (
            RedundancyDetector()
        )

        self.optimizer = TokenOptimizer(
            evidence_weight=evidence_weight,
            redundancy_weight=redundancy_weight
        )

    def _build_candidates(
        self,
        query: str,
        sentences: list[str]
    ) -> list[dict]:

        relevance_scores = (
            self.relevance_scorer.score(
                query,
                sentences
            )
        )

        evidence_scores = [
            evidence_bonus(sentence)
            for sentence in sentences
        ]

        redundancy_scores = (
            self.redundancy_detector.score(
                sentences
            )
        )

        final_scores = (
            self.optimizer.calculate_scores(
                relevance_scores,
                evidence_scores,
                redundancy_scores
            )
        )

        candidates = (
            self.optimizer.calculate_token_values(
                sentences,
                final_scores
            )
        )

        for index, candidate in enumerate(
            candidates
        ):

            candidate["relevance"] = (
                relevance_scores[index]
            )

            candidate["evidence"] = (
                evidence_scores[index]
            )

            candidate["redundancy"] = (
                redundancy_scores[index]
            )

            candidate["score"] = (
                final_scores[index]
            )

        return candidates

    def _coverage_guard(
        self,
        query: str,
        selected: list[dict],
        removed: list[dict]
    ) -> tuple[list[dict], list[dict], dict]:

        compressed_context = " ".join(
            candidate["sentence"]
            for candidate in selected
        )

        coverage_result = check_coverage(
            query=query,
            context=compressed_context,
            threshold=self.coverage_threshold
        )

        # Coverage already satisfies the threshold.
        if coverage_result["passed"]:

            return (
                selected,
                removed,
                coverage_result
            )

        missing_concepts = set(
            coverage_result[
                "missing_concepts"
            ]
        )

        recovery_candidates = []

        # Search removed sentences for semantic
        # equivalents of missing concepts.
        for candidate in removed:

            sentence = candidate[
                "sentence"
            ]

            matches = 0

            for concept in missing_concepts:

                if concept_present(
                    concept,
                    sentence
                ):

                    matches += 1

            if matches > 0:

                recovery_candidates.append(
                    (
                        matches,
                        candidate
                    )
                )

        # Prefer candidates that cover the most
        # missing concepts, followed by evidence,
        # relevance, and final score.
        recovery_candidates.sort(
            key=lambda item: (
                item[0],
                item[1].get(
                    "evidence",
                    0.0
                ),
                item[1].get(
                    "relevance",
                    0.0
                ),
                item[1].get(
                    "score",
                    0.0
                )
            ),
            reverse=True
        )

        for (
            _,
            candidate
        ) in recovery_candidates:

            if candidate not in removed:
                continue

            selected.append(
                candidate
            )

            removed.remove(
                candidate
            )

            compressed_context = " ".join(
                item["sentence"]
                for item in selected
            )

            coverage_result = check_coverage(
                query=query,
                context=compressed_context,
                threshold=self.coverage_threshold
            )

            if coverage_result["passed"]:
                break

        return (
            selected,
            removed,
            coverage_result
        )

    def compress(
        self,
        query: str,
        context: str,
        token_budget: int
    ) -> dict:

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
        # Sentence splitting
        # -----------------------------------------------------

        sentences = split_sentences(
            context
        )

        if not sentences:

            return {
                "original_tokens": 0,
                "compressed_tokens": 0,
                "tokens_saved": 0,
                "compression_ratio": 0.0,
                "total_sentences": 0,
                "kept_sentences": 0,
                "removed_sentences": 0,
                "coverage": 1.0,
                "coverage_guard_passed": True,
                "coverage_guard_triggered": False,
                "compressed_context": "",
                "kept": [],
                "removed": [],
                "missing_concepts": []
            }

        # -----------------------------------------------------
        # Build candidates
        # -----------------------------------------------------

        candidates = self._build_candidates(
            query=query,
            sentences=sentences
        )

        # -----------------------------------------------------
        # Initial optimization
        # -----------------------------------------------------

        selected = self.optimizer.select(
            candidates=candidates,
            token_budget=token_budget
        )

        # IMPORTANT:
        #
        # Use sentence text rather than object identity.
        # This prevents duplicated bookkeeping when candidates
        # are copied by the optimizer.
        selected_sentences = {
            candidate["sentence"]
            for candidate in selected
        }

        removed = [
            candidate
            for candidate in candidates
            if candidate["sentence"]
            not in selected_sentences
        ]

        # -----------------------------------------------------
        # Coverage guard
        # -----------------------------------------------------

        before_guard_sentences = {
            candidate["sentence"]
            for candidate in selected
        }

        (
            selected,
            removed,
            coverage_result
        ) = self._coverage_guard(
            query=query,
            selected=selected,
            removed=removed
        )

        after_guard_sentences = {
            candidate["sentence"]
            for candidate in selected
        }

        coverage_guard_triggered = (
            before_guard_sentences
            !=
            after_guard_sentences
        )

        # -----------------------------------------------------
        # Rebuild removed list from final selected set.
        #
        # This guarantees:
        #
        # total = kept + removed
        #
        # with no duplicates.
        # -----------------------------------------------------

        final_selected_sentences = {
            candidate["sentence"]
            for candidate in selected
        }

        removed = [
            candidate
            for candidate in candidates
            if candidate["sentence"]
            not in final_selected_sentences
        ]

        # -----------------------------------------------------
        # Mark decisions
        # -----------------------------------------------------

        for candidate in selected:

            candidate["decision"] = (
                "KEEP"
            )

        for candidate in removed:

            candidate["decision"] = (
                "REMOVE"
            )

        # -----------------------------------------------------
        # Preserve original sentence order
        # -----------------------------------------------------

        sentence_positions = {
            sentence: index
            for index, sentence
            in enumerate(sentences)
        }

        selected.sort(
            key=lambda candidate:
            sentence_positions.get(
                candidate["sentence"],
                999999
            )
        )

        removed.sort(
            key=lambda candidate:
            sentence_positions.get(
                candidate["sentence"],
                999999
            )
        )

        # -----------------------------------------------------
        # Build compressed context
        # -----------------------------------------------------

        compressed_context = " ".join(
            candidate["sentence"]
            for candidate in selected
        )

        original_tokens = count_tokens(
            context
        )

        compressed_tokens = count_tokens(
            compressed_context
        )

        tokens_saved = (
            original_tokens
            -
            compressed_tokens
        )

        if original_tokens > 0:

            compression_ratio = (
                tokens_saved
                /
                original_tokens
            ) * 100

        else:

            compression_ratio = 0.0

        # -----------------------------------------------------
        # Final coverage check
        # -----------------------------------------------------

        final_coverage = check_coverage(
            query=query,
            context=compressed_context,
            threshold=self.coverage_threshold
        )

        # -----------------------------------------------------
        # Final result
        # -----------------------------------------------------

        return {
            "original_tokens": original_tokens,

            "compressed_tokens": compressed_tokens,

            "tokens_saved": tokens_saved,

            "compression_ratio": compression_ratio,

            "total_sentences": len(
                sentences
            ),

            "kept_sentences": len(
                selected
            ),

            "removed_sentences": len(
                removed
            ),

            "coverage": final_coverage[
                "coverage"
            ],

            "coverage_guard_passed": (
                final_coverage["passed"]
            ),

            "coverage_guard_triggered": (
                coverage_guard_triggered
            ),

            "compressed_context": (
                compressed_context
            ),

            "kept": selected,

            "removed": removed,

            "missing_concepts": (
                final_coverage[
                    "missing_concepts"
                ]
            )
        }