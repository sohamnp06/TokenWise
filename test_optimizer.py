from compressor.optimizer import (
    TokenOptimizer
)


sentences = [
    "Solar production increased by 42% between 2020 and 2025.",
    "The weather was generally stable during the period.",
    "Government incentives contributed significantly to this growth.",
    "Solar energy generation experienced a 42% increase from 2020 to 2025.",
    "Several companies announced new projects during this period.",
]


# These are temporary test values.
# Later they will come directly from our
# RelevanceScorer, Evidence system, and
# RedundancyDetector.
relevance_scores = [
    0.95,
    0.10,
    0.70,
    0.90,
    0.30,
]

evidence_scores = [
    0.55,
    0.00,
    0.00,
    0.55,
    0.15,
]

redundancy_scores = [
    0.00,
    0.10,
    0.30,
    0.90,
    0.20,
]


print("\n===== TOKEN OPTIMIZER TEST =====\n")


optimizer = TokenOptimizer(
    evidence_weight=0.25,
    redundancy_weight=0.25
)


# ---------------------------------------------------------
# 1. Calculate final sentence scores
# ---------------------------------------------------------

final_scores = optimizer.calculate_scores(
    relevance_scores=relevance_scores,
    evidence_scores=evidence_scores,
    redundancy_scores=redundancy_scores
)


print("FINAL SENTENCE SCORES")
print("---------------------")


for sentence, score in zip(
    sentences,
    final_scores
):

    print(
        f"Score: {score:.4f}"
    )

    print(
        f"Sentence: {sentence}"
    )

    print()


# ---------------------------------------------------------
# 2. Calculate token values
# ---------------------------------------------------------

candidates = optimizer.calculate_token_values(
    sentences=sentences,
    scores=final_scores
)


print("TOKEN VALUES")
print("------------")


for candidate in candidates:

    print(
        f"Score:       "
        f"{candidate['score']:.4f}"
    )

    print(
        f"Token Cost:  "
        f"{candidate['token_cost']}"
    )

    print(
        f"Token Value: "
        f"{candidate['token_value']:.6f}"
    )

    print(
        f"Sentence:    "
        f"{candidate['sentence']}"
    )

    print()


# ---------------------------------------------------------
# 3. Budget-constrained selection
# ---------------------------------------------------------

token_budget = 40


selected = optimizer.select(
    candidates=candidates,
    token_budget=token_budget
)


print(
    f"SELECTED SENTENCES "
    f"(Budget = {token_budget} tokens)"
)

print(
    "------------------------------------"
)


total_tokens = 0


for index, candidate in enumerate(
    selected,
    start=1
):

    print(
        f"{index}. "
        f"{candidate['sentence']}"
    )

    print(
        f"   Tokens: "
        f"{candidate['token_cost']}"
    )

    print(
        f"   Value: "
        f"{candidate['token_value']:.6f}"
    )

    print()

    total_tokens += (
        candidate["token_cost"]
    )


print(
    f"Total selected tokens: "
    f"{total_tokens}"
)

print(
    f"Token budget:          "
    f"{token_budget}"
)


print("\n================================\n")
