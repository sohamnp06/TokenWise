from compressor.optimizer import TokenOptimizer
from evaluation.metrics import count_tokens


print("\n")
print("=" * 70)
print("                    TOKEN OPTIMIZER TEST")
print("=" * 70)


sentences = [
    "Solar production increased by 42% between 2020 and 2025.",
    "The weather was generally stable during the period.",
    "Government incentives contributed significantly to this growth.",
    "Solar energy generation experienced a 42% increase from 2020 to 2025.",
    "Several companies announced new projects during this period."
]


relevance_scores = [
    1.00,
    0.00,
    0.17,
    0.81,
    0.29
]


evidence_scores = [
    0.55,
    0.00,
    0.00,
    0.55,
    0.00
]


redundancy_scores = [
    0.00,
    0.13,
    0.35,
    0.85,
    0.59
]


optimizer = TokenOptimizer()


# ---------------------------------------------------------
# Calculate final scores
# ---------------------------------------------------------

final_scores = optimizer.calculate_scores(
    relevance_scores=relevance_scores,
    evidence_scores=evidence_scores,
    redundancy_scores=redundancy_scores
)


print("\nFINAL SENTENCE SCORES")
print("---------------------")


for (
    sentence,
    score
) in zip(
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
# Calculate token values
# ---------------------------------------------------------

candidates = optimizer.calculate_token_values(
    sentences=sentences,
    scores=final_scores,
    evidence_scores=evidence_scores
)


print("TOKEN VALUES")
print("------------")


for candidate in candidates:

    print(
        f"Score:       "
        f"{candidate['score']:.4f}"
    )

    print(
        f"Evidence:    "
        f"{candidate['evidence']:.4f}"
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
# Selection
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

print("------------------------------------")


total_tokens = 0

for index, candidate in enumerate(
    selected,
    start=1
):

    total_tokens += candidate[
        "token_cost"
    ]

    print(
        f"{index}. "
        f"{candidate['sentence']}"
    )

    print(
        f"   Tokens: "
        f"{candidate['token_cost']}"
    )

    print(
        f"   Evidence: "
        f"{candidate['evidence']:.4f}"
    )

    print(
        f"   Value: "
        f"{candidate['token_value']:.6f}"
    )

    print(
        f"   Selection Value: "
        f"{candidate['selection_value']:.6f}"
    )

    print()


print(
    f"Total selected tokens: "
    f"{total_tokens}"
)

print(
    f"Token budget:          "
    f"{token_budget}"
)


# ---------------------------------------------------------
# Evidence preservation check
# ---------------------------------------------------------

evidence_sentence = (
    "Solar production increased by 42% "
    "between 2020 and 2025."
)

government_sentence = (
    "Government incentives contributed "
    "significantly to this growth."
)


selected_text = [
    candidate["sentence"]
    for candidate in selected
]


print("\nEVIDENCE PRESERVATION CHECK")
print("---------------------------")


if evidence_sentence in selected_text:

    print(
        "PASS: Numeric evidence sentence preserved."
    )

else:

    print(
        "WARNING: Numeric evidence sentence removed."
    )


if government_sentence in selected_text:

    print(
        "PASS: Government-incentive sentence preserved."
    )

else:

    print(
        "INFO: Government-incentive sentence not selected."
    )


print("\n")
print("=" * 70)
print("                     TEST COMPLETE")
print("=" * 70)
print()