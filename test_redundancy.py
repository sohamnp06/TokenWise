from compressor.redundancy import (
    RedundancyDetector
)


sentences = [
    "Solar production increased by 42% between 2020 and 2025.",
    "Between 2020 and 2025, solar energy generation grew by 42%.",
    "Government incentives contributed significantly to this growth.",
    "The weather was generally stable during the period.",
    "Solar energy generation experienced a 42% increase from 2020 to 2025.",
]


print("\n===== REDUNDANCY DETECTION TEST =====\n")


detector = RedundancyDetector()


# ---------------------------------------------------------
# 1. Calculate redundancy penalties
# ---------------------------------------------------------

penalties = detector.redundancy_penalties(
    sentences
)


print("REDUNDANCY PENALTIES")
print("--------------------")


for index, (
    sentence,
    penalty
) in enumerate(
    zip(
        sentences,
        penalties
    ),
    start=1
):

    print(
        f"{index}. Penalty: {penalty:.4f}"
    )

    print(
        f"   {sentence}"
    )

    print()


# ---------------------------------------------------------
# 2. Find highly redundant sentence pairs
# ---------------------------------------------------------

print("HIGHLY REDUNDANT PAIRS")
print("----------------------")


pairs = detector.find_redundant_pairs(
    sentences,
    threshold=0.80
)


if not pairs:

    print(
        "No redundant pairs found."
    )

else:

    for index, pair in enumerate(
        pairs,
        start=1
    ):

        print(
            f"\nPair {index}"
        )

        print(
            f"Similarity: "
            f"{pair['similarity']:.4f}"
        )

        print(
            f"A: {pair['sentence_a']}"
        )

        print(
            f"B: {pair['sentence_b']}"
        )


print("\n====================================\n")

print(
    f"Total sentences: {len(sentences)}"
)

print(
    f"Redundant pairs found: {len(pairs)}"
)