from compressor.pipeline import TokenDiet


query = """
What caused the increase in renewable energy
production between 2020 and 2025?
"""


context = """
Renewable energy production increased significantly
between 2020 and 2025.

Solar generation increased by 42% during this period.

Government incentives and lower solar panel costs
were major contributors to this growth.

The weather was generally stable during the period.

It is important to note that renewable energy is
becoming increasingly important around the world.

Solar energy generation experienced a 42% increase
from 2020 to 2025.

Several companies announced new renewable energy
projects during this period.

Wind energy production also increased during the
same period.

This demonstrates the growing importance of renewable
energy for global electricity generation.
"""


print("\n")
print("=" * 70)
print("                    TOKENWISE PIPELINE TEST")
print("=" * 70)


# ---------------------------------------------------------
# Initialize TokenWise
# ---------------------------------------------------------

print("\nInitializing TokenWise...\n")

token_diet = TokenDiet(
    evidence_weight=0.25,
    redundancy_weight=0.25,
    coverage_threshold=0.80
)


# ---------------------------------------------------------
# Run compression
# ---------------------------------------------------------

print("\nRunning compression...\n")

result = token_diet.compress(
    query=query,
    context=context,
    token_budget=80
)


# ---------------------------------------------------------
# Basic metrics
# ---------------------------------------------------------

print("=" * 70)
print("                         METRICS")
print("=" * 70)

print(
    f"\nOriginal tokens:     "
    f"{result['original_tokens']}"
)

print(
    f"Compressed tokens:   "
    f"{result['compressed_tokens']}"
)

print(
    f"Tokens saved:        "
    f"{result['tokens_saved']}"
)

print(
    f"Compression ratio:   "
    f"{result['compression_ratio']:.2f}%"
)

print(
    f"Total sentences:     "
    f"{result['total_sentences']}"
)

print(
    f"Kept sentences:      "
    f"{result['kept_sentences']}"
)

print(
    f"Removed sentences:   "
    f"{result['removed_sentences']}"
)

print(
    f"Query coverage:      "
    f"{result['coverage']:.2%}"
)

print(
    f"Coverage guard:      "
    f"{'PASS' if result['coverage_guard_passed'] else 'FAIL'}"
)

print(
    f"Guard triggered:     "
    f"{result['coverage_guard_triggered']}"
)


# ---------------------------------------------------------
# Compressed context
# ---------------------------------------------------------

print("\n")
print("=" * 70)
print("                    COMPRESSED CONTEXT")
print("=" * 70)

print()

print(
    result["compressed_context"]
)


# ---------------------------------------------------------
# Kept sentences
# ---------------------------------------------------------

print("\n")
print("=" * 70)
print("                         KEPT")
print("=" * 70)

print()

for index, candidate in enumerate(
    result["kept"],
    start=1
):

    print(
        f"{index}. "
        f"{candidate['sentence']}"
    )

    print(
        f"   Relevance:  "
        f"{candidate['relevance']:.4f}"
    )

    print(
        f"   Evidence:   "
        f"{candidate['evidence']:.4f}"
    )

    print(
        f"   Redundancy: "
        f"{candidate['redundancy']:.4f}"
    )

    print(
        f"   Score:      "
        f"{candidate['score']:.4f}"
    )

    print(
        f"   Tokens:     "
        f"{candidate['token_cost']}"
    )

    print(
        f"   Token Value:"
        f" {candidate['token_value']:.6f}"
    )

    print(
        f"   Decision:   "
        f"{candidate['decision']}"
    )

    print()


# ---------------------------------------------------------
# Removed sentences
# ---------------------------------------------------------

print("=" * 70)
print("                        REMOVED")
print("=" * 70)

print()

for index, candidate in enumerate(
    result["removed"],
    start=1
):

    print(
        f"{index}. "
        f"{candidate['sentence']}"
    )

    print(
        f"   Relevance:  "
        f"{candidate['relevance']:.4f}"
    )

    print(
        f"   Evidence:   "
        f"{candidate['evidence']:.4f}"
    )

    print(
        f"   Redundancy: "
        f"{candidate['redundancy']:.4f}"
    )

    print(
        f"   Score:      "
        f"{candidate['score']:.4f}"
    )

    print(
        f"   Tokens:     "
        f"{candidate['token_cost']}"
    )

    print(
        f"   Token Value:"
        f" {candidate['token_value']:.6f}"
    )

    print(
        f"   Decision:   "
        f"{candidate['decision']}"
    )

    print()


# ---------------------------------------------------------
# Missing concepts
# ---------------------------------------------------------

print("=" * 70)
print("                     MISSING CONCEPTS")
print("=" * 70)

print()

if result["missing_concepts"]:

    for concept in result["missing_concepts"]:

        print(
            f"- {concept}"
        )

else:

    print(
        "None"
    )


print("\n")
print("=" * 70)
print("                     TEST COMPLETE")
print("=" * 70)
print()