from compressor.relevance import RelevanceScorer


query = """
What caused the increase in renewable energy production
between 2020 and 2025?
"""


sentences = [
    "Renewable energy production increased significantly between 2020 and 2025.",
    "Solar generation increased by 42% during this period.",
    "Government incentives and lower solar panel costs contributed to this growth.",
    "The weather was generally stable during the period.",
    "It is important to note that renewable energy is becoming increasingly important.",
    "Several companies announced new projects during this period.",
]


print("\n===== CROSS-ENCODER TEST =====\n")


scorer = RelevanceScorer()


scores = scorer.score(
    query=query,
    sentences=sentences
)


for sentence, score in zip(
    sentences,
    scores
):

    print(
        f"Score: {score:.4f} | {sentence}"
    )


print("\n==============================\n")

print(
    f"Sentences scored: {len(scores)}"
)