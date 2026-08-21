from compressor.evidence import evidence_bonus


sentences = [
    "Solar generation increased by 42% between 2020 and 2025.",
    "Government incentives contributed significantly to this growth.",
    "It is important to note that renewable energy is becoming increasingly important.",
    "The model achieved 94.2% accuracy on the validation dataset.",
    "According to the report [12], production increased significantly.",
    "The weather was generally stable during the period.",
    "The company announced a new project in Mumbai.",
]


print("\n===== EVIDENCE BONUS TEST =====\n")


for sentence in sentences:

    score = evidence_bonus(sentence)

    print(
        f"Evidence: {score:.4f} | {sentence}"
    )


print("\n===============================\n")

print(
    f"Sentences evaluated: {len(sentences)}"
)